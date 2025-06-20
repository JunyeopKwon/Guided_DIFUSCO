import os
import warnings
from multiprocessing import Pool

import numpy as np
import scipy.sparse
import scipy.spatial
import torch
import torch.nn.functional as F
from utils.cython_merge.cython_merge import merge_cython
import networkx as nx
from scipy.spatial import distance_matrix


def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
  iterator = 0
  tour = tour.copy()
  with torch.inference_mode():
    cuda_points = torch.from_numpy(points).to(device)
    cuda_tour = torch.from_numpy(tour).to(device)
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, len(points))

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break
    tour = cuda_tour.cpu().numpy()
  return tour, iterator


def numpy_merge(points, adj_mat):
  dists = np.linalg.norm(points[:, None] - points, axis=-1)

  components = np.zeros((adj_mat.shape[0], 2)).astype(int)
  components[:] = np.arange(adj_mat.shape[0])[..., None]
  real_adj_mat = np.zeros_like(adj_mat)
  merge_iterations = 0
  for edge in (-adj_mat / dists).flatten().argsort():
    merge_iterations += 1
    a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
    if not (a in components and b in components):
      continue
    ca = np.nonzero((components == a).sum(1))[0][0]
    cb = np.nonzero((components == b).sum(1))[0][0]
    if ca == cb:
      continue
    cca = sorted(components[ca], key=lambda x: x == a)
    ccb = sorted(components[cb], key=lambda x: x == b)
    newc = np.array([[cca[0], ccb[0]]])
    m, M = min(ca, cb), max(ca, cb)
    real_adj_mat[a, b] = 1
    components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
    if len(components) == 1:
      break
  real_adj_mat[components[0, 1], components[0, 0]] = 1
  real_adj_mat += real_adj_mat.T
  return real_adj_mat, merge_iterations


def cython_merge(points, adj_mat):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
    real_adj_mat = np.asarray(real_adj_mat)
  return real_adj_mat, merge_iterations


def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
  """
  To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
  procedure.
  • Initialize extracted tour with an empty graph with N vertices.
  • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
  multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
  • For each edge (i, j) in the list:
    – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
    – If inserting (i, j) results in a graph with cycles (of length < N), continue.
    – Otherwise, insert (i, j) into the tour.
  • Return the extracted tour.
  """
  splitted_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

  if not sparse_graph:
    splitted_adj_mat = [
        adj_mat[0] + adj_mat[0].T for adj_mat in splitted_adj_mat
    ]
  else:
    splitted_adj_mat = [
        scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[0], edge_index_np[1])),
        ).toarray() + scipy.sparse.coo_matrix(
            (adj_mat, (edge_index_np[1], edge_index_np[0])),
        ).toarray() for adj_mat in splitted_adj_mat
    ]

  splitted_points = [
      np_points for _ in range(parallel_sampling)
  ]

  if np_points.shape[0] > 1000 and parallel_sampling > 1:
    with Pool(parallel_sampling) as p:
      results = p.starmap(
          cython_merge,
          zip(splitted_points, splitted_adj_mat),
      )
  else:
    results = [
        cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(splitted_points, splitted_adj_mat)
    ]

  splitted_real_adj_mat, splitted_merge_iterations = zip(*results)

  tours = []
  for i in range(parallel_sampling):
    tour = [0]
    while len(tour) < splitted_adj_mat[i].shape[0] + 1:
      n = np.nonzero(splitted_real_adj_mat[i][tour[-1]])[0]
      if len(tour) > 1:
        n = n[n != tour[-2]]
      tour.append(n.max())
    tours.append(tour)

  merge_iterations = np.mean(splitted_merge_iterations)
  return tours, merge_iterations

def nearest_neighbor_tour(points):
    N = len(points)
    unvisited = set(range(N))
    tour = [unvisited.pop()]  # 임의의 시작점
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda j: np.linalg.norm(points[last] - points[j]))
        unvisited.remove(next_city)
        tour.append(next_city)
    tour.append(tour[0])  # 돌아오기
    return np.array(tour, dtype=int)

def alpha_2opt_heuristic(points, k=10, max_iter=1000):

    def compute_alpha_nearness(points):
        N = len(points)
        dist = distance_matrix(points, points)
        G = nx.Graph()
        for i in range(N):
            for j in range(i + 1, N):
                G.add_edge(i, j, weight=dist[i, j])
        mst = nx.minimum_spanning_tree(G)
        alpha = np.full((N, N), np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                if mst.has_edge(i, j):
                    alpha[i, j] = alpha[j, i] = 0
                else:
                    try:
                        path = nx.shortest_path(mst, source=i, target=j, weight='weight')
                        max_edge = max(dist[path[k], path[k + 1]] for k in range(len(path) - 1))
                        alpha[i, j] = alpha[j, i] = dist[i, j] - max_edge
                    except nx.NetworkXNoPath:
                        continue
        return alpha, dist

    def initial_tour(points, candidates):
        N = len(points)
        visited = [False] * N
        tour = [0]
        visited[0] = True
        current = 0
        while len(tour) < N:
            next_city = None
            for j in candidates[current]:
                if not visited[j]:
                    next_city = j
                    break
            if next_city is None:
                unvisited = [i for i in range(N) if not visited[i]]
                next_city = min(unvisited, key=lambda j: dist[current][j])
            tour.append(next_city)
            visited[next_city] = True
            current = next_city
        return tour

    def two_opt(tour):
        improved = True
        count = 0
        while improved and count < max_iter:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    if j - i == 1: continue
                    a, b = tour[i - 1], tour[i]
                    c, d = tour[j], tour[(j + 1) % len(tour)]
                    if dist[a][b] + dist[c][d] > dist[a][c] + dist[b][d]:
                        tour[i:j + 1] = tour[i:j + 1][::-1]
                        improved = True
            count += 1
        return tour

    # -- Main execution --
    alpha, dist = compute_alpha_nearness(points)

    # Build candidate set
    candidates = []
    for i in range(len(points)):
        sorted_neighbors = np.argsort(alpha[i])
        candidates.append([j for j in sorted_neighbors if j != i][:k])

    tour = initial_tour(points, candidates)
    tour = two_opt(tour)
    return tour


def dummy_heuristic(approx_algo, points):
    # 1) NN 으로 초기가
    tour = approx_algo(points)
    # 2) 1회 2-opt 수행 (utils.tsp_utils.batched_two_opt_torch 활용 가능)
    improved, _ = batched_two_opt_torch(
        points.astype("float64"),
        tour[None, :],
        max_iterations=1,
        device="cpu"
    )
    return improved[0]  # (N+1,) 배열


def tour_to_adj(tour):
    N = int(np.max(tour)) + 1   # 원래 도시 개수
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i+1]
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


# def apply_forward_diffusion(adj_np, t_noise):
#     # adj_np: (N,N) binary
#     x0 = torch.tensor(adj_np, dtype=torch.long).unsqueeze(0)           # [1,N,N]
#     x0_onehot = F.one_hot(x0, num_classes=2).float()                   # [1,N,N,2]
#     # returns xt with categorical noise
#     xt = self.diffusion.sample(x0_onehot, np.array([t_noise]))       # [1,N,N] float(0/1)
#     return xt.squeeze(0)                                              # [N,N]

class TSPEvaluator(object):
  def __init__(self, points):
    self.dist_mat = scipy.spatial.distance_matrix(points, points)

  def evaluate(self, route):
    total_cost = 0
    for i in range(len(route) - 1):
      total_cost += self.dist_mat[route[i], route[i + 1]]
    return total_cost
