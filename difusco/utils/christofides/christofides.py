import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx # For graph operations, MWPM, and Eulerian circuit

def christofides(points_array):
    """
    Calculates an approximate solution to the Traveling Salesperson Problem (TSP)
    using the Christofides 1.5-approximation algorithm.

    Args:
        points_array (np.ndarray): A NumPy array of shape (N, D) where N is the
                                   number of points and D is the dimension of
                                   the points (e.g., 2 for 2D points).

    Returns:
        np.ndarray: An array of node indices representing the TSP tour,
                    starting and ending at the same node.
                    Returns an empty array if N < 1, or a simple tour if N < 3.
    """
    N = points_array.shape[0]

    if N == 0:
        return np.array([], dtype=int)
    if N == 1:
        return np.array([0, 0], dtype=int) # Tour is just the point itself, returning to it
    if N == 2:
        return np.array([0, 1, 0], dtype=int) # Simple tour for two points

    # 1. Create a complete graph G with points as nodes and Euclidean distances as edge weights.
    #    For simplicity, we'll work with a distance matrix.
    dist_matrix = squareform(pdist(points_array, metric='euclidean'))

    # 2. Find a Minimum Spanning Tree (MST) T of G.
    #    scipy.sparse.csgraph.minimum_spanning_tree returns a CSR matrix.
    mst_sparse = minimum_spanning_tree(dist_matrix)
    
    # Convert MST to a NetworkX graph for easier manipulation
    G_mst = nx.Graph()
    for i in range(N):
        G_mst.add_node(i)
    
    # Add edges from the sparse MST matrix
    mst_sources, mst_targets = mst_sparse.nonzero()
    for i in range(len(mst_sources)):
        u, v = mst_sources[i], mst_targets[i]
        weight = dist_matrix[u, v]
        G_mst.add_edge(u, v, weight=weight)

    # 3. Identify the set O of vertices with an odd degree in T.
    odd_degree_nodes = []
    for node, degree in G_mst.degree():
        if degree % 2 != 0:
            odd_degree_nodes.append(node)

    # 4. Find a Minimum Weight Perfect Matching (MWPM) M on the subgraph G_O induced by O.
    #    Create a subgraph in NetworkX consisting only of the odd-degree nodes.
    #    The edges in this subgraph should have weights from the original distance matrix.
    
    matching_edges = [] # To store edges from MWPM
    if odd_degree_nodes: # MWPM is only needed if there are odd degree nodes
        G_odd_subgraph = nx.Graph()
        for i in range(len(odd_degree_nodes)):
            for j in range(i + 1, len(odd_degree_nodes)):
                u, v = odd_degree_nodes[i], odd_degree_nodes[j]
                # For min_weight_matching, positive weights are costs (distances)
                G_odd_subgraph.add_edge(u, v, weight=dist_matrix[u, v])
        
        # nx.min_weight_matching returns a set of edges (tuples of nodes)
        # For some versions, it might return a dict. We expect a set of 2-tuples.
        min_matching_set = nx.min_weight_matching(G_odd_subgraph, weight='weight')

        for u, v in list(min_matching_set): # list() to handle potential set modification issues if any
            # In some NetworkX versions, min_weight_matching might return edges as (u, v, data_dict)
            # or just (u,v) if no data_dict. We only need u,v.
            matching_edges.append((u,v))


    # 5. Combine the edges of M and T to form a multigraph H (T U M).
    #    All vertices in H will have an even degree.
    H_multigraph = nx.MultiGraph() # Use MultiGraph to allow parallel edges
    H_multigraph.add_nodes_from(range(N))

    # Add MST edges to H
    for u, v, data in G_mst.edges(data=True):
        H_multigraph.add_edge(u, v, weight=data['weight'])

    # Add MWPM edges to H
    for u, v in matching_edges:
        H_multigraph.add_edge(u, v, weight=dist_matrix[u, v])
        
    # 6. Find an Eulerian circuit in H.
    #    NetworkX can find an Eulerian circuit if one exists.
    #    An Eulerian circuit exists because all nodes in H have even degrees.
    #    We need a starting node. Any node will do if the graph is connected.
    #    Christofides assumes a complete graph initially, so H should be connected.
    start_node_euler = 0 # Default start
    if not H_multigraph.nodes: # Handle empty graph case if it somehow occurs
         return np.array([0,0], dtype=int) if N > 0 else np.array([], dtype=int)

    if not H_multigraph.edges: # If graph has nodes but no edges (e.g., N=1)
        if N > 0:
            return np.array([0,0], dtype=int) # Tour is just the single node
        else:
            return np.array([], dtype=int)


    # Ensure the graph is connected before attempting Eulerian circuit
    # For Christofides, H should be connected if N > 0.
    # If N=1, H_multigraph has 1 node, 0 edges. nx.is_eulerian might fail or give unexpected results.
    # nx.eulerian_circuit needs a source node.
    
    # Find a node with edges to start the Eulerian circuit if possible
    source_for_euler = 0
    for node in H_multigraph.nodes():
        if H_multigraph.degree(node) > 0:
            source_for_euler = node
            break
            
    eulerian_circuit_edges = list(nx.eulerian_circuit(H_multigraph, source=source_for_euler))
    
    # The circuit is a list of edges (u,v). We need a path of nodes.
    eulerian_path_nodes = [eulerian_circuit_edges[0][0]] # Start with the first node of the first edge
    for u, v in eulerian_circuit_edges:
        eulerian_path_nodes.append(v) # Add the second node of each edge

    # 7. Convert the Eulerian circuit into a Hamiltonian circuit (the TSP tour)
    #    by shortcutting (removing repeated vertices).
    tsp_tour_nodes = []
    visited_nodes = set()
    for node in eulerian_path_nodes:
        if node not in visited_nodes:
            tsp_tour_nodes.append(node)
            visited_nodes.add(node)
            
    # Add the starting node to the end to complete the tour
    if tsp_tour_nodes: # Ensure the tour is not empty
        tsp_tour_nodes.append(tsp_tour_nodes[0])
    elif N > 0: # Fallback for very small N or if eulerian circuit was trivial
        tsp_tour_nodes = [0,0]


    return np.array(tsp_tour_nodes, dtype=int)

def calculate_tour_length(points_array, tour_indices):
    """Calculates the total length of a tour."""
    length = 0.0
    if len(tour_indices) < 2:
        return 0.0
    for i in range(len(tour_indices) - 1):
        p1_idx = tour_indices[i]
        p2_idx = tour_indices[i+1]
        length += np.linalg.norm(points_array[p1_idx] - points_array[p2_idx])
    return length

# Your nearest_neighbor_tour function (for comparison or other uses)
def nearest_neighbor_tour(points_arr):
    N = len(points_arr)
    if N == 0:
        return np.array([], dtype=int)
    if N == 1:
        return np.array([0,0], dtype=int)

    unvisited = set(range(N))
    # Start at node 0, or pop an arbitrary one. For consistency:
    current_node = 0 
    unvisited.remove(current_node)
    tour = [current_node]
    
    while unvisited:
        last_node_in_tour = tour[-1]
        # Find the nearest unvisited neighbor
        min_dist = float('inf')
        next_city_candidate = -1
        for city_idx in unvisited:
            dist = np.linalg.norm(points_arr[last_node_in_tour] - points_arr[city_idx])
            if dist < min_dist:
                min_dist = dist
                next_city_candidate = city_idx
        
        if next_city_candidate != -1: # Should always find one if unvisited is not empty
            unvisited.remove(next_city_candidate)
            tour.append(next_city_candidate)
        else: # Should not happen if unvisited is not empty and graph is complete
            break 
            
    tour.append(tour[0])  # Return to start
    return np.array(tour, dtype=int)

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


if __name__ == '__main__':
    # Example Usage
    num_points = 500 # Try with 5, 10, 50, 100
    # For reproducibility, set a seed
    np.random.seed(42)
    points = np.random.rand(num_points, 2) * 100  # N points in a 100x100 square

    print(f"Generated {num_points} points:\n", points)
    print("-" * 30)

    print("Running Christofides algorithm (Python)...")
    christofides_tour = christofides(points)
    christofides_length = calculate_tour_length(points, christofides_tour)
    print(f"Christofides tour: {christofides_tour}")
    print(f"Christofides tour length: {christofides_length:.2f}")
    print("-" * 30)

    print("Running 2-opt algorithm (Python)...")
    improved, _ = batched_two_opt_torch(
        points.astype("float64"),
        christofides_tour[None, :],
        max_iterations=1,
        device="cpu"
    )
    two_opt_length = calculate_tour_length(points, improved[0])
    print(f"2-opt improved tour: {improved[0]}")
    print(f"2-opt tour length: {two_opt_length:.2f}")

    print("Running Nearest Neighbor algorithm (Python)...")
    nn_tour = nearest_neighbor_tour(points)
    nn_length = calculate_tour_length(points, nn_tour)
    print(f"Nearest Neighbor tour: {nn_tour}")
    print(f"Nearest Neighbor tour length: {nn_length:.2f}")
    print("-" * 30)

    print("Running 2-opt algorithm (Python)...")
    improved, _ = batched_two_opt_torch(
        points.astype("float64"),
        nn_tour[None, :],
        max_iterations=1,
        device="cpu"
    )
    two_opt_length = calculate_tour_length(points, improved[0])
    print(f"2-opt improved tour: {improved[0]}")
    print(f"2-opt tour length: {two_opt_length:.2f}")

    # Example with a very small number of points
    # print("\nTesting with 3 points:")
    # points_small = np.array([[0,0], [1,1], [0,1]])
    # christofides_tour_small = christofides(points_small)
    # christofides_length_small = calculate_tour_length(points_small, christofides_tour_small)
    # print(f"Christofides tour (3 points): {christofides_tour_small}") # Expected e.g. [0, 2, 1, 0] or similar
    # print(f"Christofides length (3 points): {christofides_length_small:.2f}")

    # print("\nTesting with 1 point:")
    # points_one = np.array([[5,5]])
    # christofides_tour_one = christofides(points_one)
    # christofides_length_one = calculate_tour_length(points_one, christofides_tour_one)
    # print(f"Christofides tour (1 point): {christofides_tour_one}")
    # print(f"Christofides length (1 point): {christofides_length_one:.2f}")


# **Key components and library usage:**

# 1.  **Distance Matrix:** `scipy.spatial.distance.pdist` and `squareform` are used to efficiently calculate the all-pairs Euclidean distance matrix.
# 2.  **Minimum Spanning Tree (MST):** `scipy.sparse.csgraph.minimum_spanning_tree` finds the MST from the distance matrix. The result is converted to a `networkx.Graph` for easier degree calculation and manipulation.
# 3.  **Odd-Degree Vertices:** Standard iteration over the MST graph's degrees.
# 4.  **Minimum Weight Perfect Matching (MWPM):**
#     * A subgraph `G_odd_subgraph` is created in NetworkX containing only the odd-degree vertices and the original distances between them.
#     * `networkx.min_weight_matching` is called on this subgraph. This function implements Edmonds' Blossom algorithm (or a variant), which is complex but crucial for the 3/2 approximation guarantee.
# 5.  **Eulerian Multigraph:** Edges from the MST and the MWPM are combined into a `networkx.MultiGraph` (allowing parallel edges, which might arise if an edge is in both MST and MWPM).
# 6.  **Eulerian Circuit:** `networkx.eulerian_circuit` finds an Eulerian circuit in the multigraph. All nodes in this combined graph are guaranteed to have even degrees.
# 7.  **Hamiltonian Circuit (Shortcut):** The sequence of nodes from the Eulerian circuit is processed to remove repeated visits, forming the final TSP tour.

# This pure Python version is conceptually clear and relies on well-tested libraries for the heavy lifting. For very large problem instances, the Cython version you were working on would offer significant speedups, especially if the MWPM step (which is $O(N^3)$ in the number of odd vertices) becomes the bottlene