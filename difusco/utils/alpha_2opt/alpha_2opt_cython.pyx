# alpha_2opt_cython.pyx

import numpy as np
cimport numpy as np
import networkx as nx
from libc.math cimport sqrt

# C-level type definitions for clarity and consistency
ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

# This is a C-level function for calculating a full distance matrix.
# It operates on memoryviews for high performance.
cdef np.ndarray[DTYPE_t, ndim=2] _compute_dist_matrix(np.ndarray[DTYPE_t, ndim=2] points):
    cdef int N = points.shape[0]
    cdef int M = points.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] dist_matrix = np.empty((N, N), dtype=np.float64)
    cdef int i, j, k
    cdef double dist_sq, diff

    for i in range(N):
        dist_matrix[i, i] = 0.0
        for j in range(i + 1, N):
            dist_sq = 0.0
            for k in range(M):
                diff = points[i, k] - points[j, k]
                dist_sq += diff * diff
            dist_matrix[i, j] = dist_matrix[j, i] = sqrt(dist_sq)
            
    return dist_matrix

# Cython implementation of the 2-opt swap.
# It takes the tour as a C-typed memoryview for fast indexing.
cdef bint _two_opt(list tour, np.ndarray[DTYPE_t, ndim=2] dist, int max_iter):
    cdef int N = len(tour)
    cdef bint improved = True
    cdef int count = 0
    cdef int i, j, k
    cdef int a, b, c, d
    
    while improved and count < max_iter:
        improved = False
        for i in range(1, N - 2):
            for j in range(i + 1, N):
                if j == i + 1: continue
                # In Cython, we use list indexing which is fast.
                a = tour[i - 1]
                b = tour[i]
                c = tour[j - 1]
                d = tour[j]
                
                if dist[a, b] + dist[c, d] > dist[a, c] + dist[b, d]:
                    # Reverse the sublist in place
                    sublist = tour[i:j]
                    sublist.reverse()
                    tour[i:j] = sublist
                    improved = True
        count += 1
    return improved

# The main Cython function, callable from Python.
cpdef list alpha_2opt_heuristic_cython(np.ndarray[DTYPE_t, ndim=2] points, int k=10, int max_iter=1000):
    """
    Cython implementation of the alpha-2-opt heuristic for TSP.
    """
    cdef int N = points.shape[0]
    # FIX: Moved all C-level variable declarations to the top of the function scope.
    cdef int i, j, l, u, v, unvisited_city
    
    # --- Part 1: Compute Alpha-Nearness and Distance Matrix ---
    # The NetworkX part still uses Python objects, as it's a Python library.
    cdef np.ndarray[DTYPE_t, ndim=2] dist = _compute_dist_matrix(points)
    
    G = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            G.add_edge(i, j, weight=dist[i, j])
            
    mst = nx.minimum_spanning_tree(G)
    cdef np.ndarray[DTYPE_t, ndim=2] alpha = np.full((N, N), np.inf, dtype=np.float64)
    
    for i in range(N):
        for j in range(i + 1, N):
            if mst.has_edge(i, j):
                alpha[i, j] = alpha[j, i] = 0.0
            else:
                try:
                    path = nx.shortest_path(mst, source=i, target=j, weight='weight')
                    max_edge = 0.0
                    for l in range(len(path) - 1):
                        u, v = path[l], path[l+1]
                        if dist[u, v] > max_edge:
                            max_edge = dist[u, v]
                    alpha[i, j] = alpha[j, i] = dist[i, j] - max_edge
                except nx.NetworkXNoPath:
                    continue

    # --- Part 2: Build Candidate Set ---
    # Looping and sorting are faster in Cython.
    candidates = []
    cdef np.ndarray[ITYPE_t, ndim=1] sorted_neighbors
    for i in range(N):
        sorted_neighbors = np.argsort(alpha[i])
        candidates.append([j for j in sorted_neighbors if j != i][:k])

    # --- Part 3: Initial Tour Construction ---
    cdef list tour = [0]
    cdef np.ndarray[np.uint8_t, ndim=1] visited = np.zeros(N, dtype=np.uint8)
    visited[0] = 1
    cdef int current_city = 0
    cdef int next_city = -1
    cdef double min_dist
    
    while len(tour) < N:
        next_city = -1
        for neighbor in candidates[current_city]:
            if not visited[neighbor]:
                next_city = neighbor
                break
        
        if next_city == -1:
            # Fallback if all candidates are visited
            min_dist = np.inf
            # FIX: The `cdef` declaration for `unvisited_city` was removed from here.
            for unvisited_city in range(N):
                if not visited[unvisited_city]:
                    if dist[current_city, unvisited_city] < min_dist:
                        min_dist = dist[current_city, unvisited_city]
                        next_city = unvisited_city

        tour.append(next_city)
        visited[next_city] = 1
        current_city = next_city
        
    # --- Part 4: 2-Opt Improvement ---
    _two_opt(tour, dist, max_iter)
    
    return tour
