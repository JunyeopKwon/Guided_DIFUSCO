# farthest_cython.pyx

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# C-level type definitions for performance
ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

# Re-using the efficient C-level distance matrix function
cdef np.ndarray[DTYPE_t, ndim=2] _compute_dist_matrix_cy(np.ndarray[DTYPE_t, ndim=2] points):
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

# The main Cython function, callable from Python
cpdef list farthest_insertion_heuristic_cython(np.ndarray[DTYPE_t, ndim=2] points):
    """
    Cython implementation of the farthest insertion heuristic for TSP.
    """
    cdef int N = points.shape[0]
    
    # --- Part 1: Initialization ---
    cdef np.ndarray[DTYPE_t, ndim=2] dist = _compute_dist_matrix_cy(points)
    
    # FIX: Avoid np.unravel_index entirely.
    # Calculate the 2D index from the flat index returned by np.argmax.
    # This is a robust way to avoid the type conversion error.
    cdef ITYPE_t flat_index = np.argmax(dist)
    cdef int start_i = <int>(flat_index / N)
    cdef int start_j = <int>(flat_index % N)
    
    cdef list tour = [start_i, start_j]
    # Use a boolean array for 'visited' for fast C-level lookups
    cdef np.ndarray[np.uint8_t, ndim=1] visited = np.zeros(N, dtype=np.uint8)
    visited[start_i] = 1
    visited[start_j] = 1
    cdef int visited_count = 2

    # --- Part 2 & 3: Iterative Insertion (Optimized Section) ---
    cdef double max_dist, min_dist_to_tour, best_increase, increase
    cdef int farthest, best_pos
    cdef int k, t, idx, a, b

    while visited_count < N:
        # Step 2: Find the unvisited point farthest from the tour
        max_dist = -1.0
        farthest = -1
        for k in range(N):
            if visited[k]:
                continue
            
            min_dist_to_tour = np.inf
            for t_node in tour:
                t = t_node
                if dist[k, t] < min_dist_to_tour:
                    min_dist_to_tour = dist[k, t]
            
            if min_dist_to_tour > max_dist:
                max_dist = min_dist_to_tour
                farthest = k

        # Step 3: Find the best position to insert the farthest point
        best_increase = np.inf
        best_pos = -1
        for idx in range(len(tour)):
            a = tour[idx]
            b = tour[(idx + 1) % len(tour)] # Handle loop around
            
            increase = dist[a, farthest] + dist[farthest, b] - dist[a, b]
            
            if increase < best_increase:
                best_increase = increase
                best_pos = idx + 1

        tour.insert(best_pos, farthest)
        visited[farthest] = 1
        visited_count += 1
        
    return tour