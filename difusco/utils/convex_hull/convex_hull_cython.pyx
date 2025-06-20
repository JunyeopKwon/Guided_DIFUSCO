# convex_hull_cython.pyx

import numpy as np
cimport numpy as np
from scipy.spatial import ConvexHull
from libc.math cimport sqrt

# C-level type definitions
ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

# A C-level function to compute the distance matrix efficiently.
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

# The main Cython function callable from Python.
cpdef list convex_hull_insertion_heuristic_cython(np.ndarray[DTYPE_t, ndim=2] points):
    """
    Cython implementation of the convex hull insertion heuristic for TSP.
    """
    cdef int N = points.shape[0]
    
    # --- Part 1: Distance Matrix and Convex Hull ---
    # The distance matrix calculation is moved to a fast C function.
    cdef np.ndarray[DTYPE_t, ndim=2] dist = _compute_dist_matrix_cy(points)
    
    # ConvexHull from SciPy is already highly optimized. We call it directly.
    hull = ConvexHull(points)
    
    # We work with Python lists for the tour as insertions are efficient.
    cdef list tour = list(hull.vertices)
    cdef set visited = set(tour)
    cdef list remaining = [i for i in range(N) if i not in visited]

    # --- Part 2: Insertion Heuristic (Optimized Section) ---
    cdef double best_increase, increase
    cdef int best_insert, best_pos
    cdef int p, i, a, b
    cdef int tour_len
    
    while remaining:
        best_increase = np.inf
        best_insert = -1
        best_pos = -1
        tour_len = len(tour)

        # This nested loop is where Cython provides the most benefit.
        for p_val in remaining:
            p = p_val
            for i in range(tour_len):
                a = tour[i]
                b = tour[(i + 1) % tour_len]
                
                increase = dist[a, p] + dist[p, b] - dist[a, b]
                
                if increase < best_increase:
                    best_increase = increase
                    best_insert = p
                    best_pos = (i + 1)

        tour.insert(best_pos, best_insert)
        remaining.remove(best_insert)
        
    return tour
