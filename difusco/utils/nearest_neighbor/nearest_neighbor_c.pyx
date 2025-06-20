# dist/nearest_neighbor_c.pyx

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# We declare the type of the numpy array to be a memoryview.
# This allows for fast, direct access to the memory buffer of the array.
# double[:, :] specifies a 2D array of doubles.
ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

# This is a C-level function for calculating Euclidean distance.
# Using 'cdef' makes it a C function, which is faster than a Python function.
# The 'inline' keyword suggests the compiler to insert this function's code
# where it is called, avoiding function call overhead.
cdef inline double euclidean_dist_sq(double[:] p1, double[:] p2):
    cdef double dist_sq = 0.0
    cdef int i
    for i in range(p1.shape[0]):
        dist_sq += (p1[i] - p2[i]) * (p1[i] - p2[i])
    return dist_sq

# This is the main Cython function.
# 'cpdef' creates both a C-level function and a Python wrapper,
# so it can be called from both Python and other Cython code.
cpdef np.ndarray[ITYPE_t, ndim=1] nearest_neighbor_c(np.ndarray[DTYPE_t, ndim=2] points):
    """
    Calculates a tour using the nearest neighbor heuristic.
    This is the Cython implementation.
    """
    # Type declarations for performance
    cdef int N = points.shape[0]
    cdef set unvisited = set(range(N))
    cdef list tour = []
    cdef int last, current_city, next_city
    cdef double min_dist_sq, dist_sq

    # Start with the first point
    current_city = unvisited.pop()
    tour.append(current_city)

    while unvisited:
        last = tour[-1]
        min_dist_sq = -1.0
        
        # Find the nearest unvisited city
        # We iterate through the set directly for efficiency
        for city in unvisited:
            dist_sq = euclidean_dist_sq(points[last], points[city])
            if min_dist_sq == -1.0 or dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                next_city = city
        
        unvisited.remove(next_city)
        tour.append(next_city)
    
    # Complete the tour by returning to the start
    tour.append(tour[0])
    
    return np.array(tour, dtype=np.intp)
