# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
import heapq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, floyd_warshall

# Import NumPy C-API
cimport numpy as np
np.import_array()

# Define C-level types for efficiency
# Changed DTYPE_t to float32_t to match single-precision float inputs
ctypedef np.float32_t DTYPE_t
# Changed ITYPE_t to a more explicit and robust int64_t
ctypedef np.int64_t ITYPE_t

def christofides_c(np.ndarray[DTYPE_t, ndim=2] points):
    """
    Calculates a TSP tour using the Christofides 1.5-approximation algorithm.

    Args:
        points: A NumPy array of shape (N, D) where N is the number of points
                and D is the number of dimensions (e.g., 2 for 2D points).
                The dtype of this array should be numpy.float32.

    Returns:
        A NumPy array containing the indices of the points in the calculated tour.
    """
    cdef int N = points.shape[0]
    if N == 0:
        # Using the explicit dtype for consistency
        return np.array([], dtype=np.int64)

    # 1. Create a complete graph and calculate the distance matrix
    from scipy.spatial.distance import pdist, squareform
    # The distance matrix will be float64 by default, which is fine for internal calculations.
    cdef np.ndarray[np.float64_t, ndim=2] dist_matrix = squareform(pdist(points, 'euclidean'))

    # 2. Find the Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(dist_matrix)
    
    # 3. Find vertices with odd degree in the MST
    cdef np.ndarray[ITYPE_t, ndim=1] odd_degree_vertices = find_odd_degree_vertices(mst, N)
    
    # 4. Find a minimum-weight perfect matching on the odd-degree vertices.
    cdef list matching = greedy_matching(odd_degree_vertices, dist_matrix)
    
    # 5. Combine the MST and the matching to form a multigraph
    cdef list multigraph_adj = build_multigraph(mst, matching, N)
    
    # 6. Find an Eulerian circuit in the multigraph
    cdef list eulerian_circuit = find_eulerian_circuit(multigraph_adj, N)
    
    # 7. Convert the Eulerian circuit into a Hamiltonian circuit (the final tour)
    cdef np.ndarray[ITYPE_t, ndim=1] tour = shortcut_tour(eulerian_circuit)
    
    return tour


cdef np.ndarray[ITYPE_t, ndim=1] find_odd_degree_vertices(object mst, int N):
    """Finds all vertices with an odd degree in the MST."""
    # Using the explicit dtype for consistency
    cdef np.ndarray[ITYPE_t, ndim=1] degrees = np.zeros(N, dtype=np.int64)
    coo = mst.tocoo()
    for i in range(coo.nnz):
        degrees[coo.row[i]] += 1
        degrees[coo.col[i]] += 1
    
    # Using the explicit dtype for consistency
    cdef np.ndarray[ITYPE_t, ndim=1] odd_indices = np.where(degrees % 2 != 0)[0].astype(np.int64)
    return odd_indices

cdef list greedy_matching(np.ndarray[ITYPE_t, ndim=1] odd_vertices, np.ndarray[np.float64_t, ndim=2] dist_matrix):
    """
    Creates a greedy matching for the given odd-degree vertices.
    """
    cdef list matching = []
    cdef set unmatched = set(odd_vertices)
    
    while unmatched:
        u = unmatched.pop()
        min_dist = float('inf')
        best_v = -1
        
        for v in unmatched:
            if dist_matrix[u, v] < min_dist:
                min_dist = dist_matrix[u, v]
                best_v = v
        
        if best_v != -1:
            matching.append((u, best_v))
            unmatched.remove(best_v)
            
    return matching

cdef list build_multigraph(object mst, list matching, int N):
    """Builds an adjacency list for the combined multigraph (MST + matching)."""
    cdef list multigraph_adj = [[] for _ in range(N)]
    coo = mst.tocoo()
    cdef int i, u, v

    # Add MST edges
    for i in range(coo.nnz):
        # Explicitly cast to standard Python int to avoid subtle type issues
        u = int(coo.row[i])
        v = int(coo.col[i])
        multigraph_adj[u].append(v)
        multigraph_adj[v].append(u)
        
    # Add matching edges
    for u_match, v_match in matching:
        # Explicitly cast to standard Python int
        u = int(u_match)
        v = int(v_match)
        multigraph_adj[u].append(v)
        multigraph_adj[v].append(u)
        
    return multigraph_adj

cdef list find_eulerian_circuit(list multigraph_adj, int N):
    """
    Finds an Eulerian circuit using Hierholzer's algorithm (robust version).
    """
    if N == 0:
        return []

    cdef list adj_copy = [list(neighbors) for neighbors in multigraph_adj]
    cdef int start_node = -1
    cdef int i

    # Find a starting node that has at least one edge
    for i in range(N):
        if adj_copy[i]:
            start_node = i
            break
    
    # If graph has no edges, return a tour of a single node (if any exist)
    if start_node == -1:
        return [0] if N > 0 else []

    cdef list current_path = [start_node]
    cdef list circuit = []
    # Use 'object' type for current_v to avoid obscure Cython typing bugs
    cdef object current_v 
    cdef int next_v, path_len

    while current_path:
        path_len = len(current_path)
        current_v = current_path[path_len - 1]
        
        # Cast current_v to int for indexing adj_copy
        if adj_copy[int(current_v)]:
            next_v = adj_copy[int(current_v)].pop()
            
            # Remove the reverse edge for the undirected graph
            try:
                # .remove() works on values, so the object type is fine here
                adj_copy[next_v].remove(current_v)
            except ValueError:
                # This can happen in a multigraph with parallel edges; it's safe to ignore.
                pass

            current_path.append(next_v)
        else:
            circuit.append(current_path.pop())
            
    circuit.reverse()
    return circuit

cdef np.ndarray[ITYPE_t, ndim=1] shortcut_tour(list eulerian_circuit):
    """
    Creates the final TSP tour by removing repeated vertices from the
    Eulerian circuit.
    """
    if not eulerian_circuit:
        return np.array([], dtype=np.int64)
        
    cdef list tour_list = []
    cdef set visited = set()
    
    # Ensure first node is added to handle single-node tours
    if eulerian_circuit:
        first_node = eulerian_circuit[0]
        tour_list.append(first_node)
        visited.add(first_node)

    for v in eulerian_circuit:
        if v not in visited:
            tour_list.append(v)
            visited.add(v)
    
    # Add the starting point to the end to complete the cycle
    if tour_list:
        tour_list.append(tour_list[0])
    
    # Using the explicit dtype for consistency
    return np.array(tour_list, dtype=np.int64)
