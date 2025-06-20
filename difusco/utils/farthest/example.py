# main_farthest.py

import numpy as np
from scipy.spatial import distance_matrix
import time

# Import the compiled Cython module
import farthest_cython

# --- Original Python function from the file for comparison ---
def farthest_insertion_heuristic_python(points):
    N = len(points)
    dist = distance_matrix(points, points)

    # Step 1
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    tour = [i, j] 
    visited = {i, j}

    while len(visited) < N:
        # Step 2
        max_dist = -1
        farthest = -1
        # In Python, we have to create the list of unvisited nodes first
        unvisited_nodes = [k for k in range(N) if k not in visited]
        for k in unvisited_nodes:
            min_dist_to_tour = min(dist[k][t] for t in tour)
            if min_dist_to_tour > max_dist:
                max_dist = min_dist_to_tour
                farthest = k

        # Step 3
        best_increase = float('inf')
        best_pos = -1
        for idx in range(len(tour)):
            a, b = tour[idx], tour[(idx + 1) % len(tour)]
            increase = dist[a][farthest] + dist[farthest][b] - dist[a][b]
            if increase < best_increase:
                best_increase = increase
                best_pos = idx + 1

        tour.insert(best_pos, farthest)
        visited.add(farthest)

    return tour


if __name__ == "__main__":
    num_points = 100
    print(f"Generating tour for {num_points} points.")

    # Generate points and ensure correct dtype for Cython
    points = np.random.rand(num_points, 2).astype(np.float64)

    # --- Benchmark Python Version ---
    print("\nRunning pure Python version...")
    start_time_py = time.time()
    tour_py = farthest_insertion_heuristic_python(points)
    end_time_py = time.time()
    duration_py = end_time_py - start_time_py
    print(f"Python version took: {duration_py:.6f} seconds.")

    # --- Benchmark Cython Version ---
    print("\nRunning Cython version...")
    start_time_cy = time.time()
    tour_cy = farthest_cython.farthest_insertion_heuristic_cython(points)
    end_time_cy = time.time()
    duration_cy = end_time_cy - start_time_cy
    print(f"Cython version took: {duration_cy:.6f} seconds.")

    # --- Comparison ---
    print("\n--- Comparison ---")
    if duration_cy > 0:
        speedup = duration_py / duration_cy
        print(f"Cython is approximately {speedup:.2f}x faster.")
    else:
        print("Cython execution was too fast to measure a speedup.")

    # Verify that both algorithms produce the same result
    assert set(tour_py) == set(tour_cy), "The tours do not contain the same cities!"
    print("Verification successful: Both tours visit the same set of cities.")
