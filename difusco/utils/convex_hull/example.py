# main_hull.py

import numpy as np
from scipy.spatial import distance_matrix, ConvexHull
import time

# Import the newly compiled Cython module
import convex_hull_cython

# --- Original Python Function from your file (for comparison) ---
def convex_hull_insertion_heuristic_python(points):
    N = len(points)
    dist = distance_matrix(points, points)

    hull = ConvexHull(points)
    hull_vertices = list(hull.vertices)
    
    tour = hull_vertices[:] 
    visited = set(tour)
    remaining = [i for i in range(N) if i not in visited]

    while remaining:
        best_increase = float('inf')
        best_insert = None
        best_pos = None

        for p in remaining:
            for i in range(len(tour)):
                a, b = tour[i], tour[(i + 1) % len(tour)]
                increase = dist[a, p] + dist[p, b] - dist[a, b]
                if increase < best_increase:
                    best_increase = increase
                    best_insert = p
                    best_pos = i + 1

        tour.insert(best_pos, best_insert)
        remaining.remove(best_insert)

    return tour


if __name__ == "__main__":
    num_points = 100
    print(f"Generating tour for {num_points} points.")
    
    # Generate points and ensure correct dtype for Cython
    points = np.random.rand(num_points, 2).astype(np.float64)

    # --- Benchmark Python Version ---
    print("\nRunning pure Python version...")
    start_time_py = time.time()
    tour_py = convex_hull_insertion_heuristic_python(points)
    end_time_py = time.time()
    duration_py = end_time_py - start_time_py
    print(f"Python version took: {duration_py:.6f} seconds.")

    # --- Benchmark Cython Version ---
    print("\nRunning Cython version...")
    start_time_cy = time.time()
    tour_cy = convex_hull_cython.convex_hull_insertion_heuristic_cython(points)
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

    # Optional: Verify that the tours are the same
    # Note: Small floating point differences might lead to different but equally valid tours.
    # We can check if the set of cities in the tour is the same.
    assert set(tour_py) == set(tour_cy), "The set of cities in the tours does not match!"
    print("Verification successful: Both tours visit the same set of cities.")

