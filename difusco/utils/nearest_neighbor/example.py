# main.py

import numpy as np
import time
# Import the compiled Cython module
import nearest_neighbor_c

def nearest_neighbor_tour_python(points):
    """Original Python implementation for comparison."""
    N = len(points)
    unvisited = set(range(N))
    tour = [unvisited.pop()]
    while unvisited:
        last = tour[-1]
        # This part is slower in Python due to the lambda function and interpretation overhead
        next_city = min(unvisited, key=lambda j: np.linalg.norm(points[last] - points[j]))
        unvisited.remove(next_city)
        tour.append(next_city)
    tour.append(tour[0])
    return np.array(tour, dtype=int)

if __name__ == "__main__":
    # Generate some random points for testing
    num_points = 1500
    print(f"Generating a tour for {num_points} points.")
    points = np.random.rand(num_points, 2)

    # --- Benchmark Python Version ---
    print("\nRunning pure Python version...")
    start_time_py = time.time()
    tour_py = nearest_neighbor_tour_python(points)
    end_time_py = time.time()
    duration_py = end_time_py - start_time_py
    print(f"Python version took: {duration_py:.6f} seconds.")
    # print("Python tour:", tour_py) # Uncomment to see the tour

    # --- Benchmark Cython Version ---
    print("\nRunning Cython version...")
    start_time_cy = time.time()
    # Call the function from our compiled module
    tour_cy = nearest_neighbor_c.nearest_neighbor_c(points)
    end_time_cy = time.time()
    duration_cy = end_time_cy - start_time_cy
    print(f"Cython version took: {duration_cy:.6f} seconds.")
    # print("Cython tour:", tour_cy) # Uncomment to see the tour

    # --- Comparison ---
    print("\n--- Comparison ---")
    if duration_cy > 0:
        speedup = duration_py / duration_cy
        print(f"Cython is approximately {speedup:.2f}x faster.")
    else:
        print("Cython execution was too fast to measure a speedup.")

    # Verify that the tours are identical
    assert np.array_equal(tour_py, tour_cy), "Tours do not match!"
    print("Tours from both versions are identical. Verification successful.")

