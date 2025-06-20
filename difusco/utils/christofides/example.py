import numpy as np
import time
import matplotlib.pyplot as plt

# --- Step 1: Compile the Cython code ---
# Before running this script, you must compile the .pyx file.
# Open your terminal in the same directory as the files and run:
# python setup.py build_ext --inplace
# This will create a file like 'christofides_tsp.cpython-39-x86_64-linux-gnu.so'
# You can then import it like a regular Python module.

try:
    from christofides_c import christofides_c
except ImportError:
    print("="*50)
    print("Cython module not compiled!")
    print("Please run 'python setup.py build_ext --inplace' first.")
    print("="*50)
    exit()

def plot_tour(points, tour, title="TSP Tour"):
    """Helper function to visualize the TSP tour."""
    plt.figure(figsize=(8, 8))
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', zorder=2, label='Cities')
    for i, p in enumerate(points):
        plt.text(p[0] + 0.01, p[1] + 0.01, str(i))

    # Plot tour
    tour_points = points[tour]
    plt.plot(tour_points[:, 0], tour_points[:, 1], 'r-', zorder=1, label='Tour')
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_tour_length(points, tour):
    """Calculates the total length of the TSP tour."""
    dist = 0
    for i in range(len(tour) - 1):
        p1 = points[tour[i]]
        p2 = points[tour[i+1]]
        dist += np.linalg.norm(p1 - p2)
    return dist

# --- Step 2: Generate data and run the algorithm ---

# Generate some random points for the TSP problem
NUM_POINTS = 50
np.random.seed(42)
points = np.random.rand(NUM_POINTS, 2)

print(f"Solving TSP for {NUM_POINTS} points using Christofides algorithm...")

# Time the execution
start_time = time.time()
tour = christofides_c(points)
end_time = time.time()

# Calculate tour length
tour_length = calculate_tour_length(points, tour)

print(f"\nTour found: {tour}")
print(f"Total tour length: {tour_length:.4f}")
print(f"Execution time: {end_time - start_time:.6f} seconds")

# --- Step 3: Visualize the result ---
plot_tour(points, tour, title=f"Christofides TSP Solution ({NUM_POINTS} Cities)")

