import numpy as np
import matplotlib.pyplot as plt


c_space_dimensions = [3, 6, 9]

min_samples_symmetry = [1, 4, 5]
avg_samples_symmetry = [10, 31, 100]
max_samples_symmetry = [100, 128, 300]

min_samples_baseline = [1, 4, 5]
avg_samples_baseline = [10, 100, 1000]
max_samples_baseline = [125, 369, 4560]

samples_err_symmetry = [min_samples_symmetry, max_samples_symmetry]
samples_err_baseline = [min_samples_baseline, max_samples_baseline]

plt.errorbar(c_space_dimensions, avg_samples_symmetry, yerr=samples_err_symmetry, capsize=3, fmt="--o", label="Symmetry-Aware Average Samples")
plt.errorbar(c_space_dimensions, avg_samples_baseline, yerr=samples_err_baseline, capsize=3, fmt="--o", label="Baseline Average Samples")

plt.yscale("log")
plt.xticks(c_space_dimensions)
plt.legend()
plt.title("Samples Required by Configuration-Space Dimension")

plt.plot()
plt.show()