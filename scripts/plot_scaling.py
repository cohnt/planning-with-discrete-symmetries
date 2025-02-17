import numpy as np
import matplotlib.pyplot as plt

samples_symmetry = [
    [94, 73, 159, 116, 142],
    [260, 1022, 809, 334, 1532],
    [213, 677, 2513, np.nan, np.nan]
]
samples_baseline = [
    [100, 183, 301, 542, 151],
    [303, 1160, 4534, 317, 1803],
    [292, 675, 1120, np.nan, np.nan]
]

c_space_dimensions = [3, 6, 9]

min_samples_symmetry = np.nanmin(samples_symmetry, axis=1)
avg_samples_symmetry = np.nanmean(samples_symmetry, axis=1)
max_samples_symmetry = np.nanmax(samples_symmetry, axis=1)

min_samples_baseline = np.nanmin(samples_baseline, axis=1)
avg_samples_baseline = np.nanmean(samples_baseline, axis=1)
max_samples_baseline = np.nanmax(samples_baseline, axis=1)

# min_samples_symmetry = [1, 4, 5]
# avg_samples_symmetry = [10, 31, 100]
# max_samples_symmetry = [100, 128, 300]

# min_samples_baseline = [1, 4, 5]
# avg_samples_baseline = [10, 100, 1000]
# max_samples_baseline = [125, 369, 4560]

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