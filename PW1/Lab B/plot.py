import matplotlib.pyplot as plt
import numpy as np

# Constants
LAMBDA = 0.3

# TODO 1: Read the CSV file (skip header row)
data = np.loadtxt("decay_observed.csv", delimiter=",", skiprows=1)
t = data[:, 0]
observed = data[:, 1]

# TODO 2: Set N0 to the first observed value and compute the analytical law
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# TODO 3: Create a 1x2 subplot with shared x and y axes
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))

# Left plot: Scatter of observed data
ax1.scatter(t, observed, color="blue", label="Observed Data", s=15)
ax1.set_title("Observed Data")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Count (N)")
ax1.grid(True)
ax1.legend()

# Right plot: Line of analytical law
ax2.plot(t, analytical, color="red", label=r"$N_0 e^{-\lambda t}$", linewidth=2)
ax2.set_title("Analytical Decay Law")
ax2.set_xlabel("Time (t)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# TODO 4: Save the figure as figure.png
plt.savefig("figure.png", dpi=300)
print("Figure successfully saved as figure.png")