"""
PW2 Lab B Part 3 -- fit a reaction's rate constant to measured data.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# TODO 1: read kinetics.csv (columns time, concentration) into arrays t, C.
#         Set C0 = the first concentration.
data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
t = data[:, 0]
C = data[:, 1]
C0 = C[0]

# TODO 2: write total_error(k) = sum of (measured - C0*exp(-k*t))^2.
def total_error(k):
    C_model = C0 * np.exp(-k * t)
    return np.sum((C - C_model)**2)

# TODO 3: minimise total_error with scipy.optimize.minimize
res = minimize(total_error, x0=0.5, method="SLSQP", bounds=[(0, 5)])
fitted_k = res.x[0]
print(f"Fitted k: {fitted_k:.4f}")

# TODO 4: plot the measured data and fitted curve. Save as kinetics.png.
plt.figure()
plt.plot(t, C, 'o', label="Data (measured)")
plt.plot(t, C0 * np.exp(-fitted_k * t), label=f"Fit (k={fitted_k:.4f})")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.savefig("kinetics.png")
print("kinetics.png was created successfully!")