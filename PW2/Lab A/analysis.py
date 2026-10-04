import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

# ==========================================
# Part 2: From position to velocity and acceleration
# ==========================================
# (a) Read freefall.csv into arrays t and y
data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)
t = data[:, 0]
y = data[:, 1]

# (b) Compute velocity and acceleration using np.gradient
v = np.gradient(y, t)
a = np.gradient(v, t)

# (c) Print mean acceleration and standard deviation
mean_a = np.mean(a)
std_a = np.std(a)

print(f"Mean acceleration: {mean_a:.4f} m/s^2")
print(f"Standard deviation of acceleration: {std_a:.4f} m/s^2")

# ==========================================
# Part 4: Integrating back
# ==========================================
# Integrate acceleration back to velocity and position
v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

# Compare recovered position with original
max_diff = np.max(np.abs(y_rec - y))
print(f"Max difference between original and recovered position: {max_diff:.4f} m")

# ==========================================
# Part 5: Plotting and saving figure
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Panel 1: Position
ax1.plot(t, y, label='Position (y)', color='blue')
ax1.set_ylabel('Position (m)')
ax1.legend()
ax1.grid(True)

# Panel 2: Velocity
ax2.plot(t, v, label='Velocity (v)', color='orange')
ax2.set_ylabel('Velocity (m/s)')
ax2.legend()
ax2.grid(True)

# Panel 3: Acceleration
ax3.plot(t, a, label='Calculated Acceleration (a)', color='red', alpha=0.7)
ax3.axhline(-9.81, color='black', linestyle='--', label='Theoretical g (-9.81 m/s^2)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (m/s^2)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig('motion.png')
print("Plot saved as motion.png")

# ==========================================
# BONUS TASK: Noise Reduction using Savitzky-Golay Filter
# ==========================================
# Apply Savitzky-Golay filter to smooth position data (y)
y_smooth = savgol_filter(y, window_length=15, polyorder=2)

# Compute velocity and acceleration from smoothed position data
v_smooth = np.gradient(y_smooth, t)
a_smooth = np.gradient(v_smooth, t)

# Calculate bonus metrics
bonus_mean_acc = np.mean(a_smooth)
bonus_std_acc = np.std(a_smooth)

print("\n--- Bonus Task Results ---")
print(f"Smoothed Mean Acceleration: {bonus_mean_acc:.4f} m/s^2")
print(f"Smoothed Std Acceleration: {bonus_std_acc:.4f} m/s^2")