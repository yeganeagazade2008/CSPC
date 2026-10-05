"""
PW2 Lab B Part 4 -- Chemical equilibrium
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 50.0
sqrt_K = np.sqrt(K)

# Basitleştirilmiş k_imbalance fonksiyonu (Sıfıra eşitlenecek denklem)
def k_imbalance_simple(x):
    return (2.0 + sqrt_K) * x - sqrt_K

def k_imbalance_simple_prime(x):
    return 2.0 + sqrt_K

# 1. Newton metodu (Sorunsuz çalışır)
x_newton = newton(k_imbalance_simple, x0=0.5, fprime=k_imbalance_simple_prime)

# 2. SLSQP minimize (Orijinal denklem üzerinden minimizasyon)
def obj_func(x):
    return (((2.0 * x[0])**2) / ((1.0 - x[0])**2) - K)**2

res = minimize(obj_func, x0=[0.5], method="SLSQP", bounds=[(0.001, 0.999)])
x_slsqp = res.x[0]

print(f"Equilibrium x -> Newton: {x_newton:.4f}, SLSQP: {x_slsqp:.4f}")
print(f"Equilibrium amounts -> H2: {1-x_newton:.2f} mol, I2: {1-x_newton:.2f} mol, HI: {2*x_newton:.2f} mol")

# Grafik çizimi
x_vals = np.linspace(0.01, 0.95, 100)
plt.figure()
plt.plot(x_vals, 1 - x_vals, label="H2 / I2")
plt.plot(x_vals, 2 * x_vals, label="HI")
plt.axvline(x_newton, color="red", linestyle="--", label=f"Equilibrium (x={x_newton:.2f})")
plt.xlabel("Extent of reaction (x)")
plt.ylabel("Moles")
plt.legend()
plt.savefig("equilibrium.png")
print("equilibrium.png was created successfully!")