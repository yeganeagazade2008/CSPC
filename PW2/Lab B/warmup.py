"""
PW2 Lab B Part 2 -- Three routes to a minimum
"""
import numpy as np
from scipy.optimize import newton, minimize

# 2A. Convex function
def f(x):
    return (x - 3)**2 + 1

def df(x):
    return 2 * (x - 3)

def ddf(x):
    return 2

# (1) Gradient descent by hand
x_gd = 0.0
lr = 0.1
for _ in range(100):
    x_gd -= lr * df(x_gd)

# (2) Newton's method
x_newton = newton(df, x0=0.0, fprime=ddf)

# (3) SLSQP
res = minimize(f, x0=0.0, method="SLSQP")
x_slsqp = res.x[0]

print(f"Part 2A -> GD: {x_gd:.4f}, Newton: {x_newton:.4f}, SLSQP: {x_slsqp:.4f}")

# 2B. Harder landscape
def g(x):
    return x**4 - 3*x**2 + x + 5

def dg(x):
    return 4*x**3 - 6*x + 1

def ddg(x):
    return 12*x**2 - 6

for x0 in [0.0, 2.0]:
    x_gd_b = x0
    for _ in range(100):
        x_gd_b -= 0.01 * dg(x_gd_b)
    
    x_newton_b = newton(dg, x0=x0, fprime=ddg)
    is_min = ddg(x_newton_b) > 0
    res_b = minimize(g, x0=x0, method="SLSQP")
    
    print(f"Part 2B (x0={x0}) -> GD: {x_gd_b:.4f}, Newton: {x_newton_b:.4f} (Is Min? {is_min}), SLSQP: {res_b.x[0]:.4f}")