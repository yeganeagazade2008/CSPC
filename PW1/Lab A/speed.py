import time
from decay import simulate_loop, simulate

N0 = 200_000
lam = 0.4
dt = 0.05
steps = 200

# Pure Python loop
start = time.perf_counter()
simulate_loop(N0, lam, dt=dt, steps=steps, seed=0)
loop_time = time.perf_counter() - start

# NumPy
start = time.perf_counter()
simulate(N0, lam, dt=dt, steps=steps, seed=0)
numpy_time = time.perf_counter() - start

print(f"Loop: {loop_time:.4f} s")
print(f"NumPy: {numpy_time:.4f} s")
print(f"Speed-up: {loop_time / numpy_time:.2f}x")