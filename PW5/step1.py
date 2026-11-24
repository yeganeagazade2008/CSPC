import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
N = d["N"].to_numpy()
E = d["E"].to_numpy()

fs = 1 / (t[1] - t[0])
print("fs =", fs, "Hz, uzunluq =", t[-1] - t[0], "s, nümunə sayı =", len(t))

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
for a, x, name in zip(ax, (Z, N, E), "ZNE"):
    a.plot(t, x * 1000)      # m/s -> mm/s
    a.set_ylabel(f"{name} (mm/s)")
ax[-1].set_xlabel("time (s)")
fig.savefig("figures/step1.png", dpi=150)

fig2, ax2 = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
m = t < 60
for a, x, name in zip(ax2, (Z, N, E), "ZNE"):
    a.plot(t[m], x[m] * 1000)
    a.set_ylabel(f"{name} (mm/s)")
ax2[-1].set_xlabel("time (s)")
fig2.savefig("figures/step1_zoom.png", dpi=150)

plt.show()