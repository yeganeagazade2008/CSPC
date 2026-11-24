import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
fs = 1 / (t[1] - t[0])

f, ts, S = spectrogram(Z, fs=fs, nperseg=256)
m = f <= 10

plt.figure(figsize=(10, 4))
plt.pcolormesh(ts, f[m], 10 * np.log10(S[m]), shading="auto")
plt.xlabel("time (s)")
plt.ylabel("frequency (Hz)")
plt.colorbar(label="power (dB)")
plt.savefig("figures/step7.png", dpi=150)
plt.show()