import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), d=1/fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real

def envelope(x, fs, win):
    w = int(win * fs)
    return np.convolve(np.abs(x), np.ones(w)/w, mode="same")

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z, N, E = (d[c].to_numpy() for c in ("Z", "N", "E"))
fs = 1 / (t[1] - t[0])

Zf = bandpass(Z, fs, 0.5, 10)
Nf = bandpass(N, fs, 0.5, 10)
Ef = bandpass(E, fs, 0.5, 10)
Hf = np.sqrt(Nf**2 + Ef**2)

# 0.5 s: Z və horizontal
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
ax[0].plot(t, envelope(Zf, fs, 0.5) * 1000); ax[0].set_ylabel("Z envelope (mm/s)")
ax[1].plot(t, envelope(Hf, fs, 0.5) * 1000); ax[1].set_ylabel("horizontal envelope (mm/s)")
ax[1].set_xlabel("time (s)")
fig.savefig("figures/step4a.png", dpi=150)

# Z üçün üç pəncərə
plt.figure(figsize=(10, 4))
for w in (0.05, 0.5, 5):
    plt.plot(t, envelope(Zf, fs, w) * 1000, label=f"{w} s", alpha=0.8)
plt.xlim(95, 200)       # P və S ətrafına zoom
plt.xlabel("time (s)"); plt.ylabel("Z envelope (mm/s)"); plt.legend()
plt.savefig("figures/step4b.png", dpi=150)
plt.show()