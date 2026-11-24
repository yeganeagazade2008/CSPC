import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

Zf, Nf, Ef = (bandpass(x, fs, 0.5, 10) for x in (Z, N, E))
Hf = np.sqrt(Nf**2 + Ef**2)
envZ = envelope(Zf, fs, 0.5)
envH = envelope(Hf, fs, 0.5)

# P: Z zərfi max-ın 5%-ni ilk keçən an
tP = t[np.argmax(envZ > 0.05 * envZ.max())]
# S: tP+2 s-dən sonra horizontal zərf max-ın 10%-ni ilk keçən an
j0 = np.searchsorted(t, tP + 2)
tS = t[j0 + np.argmax(envH[j0:] > 0.10 * envH.max())]

# horizontal zərfin maksimumu
pk, _ = find_peaks(envH, height=0.5 * envH.max())
t_max = t[pk[np.argmax(envH[pk])]]

print(f"tP = {tP:.2f} s")
print(f"tS = {tS:.2f} s")
print(f"horizontal envelope max at {t_max:.2f} s (tS-dən {t_max - tS:.2f} s sonra)")

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
for a, x, name in zip(ax, (Zf, Nf, Ef), "ZNE"):
    a.plot(t, x * 1000)
    a.axvline(tP, color="g", label=f"tP = {tP:.1f} s")
    a.axvline(tS, color="r", label=f"tS = {tS:.1f} s")
    a.axvline(t_max, color="k", ls="--", label="env max")
    a.set_ylabel(f"{name} (mm/s)")
ax[0].legend(loc="upper right")
ax[-1].set_xlabel("time (s)")
fig.savefig("figures/step5.png", dpi=150)
plt.show()