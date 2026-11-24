import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), d=1/fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z, N, E = (d[c].to_numpy() for c in ("Z", "N", "E"))
fs = 1 / (t[1] - t[0])

# (a) ifft(fft(Z)) == Z ?
print("allclose:", np.allclose(np.fft.ifft(np.fft.fft(Z)).real, Z))

# (b) filtr
Zf = bandpass(Z, fs, 0.5, 10)
Nf = bandpass(N, fs, 0.5, 10)
Ef = bandpass(E, fs, 0.5, 10)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
ax[0].plot(t, Z * 1000);  ax[0].set_ylabel("raw Z (mm/s)")
ax[1].plot(t, Zf * 1000); ax[1].set_ylabel("filtered Z (mm/s)")
ax[1].set_xlabel("time (s)")
fig.savefig("figures/step3b.png", dpi=150)

# (c) çıxarılan hissə
plt.figure(figsize=(10, 3))
plt.plot(t, (Z - Zf) * 1000)
plt.xlabel("time (s)"); plt.ylabel("Z - clean (mm/s)")
plt.savefig("figures/step3c.png", dpi=150)
plt.show()