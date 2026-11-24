import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("data/seismogram_A.csv")
t = d["time_s"].to_numpy()
Z = d["Z"].to_numpy()
fs = 1 / (t[1] - t[0])

# (a) indeks vs tezlik
X = np.fft.fft(Z)
f = np.fft.fftfreq(len(Z), d=1/fs)
amp = np.abs(X) / len(Z) * 2

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(amp);  ax[0].set_xlabel("array index")
ax[1].plot(f, amp); ax[1].set_xlabel("frequency (Hz)")
for a in ax: a.set_ylabel("amplitude (m/s)")
fig.savefig("figures/step2a.png", dpi=150)

# (b) sakit hissə vs bütün qeyd
def spec(x):
    fr = np.fft.fftfreq(len(x), d=1/fs)
    A = np.abs(np.fft.fft(x)) / len(x) * 2
    m = fr > 0
    return fr[m], A[m]

q = t < 60
plt.figure(figsize=(7, 5))
plt.loglog(*spec(Z[q]), label="quiet (first 60 s)")
plt.loglog(*spec(Z), label="whole record")
plt.xlabel("frequency (Hz)"); plt.ylabel("amplitude (m/s)")
plt.legend(); plt.grid(True, which="both", alpha=0.3)
plt.savefig("figures/step2b.png", dpi=150)
plt.show()