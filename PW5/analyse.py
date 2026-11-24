import os
import numpy as np
import pandas as pd

VP, VS = 6.2, 3.6   # km/s

def bandpass(x, fs, a, b):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), d=1/fs)
    X[(np.abs(f) < a) | (np.abs(f) > b)] = 0
    return np.fft.ifft(X).real

def envelope(x, fs, win):
    w = int(win * fs)
    return np.convolve(np.abs(x), np.ones(w)/w, mode="same")

def analyse(csv_file):
    d = pd.read_csv(csv_file)
    t = d["time_s"].to_numpy()
    Z, N, E = (d[c].to_numpy() for c in ("Z", "N", "E"))
    fs = 1 / (t[1] - t[0])

    Zf, Nf, Ef = (bandpass(x, fs, 0.5, 10) for x in (Z, N, E))
    Hf = np.sqrt(Nf**2 + Ef**2)
    envZ = envelope(Zf, fs, 0.5)
    envH = envelope(Hf, fs, 0.5)

    tP = t[np.argmax(envZ > 0.05 * envZ.max())]
    j0 = np.searchsorted(t, tP + 2)
    tS = t[j0 + np.argmax(envH[j0:] > 0.10 * envH.max())]

    R = (tS - tP) / (1/VS - 1/VP)
    origin = tP - R / VP

    PGV = np.max(np.sqrt(N**2 + E**2)) * 100      # cm/s, xam siqnal
    M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55

    return dict(record=os.path.basename(csv_file), tP=round(tP, 2),
                tS=round(tS, 2), R_km=round(R, 1), origin_time=round(origin, 1),
                PGV_cms=round(PGV, 3), M=round(M, 2))

if __name__ == "__main__":
    rows = [analyse(f) for f in ("data/seismogram_A.csv", "data/seismogram_B.csv")]
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/events.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))