import numpy as np
import pandas as pd

d = pd.read_csv("data/seismogram_A.csv")
N = d["N"].to_numpy()
E = d["E"].to_numpy()

R = 306.5  
PGV = np.max(np.sqrt(N**2 + E**2)) * 100   
M = (np.log10(PGV) + np.log10(R) + 1.0) / 0.55

print(f"PGV = {PGV:.3f} cm/s")
print(f"M = {M:.2f}")
print(f"ignoring log10 R at 2R: M changes by {np.log10(2)/0.55:.2f}")
print(f"+1 magnitude -> PGV x {10**0.55:.2f}")