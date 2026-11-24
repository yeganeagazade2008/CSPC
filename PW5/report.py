import pandas as pd

df = pd.read_csv("results/events.csv")
lines = []
for _, r in df.iterrows():
    lines.append(
        f"{r.record}: P arrived at {r.tP:.1f} s and S at {r.tS:.1f} s, "
        f"so the earthquake lies {r.R_km:.0f} km away and happened at "
        f"{r.origin_time:.1f} s. Peak ground velocity {r.PGV_cms:.2f} cm/s, "
        f"magnitude {r.M:.1f}."
    )
closer = df.loc[df.R_km.idxmin(), "record"]
stronger = df.loc[df.M.idxmax(), "record"]
lines.append(f"Closer earthquake: {closer}. Stronger earthquake: {stronger}.")

with open("results/report.txt", "w") as fh:
    fh.write("\n".join(lines) + "\n")