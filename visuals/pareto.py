import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from temp_data_dump import axioms, c_at_10, lt_at_10, kt

# ---------- Assemble into one dataframe ----------
methods = sorted(set(axioms) | set(c_at_10) | set(lt_at_10) | set(kt))

df = pd.DataFrame(index=methods, columns=["JR", "PJR", "EJR", "KT", "C@10", "LT@10"], dtype=float)

for m, (jr, pjr, ejr) in axioms.items():
    df.loc[m, ["JR", "PJR", "EJR"]] = [jr, pjr, ejr]

for m, v in kt.items():
    df.loc[m, "KT"] = v

for m, v in c_at_10.items():
    df.loc[m, "C@10"] = v

for m, v in lt_at_10.items():
    df.loc[m, "LT@10"] = v

def pareto_frontier(points):
    """
    points: list of (x, y, label)
    returns: frontier list of (x, y, label) sorted by x
    """
    pts = sorted(points, key=lambda t: t[0], reverse=True)
    frontier = []
    best_y = -np.inf
    for x, y, lab in pts:
        if y > best_y + 1e-12:
            frontier.append((x, y, lab))
            best_y = y
    return sorted(frontier, key=lambda t: t[0])

# Example: KT vs LT@10 frontier
pts = [(df.loc[m, "KT"], df.loc[m, "LT@10"], m) for m in df.index]
front = pareto_frontier(pts)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(df["KT"], df["LT@10"])
for m in df.index:
    ax.annotate(m, (df.loc[m, "KT"], df.loc[m, "LT@10"]), fontsize=8, alpha=0.8)

fx = [p[0] for p in front]
fy = [p[1] for p in front]
ax.plot(fx, fy, linewidth=2)

ax.set_title("ML-1M: KT vs LT@10 with Pareto frontier")
ax.set_xlabel("KT ↑")
ax.set_ylabel("LT@10 ↑")
plt.tight_layout()
plt.savefig("ml1m_kt_vs_lt10_pareto.png", dpi=300)
