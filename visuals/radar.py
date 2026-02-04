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

# Pick a small set
radar_methods = [
    "Our_Prefix_ILP",
    "Our_Prefix_Fair_ILP",
    "Median",
    "DIBRA",
    "RRF",
    "FairMedian",
    "KuhlmanConsensus",
    "CombMAX",
    "CombMIN",
    "MC1",
    "Joe_Prefix_JR",
]

metrics = ["JR", "PJR", "EJR", "KT", "C@10", "LT@10"]

# Normalize each metric to [0,1] so axes are comparable
Z = df[metrics].apply(lambda c: (c - c.min()) / (c.max() - c.min()) if not np.isclose(c.max(), c.min()) else 0.5)

# Radar setup
labels = metrics
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for m in radar_methods:
    vals = Z.loc[m, labels].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=1, label=m)
    ax.fill(angles, vals, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticklabels([])
ax.set_title("ML-1M Radar (per-metric min-max normalized)")

ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.savefig("ml1m_methods_metrics_radar.png", dpi=300)

