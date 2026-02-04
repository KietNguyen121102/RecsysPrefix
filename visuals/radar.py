import matplotlib.pyplot as plt
import numpy as np

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

