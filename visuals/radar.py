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

# Pick a curated set: your methods + key baselines + interesting contrasts
radar_methods = [
    "Our_Prefix_ILP",      # Your method
    "Median",              # Top baseline (highest KT)
    "BordaCount",          # Classic baseline
    "FairMedian",          # Fair method comparison
    "MC1",                 # Low performer (for contrast)
]

# Define distinct styles for each method
method_styles = {
    "Our_Prefix_ILP":  {"color": "#e41a1c", "linestyle": "-",  "linewidth": 3, "marker": "o"},  # Red solid
    "Median":          {"color": "#377eb8", "linestyle": "--", "linewidth": 2.5, "marker": "s"},  # Blue dashed
    "BordaCount":      {"color": "#4daf4a", "linestyle": "-.", "linewidth": 2.5, "marker": "^"},  # Green dash-dot
    "FairMedian":      {"color": "#984ea3", "linestyle": ":",  "linewidth": 2.5, "marker": "D"},  # Purple dotted
    "MC1":             {"color": "#ff7f00", "linestyle": "-",  "linewidth": 2, "marker": "v"},    # Orange solid
}

metrics = ["JR", "PJR", "EJR", "KT", "C@10", "LT@10"]

# Normalize each metric to [0,1] so axes are comparable
Z = df[metrics].apply(lambda c: (c - c.min()) / (c.max() - c.min()) if not np.isclose(c.max(), c.min()) else 0.5)

# Radar setup
labels = metrics
N = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

for m in radar_methods:
    if m not in df.index:
        continue
    vals = Z.loc[m, labels].tolist()
    vals += vals[:1]
    style = method_styles[m]
    ax.plot(angles, vals, 
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=8,
            label=m)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels([])
ax.set_title("ML-1M Radar (per-metric min-max normalized)", fontsize=14, pad=20)

# Grid styling
ax.grid(True, alpha=0.3)

ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=11, frameon=True)
plt.tight_layout()
plt.savefig("ml1m_methods_metrics_radar.png", dpi=300, bbox_inches='tight')
plt.close()

