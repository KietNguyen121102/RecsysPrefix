import matplotlib.pyplot as plt
import pandas as pd 


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

# Optional: drop methods with missing metrics (should be none for ML-1M here)
# df = df.dropna()
print(df.head())

# ---------- Plotting ----------

import matplotlib.pyplot as plt

def minmax_normalize(col: pd.Series) -> pd.Series:
    mn, mx = col.min(), col.max()
    if np.isclose(mx, mn):
        return pd.Series(0.5, index=col.index)  # constant column
    return (col - mn) / (mx - mn)

# Choose which metrics to show in heatmap
metrics = ["JR", "PJR", "EJR", "KT", "C@10", "LT@10"]

# Normalized for coloring; keep raw for annotation
Z_raw = df[metrics].copy()
Z = Z_raw.apply(minmax_normalize, axis=0)

# Optionally sort methods by something (e.g., KT descending)
order = df.sort_values(by="KT", ascending=False).index
Z = Z.loc[order]
Z_raw = Z_raw.loc[order]

fig, ax = plt.subplots(figsize=(16, 0.55 * len(Z) + 3))
im = ax.imshow(Z.values, aspect="auto")

ax.set_xticks(np.arange(len(metrics)))
ax.set_xticklabels(metrics, rotation=0, ha="center", fontsize=20)
ax.set_yticks(np.arange(len(Z.index)))
ax.set_yticklabels(Z.index, fontsize=20)

# Annotate raw values (format by metric)
fmt = {
    "JR": "{:.0f}",
    "PJR": "{:.0f}",
    "EJR": "{:.0f}",
    "KT": "{:.3f}",
    "C@10": "{:.3f}",
    "LT@10": "{:.3f}",
}

for i, method in enumerate(Z.index):
    for j, met in enumerate(metrics):
        val = Z_raw.loc[method, met]
        norm_val = Z.loc[method, met]
        # Use white text on dark cells, black text on light cells
        text_color = "white" if norm_val < 0.5 else "black"
        ax.text(j, i, fmt[met].format(val), ha="center", va="center", fontsize=20, color=text_color, fontweight="medium")

ax.set_title("ML-1M: Methods × Metrics (colors are per-metric min-max normalized)", fontsize=22)
cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Normalized Score", fontsize=20)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig("ml1m_methods_metrics_heatmap.png", dpi=300)
