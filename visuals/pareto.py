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

# ---------- Define unique markers and colors for each method ----------
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>', '8', 'd', 'H', '+', 'x']
colors = plt.cm.tab20(np.linspace(0, 1, 20))

# Create a mapping for each method
method_style = {}
for i, m in enumerate(sorted(df.index)):
    method_style[m] = {
        'marker': markers[i % len(markers)],
        'color': colors[i % len(colors)]
    }

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

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each method with unique marker/color
for m in df.index:
    ax.scatter(df.loc[m, "KT"], df.loc[m, "LT@10"],
               s=150,
               c=[method_style[m]['color']],
               marker=method_style[m]['marker'],
               edgecolors='black',
               linewidths=0.5,
               label=m,
               zorder=5)

# Draw Pareto frontier line
fx = [p[0] for p in front]
fy = [p[1] for p in front]
ax.plot(fx, fy, 'k--', linewidth=2, alpha=0.7, label='Pareto frontier')

ax.set_title("ML-1M: KT vs LT@10 with Pareto frontier", fontsize=16)
ax.set_xlabel("KT ↑", fontsize=14)
ax.set_ylabel("LT@10 ↑", fontsize=14)
ax.tick_params(labelsize=12)

# Legend outside plot
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig("ml1m_kt_vs_lt10_pareto.png", dpi=300, bbox_inches='tight')
plt.close()
