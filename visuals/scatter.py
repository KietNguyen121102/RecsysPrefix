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

def scatter_with_labels(x, y, title, xlab, ylab, xlim=None, outlier_note=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter methods within xlim if specified
    if xlim:
        in_range = [m for m in df.index if m in x.index and xlim[0] <= x[m] <= xlim[1]]
    else:
        in_range = [m for m in df.index if m in x.index]
    
    # Plot each method with unique marker/color
    for m in in_range:
        ax.scatter(x[m], y[m], 
                   s=150, 
                   c=[method_style[m]['color']], 
                   marker=method_style[m]['marker'],
                   edgecolors='black', 
                   linewidths=0.5,
                   label=m,
                   zorder=5)

    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    
    # Add outlier note
    if outlier_note:
        ax.annotate(outlier_note, xy=(xlim[0] + 0.01, ax.get_ylim()[1] * 0.95),
                   fontsize=11, fontstyle='italic', color='gray',
                   ha='left', va='top')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    ax.tick_params(labelsize=12)
    
    # Legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"ml1m_scatter_{xlab.replace(' ', '_')}_vs_{ylab.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Choose axes
x_kt = df["KT"]
y_ejr = df["EJR"]
y_lt = df["LT@10"]
y_c = df["C@10"]

# Zoom in on main cluster, note outlier
xlim = (0.58, 0.75)
outlier_note = "← Joe_Prefix_JR (KT=0.09)"

scatter_with_labels(
    x=x_kt, y=y_ejr,
    title="ML-1M: KT vs EJR (quality vs fairness)",
    xlab="Kendall tau (KT) ↑",
    ylab="EJR satisfaction (%) ↑",
    xlim=xlim,
    outlier_note=outlier_note
)

scatter_with_labels(
    x=x_kt, y=y_lt,
    title="ML-1M: KT vs LT@10 (quality vs diversity)",
    xlab="Kendall tau (KT) ↑",
    ylab="LT@10 ↑",
    xlim=xlim,
    outlier_note=outlier_note
)

scatter_with_labels(
    x=x_kt, y=y_c,
    title="ML-1M: KT vs C@10 (quality vs coverage/diversity)",
    xlab="Kendall tau (KT) ↑",
    ylab="C@10 ↑",
    xlim=xlim,
    outlier_note=outlier_note
)