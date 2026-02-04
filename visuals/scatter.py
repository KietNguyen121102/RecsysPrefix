import matplotlib.pyplot as plt
import numpy as np

def scatter_with_labels(x, y, title, xlab, ylab, highlight=None):
    highlight = set(highlight or [])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)

    for m in df.index:
        ax.annotate(m, (x[m], y[m]), fontsize=8, alpha=0.85)

    # Highlight chosen methods
    if highlight:
        ax.scatter([x[m] for m in highlight], [y[m] for m in highlight], s=80)
        for m in highlight:
            ax.annotate(m, (x[m], y[m]), fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    plt.tight_layout()
    plt.savefig(f"ml1m_scatter_{xlab.replace(' ', '_')}_vs_{ylab.replace(' ', '_')}.png", dpi=300)

# Choose axes
x_kt = df["KT"]
y_ejr = df["EJR"]
y_lt = df["LT@10"]
y_c = df["C@10"]

highlight = ["Our_Prefix_ILP", "Our_Prefix_Fair_ILP", "Median", "DIBRA", "RRF", "FairMedian", "MC1", "Joe_Prefix_JR"]

scatter_with_labels(
    x=x_kt, y=y_ejr,
    title="ML-1M: KT vs EJR (quality vs fairness)",
    xlab="Kendall tau (KT) ↑",
    ylab="EJR satisfaction (%) ↑",
    highlight=highlight
)

scatter_with_labels(
    x=x_kt, y=y_lt,
    title="ML-1M: KT vs LT@10 (quality vs diversity)",
    xlab="Kendall tau (KT) ↑",
    ylab="LT@10 ↑",
    highlight=highlight
)

scatter_with_labels(
    x=x_kt, y=y_c,
    title="ML-1M: KT vs C@10 (quality vs coverage/diversity)",
    xlab="Kendall tau (KT) ↑",
    ylab="C@10 ↑",
    highlight=highlight
)