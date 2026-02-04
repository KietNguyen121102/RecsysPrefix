import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

from temp_data_dump import axioms, c_at_10, lt_at_10, kt, kt_group

# ---------- Build DataFrame ----------
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

# Define colors for method categories
def get_method_color(m):
    if "Our_Prefix" in m or "Joe_Prefix" in m:
        return "#e41a1c"  # Red - your methods
    elif m in ["Median", "DIBRA", "RRF", "BordaCount"]:
        return "#377eb8"  # Blue - top baselines
    elif m in ["FairMedian", "KuhlmanConsensus"]:
        return "#984ea3"  # Purple - fair methods
    elif m.startswith("MC"):
        return "#ff7f00"  # Orange - Markov chains
    elif m.startswith("Comb"):
        return "#4daf4a"  # Green - Comb family
    else:
        return "#999999"  # Gray - others

# =============================================================================
# PLOT 1: Horizontal Bar Chart - KT Ranking
# =============================================================================
def plot_kt_ranking():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    df_sorted = df.sort_values("KT", ascending=True)
    colors = [get_method_color(m) for m in df_sorted.index]
    
    bars = ax.barh(df_sorted.index, df_sorted["KT"], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, df_sorted["KT"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=10)
    
    ax.set_xlabel("Kendall Tau (KT)", fontsize=12)
    ax.set_title("ML-1M: Methods Ranked by Quality (KT)", fontsize=14)
    ax.set_xlim(0, 0.85)
    ax.tick_params(axis='y', labelsize=11)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#e41a1c', label='Our Methods'),
        Patch(facecolor='#377eb8', label='Top Baselines'),
        Patch(facecolor='#984ea3', label='Fair Methods'),
        Patch(facecolor='#ff7f00', label='Markov Chains'),
        Patch(facecolor='#4daf4a', label='Comb Family'),
        Patch(facecolor='#999999', label='Others'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("plot1_kt_ranking.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot1_kt_ranking.png")

# =============================================================================
# PLOT 2: Grouped Bar Chart - Key Metrics Comparison
# =============================================================================
def plot_grouped_bars():
    # Select key methods and metrics
    key_methods = ["Our_Prefix_ILP", "Median", "DIBRA", "BordaCount", "FairMedian", "MC1", "CombMIN"]
    key_methods = [m for m in key_methods if m in df.index]
    metrics = ["KT", "EJR", "C@10", "LT@10"]
    
    # Normalize for comparison (KT and diversity are 0-1, EJR is 0-100)
    df_plot = df.loc[key_methods, metrics].copy()
    df_plot["EJR"] = df_plot["EJR"] / 100  # Normalize to 0-1
    
    x = np.arange(len(key_methods))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3']
    for i, met in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df_plot[met], width, label=met, color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel("Score (normalized 0-1)", fontsize=12)
    ax.set_title("ML-1M: Key Methods Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(key_methods, rotation=30, ha='right', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plot2_grouped_bars.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot2_grouped_bars.png")

# =============================================================================
# PLOT 3: Trade-off Quadrant Plot - KT vs EJR
# =============================================================================
def plot_tradeoff_quadrant():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = df["KT"]
    y = df["EJR"]
    
    # Calculate medians for quadrant lines
    x_med = x.median()
    y_med = y.median()
    
    # Plot each method
    for m in df.index:
        color = get_method_color(m)
        ax.scatter(x[m], y[m], s=150, c=color, edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(m, (x[m], y[m]), fontsize=9, xytext=(5, 5), textcoords='offset points')
    
    # Quadrant lines
    ax.axvline(x=x_med, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=y_med, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax.text(0.75, 102, "Best: High Quality + High Fairness", fontsize=10, ha='center', style='italic', color='green')
    ax.text(0.2, 102, "High Fairness, Low Quality", fontsize=10, ha='center', style='italic', color='orange')
    ax.text(0.75, 15, "High Quality, Low Fairness", fontsize=10, ha='center', style='italic', color='orange')
    ax.text(0.2, 15, "Worst: Low Quality + Low Fairness", fontsize=10, ha='center', style='italic', color='red')
    
    ax.set_xlabel("Kendall Tau (KT) - Quality ↑", fontsize=12)
    ax.set_ylabel("EJR Satisfaction (%) - Fairness ↑", fontsize=12)
    ax.set_title("ML-1M: Quality vs Fairness Trade-off", fontsize=14)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig("plot3_tradeoff_quadrant.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot3_tradeoff_quadrant.png")

# =============================================================================
# PLOT 4: Parallel Coordinates
# =============================================================================
def plot_parallel_coordinates():
    from pandas.plotting import parallel_coordinates
    
    # Select key methods
    key_methods = ["Our_Prefix_ILP", "Median", "DIBRA", "BordaCount", "FairMedian", "MC1", "CombMIN"]
    key_methods = [m for m in key_methods if m in df.index]
    
    # Prepare data - normalize all to 0-1
    df_norm = df.loc[key_methods, ["KT", "EJR", "C@10", "LT@10"]].copy()
    for col in df_norm.columns:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        if mx > mn:
            df_norm[col] = (df_norm[col] - mn) / (mx - mn)
    
    df_norm["Method"] = df_norm.index
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [get_method_color(m) for m in key_methods]
    parallel_coordinates(df_norm, "Method", color=colors, ax=ax, linewidth=2.5)
    
    ax.set_ylabel("Normalized Score (0-1)", fontsize=12)
    ax.set_title("ML-1M: Method Profiles (Parallel Coordinates)", fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.tick_params(axis='x', labelsize=11)
    
    plt.tight_layout()
    plt.savefig("plot4_parallel_coords.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot4_parallel_coords.png")

# =============================================================================
# PLOT 5: Lollipop Chart - Diversity Comparison
# =============================================================================
def plot_lollipop_diversity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Sort by C@10
    df_c = df.sort_values("C@10", ascending=True)
    df_lt = df.sort_values("LT@10", ascending=True)
    
    # Remove Joe_Prefix_JR for cleaner plot (outlier)
    df_c = df_c[df_c.index != "Joe_Prefix_JR"]
    df_lt = df_lt[df_lt.index != "Joe_Prefix_JR"]
    
    # C@10
    colors_c = [get_method_color(m) for m in df_c.index]
    axes[0].hlines(y=df_c.index, xmin=0, xmax=df_c["C@10"], color=colors_c, linewidth=2)
    axes[0].scatter(df_c["C@10"], df_c.index, color=colors_c, s=100, zorder=5, edgecolors='black')
    axes[0].set_xlabel("C@10 (Coverage)", fontsize=12)
    axes[0].set_title("Coverage Diversity", fontsize=13)
    axes[0].set_xlim(0, 0.45)
    
    # LT@10
    colors_lt = [get_method_color(m) for m in df_lt.index]
    axes[1].hlines(y=df_lt.index, xmin=0, xmax=df_lt["LT@10"], color=colors_lt, linewidth=2)
    axes[1].scatter(df_lt["LT@10"], df_lt.index, color=colors_lt, s=100, zorder=5, edgecolors='black')
    axes[1].set_xlabel("LT@10 (Long-tail)", fontsize=12)
    axes[1].set_title("Long-tail Diversity", fontsize=13)
    axes[1].set_xlim(0, 0.25)
    
    fig.suptitle("ML-1M: Diversity Metrics (excl. Joe_Prefix_JR outlier)", fontsize=14)
    plt.tight_layout()
    plt.savefig("plot5_lollipop_diversity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot5_lollipop_diversity.png")

# =============================================================================
# PLOT 6: Bump Chart - Rankings Across Metrics
# =============================================================================
def plot_bump_chart():
    metrics = ["KT", "EJR", "C@10", "LT@10"]
    
    # Calculate ranks (1 = best)
    ranks = pd.DataFrame(index=df.index)
    for met in metrics:
        ranks[met] = df[met].rank(ascending=False)
    
    # Select key methods
    key_methods = ["Our_Prefix_ILP", "Median", "DIBRA", "BordaCount", "FairMedian", "MC1", "CombMIN"]
    key_methods = [m for m in key_methods if m in df.index]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(metrics))
    
    for m in key_methods:
        color = get_method_color(m)
        y = ranks.loc[m, metrics].values
        ax.plot(x, y, 'o-', color=color, linewidth=2.5, markersize=10, label=m)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Rank (1 = Best)", fontsize=12)
    ax.set_title("ML-1M: Method Rankings Across Metrics", fontsize=14)
    ax.invert_yaxis()  # Rank 1 at top
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plot6_bump_chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot6_bump_chart.png")

# =============================================================================
# PLOT 7: KT Performance Gap by Group (Stacked/Grouped)
# =============================================================================
def plot_kt_group_gap():
    # Calculate gap from best (Median Group 0)
    df_grp = pd.DataFrame(kt_group).T
    df_grp.columns = ["Group 0", "Group 1", "Group 2"]
    
    # Sort by Group 0
    df_grp = df_grp.sort_values("Group 0", ascending=False)
    
    # Remove Joe_Prefix_JR outlier
    df_grp = df_grp[df_grp.index != "Joe_Prefix_JR"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(df_grp))
    width = 0.25
    
    colors = ['#377eb8', '#4daf4a', '#e41a1c']
    for i, col in enumerate(df_grp.columns):
        ax.bar(x + i*width, df_grp[col], width, label=col, color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel("Kendall Tau (KT)", fontsize=12)
    ax.set_title("ML-1M: KT Performance by User Group", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df_grp.index, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0.55, 0.82)
    
    plt.tight_layout()
    plt.savefig("plot7_kt_by_group.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot7_kt_by_group.png")

# =============================================================================
# PLOT 8: Scatter Matrix (Pairwise)
# =============================================================================
def plot_scatter_matrix():
    from pandas.plotting import scatter_matrix
    
    key_metrics = ["KT", "EJR", "C@10", "LT@10"]
    df_plot = df[key_metrics].copy()
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    scatter_matrix(df_plot, ax=axes, diagonal='hist', alpha=0.7, 
                   figsize=(12, 12), marker='o', s=80, edgecolors='black')
    
    fig.suptitle("ML-1M: Metric Correlations (Scatter Matrix)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("plot8_scatter_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot8_scatter_matrix.png")

# =============================================================================
# Run all plots
# =============================================================================
if __name__ == "__main__":
    print("Generating all plots...\n")
    
    plot_kt_ranking()
    plot_grouped_bars()
    plot_tradeoff_quadrant()
    plot_parallel_coordinates()
    plot_lollipop_diversity()
    plot_bump_chart()
    plot_kt_group_gap()
    plot_scatter_matrix()
    
    print("\n✓ All 8 plots generated!")
    print("\nPlot descriptions:")
    print("  1. KT Ranking - Horizontal bar chart of quality rankings")
    print("  2. Grouped Bars - Key metrics side-by-side comparison")
    print("  3. Trade-off Quadrant - KT vs EJR with quadrant labels")
    print("  4. Parallel Coords - Method profiles across metrics")
    print("  5. Lollipop Diversity - C@10 and LT@10 comparison")
    print("  6. Bump Chart - Ranking changes across metrics")
    print("  7. KT by Group - Performance by user group")
    print("  8. Scatter Matrix - Pairwise metric correlations")
