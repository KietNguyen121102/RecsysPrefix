import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1) Build dataframe from your KT group results
# ==============================

from temp_data_dump import kt_group  # Assuming kt_group is defined in this module

df = pd.DataFrame(kt_group).T
df.columns = ["Group 0", "Group 1", "Group 2"]

# Optional: sort by average KT
df["avg"] = df.mean(axis=1)
df = df.sort_values("avg", ascending=False).drop(columns="avg")


# ==============================
# 2) Plot grouped bars
# ==============================

methods = df.index.tolist()
x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))

bars0 = ax.bar(x - width, df["Group 0"], width, label="Group 0")
bars1 = ax.bar(x,         df["Group 1"], width, label="Group 1")
bars2 = ax.bar(x + width, df["Group 2"], width, label="Group 2")
highlight = {"Our_Prefix_ILP", "Our_Prefix_Fair_ILP", "Joe_Prefix_JR"}

for i, m in enumerate(methods):
    if m in highlight:
        for b in [bars0[i], bars1[i], bars2[i]]:
            b.set_edgecolor("black")
            b.set_linewidth(2.5)

ax.set_ylabel("Kendall tau (KT)")
ax.set_title("ML-1M: KT Performance by Group")
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=75, ha="right")
ax.legend()

ax.set_ylim(0, 0.85)

plt.tight_layout()
plt.savefig("ml1m_kt_by_group.png", dpi=300)
