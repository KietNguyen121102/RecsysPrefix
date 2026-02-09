import matplotlib.pyplot as plt

# Data
k_values = [5, 10, 15, 20, 25]

# fairmedian = [0.704722, 0.689936, 0.686046, 0.688989, 0.701457]
# kuhlman = [0.704161, 0.689080, 0.685559, 0.688584, 0.701090]
our_ilp = [0.698451, 0.683922, 0.679738, 0.682699, 0.694441]

# Plot
plt.figure()
# plt.plot(k_values, fairmedian, marker='o', label='FairMedian')
# plt.plot(k_values, kuhlman, marker='o', label='KuhlmanConsensus')
plt.plot(k_values, our_ilp, marker='o', label='Our_Prefix_Fair_ILP')

plt.xlabel('k')
plt.ylabel('Mean (KT)')
plt.legend()
plt.tight_layout()
plt.savefig('ml1m_k_and_KT.png')



import matplotlib.pyplot as plt

# k values
k = [5, 10, 15, 20, 25]

# Percentages extracted from the table
data = {
    "Our_Prefix_Fair_ILP": {
        "JR":  [100, 100, 100, 100, 100],
        "PJR": [81, 99, 100, 100, 97],
        "EJR": [78, 98, 100, 100, 97],
    },
}

# Plot: one figure per axiom
for axiom in ["JR", "PJR", "EJR"]:
    plt.figure()
    for method, vals in data.items():
        plt.plot(k, vals[axiom], marker="o", label=method)

    plt.xlabel("k")
    plt.ylabel(f"{axiom} Satisfaction (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ml1m_k_and_satisfaction.png')

