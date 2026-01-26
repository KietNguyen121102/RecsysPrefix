
import os
import glob
import pickle
import pandas as pd
import ipdb 

def gather_axiom_satisfaction_results():
    """
    Gather axiom satisfaction results from multiple sample directories,
    and produce a summary table showing fraction/count of samples
    where each aggregation method satisfied each axiom.
    """
    BASE = "/data2/rsalgani/Prefix/ml-1m/agg_files"   # change if needed
    pattern = os.path.join(BASE, "sample_*", "axiom_satisfaction_results.pkl")

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    dfs = []
    sample_names = []

    for p in paths:
        df = pickle.load(open(p, "rb"))   # index=methods, cols=[JR,PJR,EJR]
        # ipdb.set_trace() 
        df = df.astype(bool)
        dfs.append(df)
        sample_names.append(os.path.basename(os.path.dirname(p)))  # sample_0, sample_1, ...

    # Stack into one MultiIndex frame: (sample, method) x axioms
    big = pd.concat(dfs, keys=sample_names, names=["sample", "method"])

    # 1) Fraction of samples where True (mean over samples)
    frac_true = big.groupby("method").mean(numeric_only=True)

    # 2) Count of samples where True (sum over samples)
    count_true = big.groupby("method").sum(numeric_only=True).astype(int)

    # (optional) Also count how many samples actually had that method (in case some missing)
    n_samples_per_method = big.groupby("method").size()

    # Nice combined view: "count/total (fraction)"
    summary = count_true.copy()
    for col in summary.columns:
        summary[col] = summary[col].astype(str) + "/" + n_samples_per_method.astype(str) + \
                    " (" + (frac_true[col] * 100).round(1).astype(str) + "%)"

    print("\n=== Combined ===")
    print(summary)

    # Save outputs if you want
    out_dir = os.path.join(BASE, "summary_axioms")
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "count_total_percent.csv"))
    print(f"\nSaved to: {out_dir}")
  
def gather_diversity_results(): 
    return 0   

gather_axiom_satisfaction_results()