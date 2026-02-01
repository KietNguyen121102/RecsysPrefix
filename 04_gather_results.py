import os
import glob
import pickle
import pandas as pd
import ipdb 
import argparse
import numpy as np

def gather_axiom_satisfaction_results(dataset):
    """
    Gather axiom satisfaction results from multiple sample directories,
    and produce a summary table showing fraction/count of samples
    where each aggregation method satisfied each axiom.
    """
    BASE = f"/data2/rsalgani/Prefix/{dataset}/agg_files"   # change if needed
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
  
def gather_kendall_results_old(dataset): 
   
    BASE = f"/data2/rsalgani/Prefix/{dataset}/agg_files"   # change if needed
    pattern = os.path.join(BASE, "sample_*", "kendall_results.pkl")

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    dfs = []
    sample_names = []

    for p in paths:
        data = pickle.load(open(p, "rb"))   
        df = pd.DataFrame(data, columns=["method", "KT"])
        df['sample'] = os.path.basename(os.path.dirname(p))
        dfs.append(df)
        sample_names.append(os.path.basename(os.path.dirname(p))) 

    # ipdb.set_trace()
    # Stack into one MultiIndex frame: (sample, method) x axioms
    big = pd.concat(dfs)
    methods_df = big.groupby('method')['KT'].agg({'mean', 'std'})
    print(methods_df)
    # # Save outputs if you want
    out_dir = os.path.join(BASE, "summary_kt")
    os.makedirs(out_dir, exist_ok=True)
    methods_df.to_csv(os.path.join(out_dir, "kt_performance.csv"))
    print(f"\nSaved to: {out_dir}")


def gather_kendall_results(dataset, group_order=None):
    """
    Aggregates Kendall Tau results across samples.

    Supports:
      - old format: list[(method, KT)]
      - new format: list[{"method","overall","by_group","counts"}]

    Writes:
      summary_kt/kt_overall.csv
      summary_kt/kt_by_group.csv
      summary_kt/kt_long.csv
    """
    BASE = f"/data2/rsalgani/Prefix/{dataset}/agg_files"
    # If you've switched filenames, update this:
    pattern_old = os.path.join(BASE, "sample_*", "kendall_results.pkl")
    pattern_new = os.path.join(BASE, "sample_*", "kendall_results_by_group.pkl")

    paths = sorted(glob.glob(pattern_new))
    fmt = "new"
    if not paths:
        paths = sorted(glob.glob(pattern_old))
        fmt = "old"

    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern_new} or {pattern_old}")

    rows = []
    for p in paths:
        sample = os.path.basename(os.path.dirname(p))
        data = pickle.load(open(p, "rb"))

        # -------------------------
        # NEW FORMAT: list of dicts
        # -------------------------
        if fmt == "new" or (len(data) > 0 and isinstance(data[0], dict)):
            for r in data:
                method = r["method"]
                overall = r.get("overall", np.nan)

                by_group = r.get("by_group", {}) or {}
                counts = r.get("counts", {}) or {}

                # record overall
                rows.append({
                    "sample": sample,
                    "method": method,
                    "metric": "overall",
                    "group": None,
                    "KT": overall,
                    "n_users": None
                })

                # record each group
                for g, kt in by_group.items():
                    rows.append({
                        "sample": sample,
                        "method": method,
                        "metric": "group",
                        "group": g,
                        "KT": kt,
                        "n_users": counts.get(g, None)
                    })

        # -------------------------
        # OLD FORMAT: list of tuples
        # -------------------------
        else:
            df = pd.DataFrame(data, columns=["method", "KT"])
            for _, row in df.iterrows():
                rows.append({
                    "sample": sample,
                    "method": row["method"],
                    "metric": "overall",
                    "group": None,
                    "KT": row["KT"],
                    "n_users": None
                })

    big_long = pd.DataFrame(rows)

    # optional: enforce a fixed group order (e.g. [0,1,2])
    if group_order is not None:
        big_long["group"] = pd.Categorical(big_long["group"], categories=group_order, ordered=True)

    # -------------------------
    # Summaries
    # -------------------------
    overall_long = big_long[big_long["metric"] == "overall"].copy()
    overall_summary = overall_long.groupby("method")["KT"].agg(mean="mean", std="std", count="count").sort_values("mean", ascending=False)

    group_long = big_long[big_long["metric"] == "group"].copy()
    group_summary = (
        group_long.groupby(["method", "group"])["KT"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values(["method", "group"])
    )

    # Pivot group summary to match your “columns per group” style
    group_pivot = group_summary.pivot(index="method", columns="group", values=["mean", "std", "count"])
    # nice ordering: (group -> mean/std/count)
    group_pivot = group_pivot.sort_index(axis=1, level=1)

    # -------------------------
    # Save
    # -------------------------
    out_dir = os.path.join(BASE, "summary_kt")
    os.makedirs(out_dir, exist_ok=True)

    overall_summary.to_csv(os.path.join(out_dir, "kt_overall.csv"))
    group_pivot.to_csv(os.path.join(out_dir, "kt_by_group.csv"))
    big_long.to_csv(os.path.join(out_dir, "kt_long.csv"), index=False)

    print(overall_summary.head(20))
    print(group_pivot.head(20))
    print(f"\nSaved to: {out_dir}")
    return overall_summary, group_pivot, big_long

def gather_div_cvg_results(dataset): 
    BASE = f"/data2/rsalgani/Prefix/{dataset}/agg_files"   # change if needed
    pattern = os.path.join(BASE, "sample_*", "pop_group_coverage*.pkl")
    
    paths = sorted(glob.glob(pattern))
    
    dfs = []
    sample_names = []
    ipdb.set_trace() 
    for p in paths:
        data = pickle.load(open(p, "rb"))   
        df = pd.DataFrame(data).T
        df = df.reset_index(names="method")
        df['sample'] = os.path.basename(os.path.dirname(p))
        dfs.append(df)
        sample_names.append(os.path.basename(os.path.dirname(p))) 

    ipdb.set_trace()
    # Stack into one MultiIndex frame: (sample, method) x axioms
    big = pd.concat(dfs)
    bins = [4, 3, 2, 0, 1]   # your bin columns
    out = big.groupby(["method"])[bins].agg(["mean","std"])
    
    # # Save outputs if you want
    out_dir = os.path.join(BASE, "summary_div")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "div_cvg_performance.csv"))
    print(f"\nSaved to: {out_dir}")

def gather_div_pct_results(dataset): 
    BASE = f"/data2/rsalgani/Prefix/{dataset}/agg_files"   # change if needed
    pattern = os.path.join(BASE, "sample_*", "pop_group_percentage*.pkl")
    
    paths = sorted(glob.glob(pattern))
    
    dfs = []
    sample_names = []
    
    for p in paths:
        data = pickle.load(open(p, "rb"))   
        df = pd.DataFrame(data).T
        df = df.reset_index(names="method")
        df['sample'] = os.path.basename(os.path.dirname(p))
        dfs.append(df)
        sample_names.append(os.path.basename(os.path.dirname(p))) 

    # Stack into one MultiIndex frame: (sample, method) x axioms
    big = pd.concat(dfs)
    bins = [4, 3, 2, 0, 1]   # your bin columns
    out = big.groupby(["method"])[bins].agg(["mean","std"])
    print(out)
    
    # # Save outputs if you want
    out_dir = os.path.join(BASE, "summary_div")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "div_pct_performance.csv"))
    print(f"\nSaved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        default="ml-1m",
        choices=["ml-1m", "goodreads"],
    )
    args = parser.parse_args()
    gather_axiom_satisfaction_results(args.dataset)
    gather_div_cvg_results(args.dataset)
    gather_div_pct_results(args.dataset)
    gather_kendall_results(args.dataset)