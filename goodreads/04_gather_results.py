"""
Goodreads Results Gatherer
==========================
Aggregates results from multiple sample runs for final analysis.

This script collects:
- Axiom satisfaction results
- Diversity metrics
- Kendall Tau correlations

And produces summary tables showing aggregate statistics across all samples.
"""

import os
import glob
import pickle
import pandas as pd
import argparse


def gather_axiom_satisfaction_results(base_dir):
    """
    Gather axiom satisfaction results from multiple sample directories,
    and produce a summary table showing fraction/count of samples
    where each aggregation method satisfied each axiom.
    """
    pattern = os.path.join(base_dir, "sample_*", "axiom_satisfaction_results.pkl")

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No axiom satisfaction files found matching: {pattern}")
        return None

    dfs = []
    sample_names = []

    for p in paths:
        df = pickle.load(open(p, "rb"))
        df = df.astype(bool)
        dfs.append(df)
        sample_names.append(os.path.basename(os.path.dirname(p)))

    # Stack into one MultiIndex frame: (sample, method) x axioms
    big = pd.concat(dfs, keys=sample_names, names=["sample", "method"])

    # Fraction of samples where True (mean over samples)
    frac_true = big.groupby("method").mean(numeric_only=True)

    # Count of samples where True (sum over samples)
    count_true = big.groupby("method").sum(numeric_only=True).astype(int)

    # Count how many samples actually had that method
    n_samples_per_method = big.groupby("method").size()

    # Combined view: "count/total (fraction)"
    summary = count_true.copy()
    for col in summary.columns:
        summary[col] = (
            summary[col].astype(str) + "/" + n_samples_per_method.astype(str) +
            " (" + (frac_true[col] * 100).round(1).astype(str) + "%)"
        )

    return summary, frac_true, count_true


def gather_diversity_results(base_dir):
    """
    Gather diversity metrics from multiple sample directories.
    """
    coverage_pattern = os.path.join(base_dir, "sample_*", "pop_group_coverage_diversity.pkl")
    percentage_pattern = os.path.join(base_dir, "sample_*", "pop_group_percentage_diversity.pkl")
    
    coverage_paths = sorted(glob.glob(coverage_pattern))
    percentage_paths = sorted(glob.glob(percentage_pattern))
    
    if not coverage_paths:
        print(f"No coverage diversity files found matching: {coverage_pattern}")
        return None, None
    
    coverage_dfs = []
    percentage_dfs = []
    sample_names = []
    
    for p in coverage_paths:
        df = pickle.load(open(p, "rb"))
        coverage_dfs.append(df)
        sample_names.append(os.path.basename(os.path.dirname(p)))
    
    for p in percentage_paths:
        df = pickle.load(open(p, "rb"))
        percentage_dfs.append(df)
    
    # Average across samples
    coverage_avg = pd.concat(coverage_dfs).groupby(level=0).mean()
    percentage_avg = pd.concat(percentage_dfs).groupby(level=0).mean()
    
    return coverage_avg, percentage_avg


def gather_kendall_tau_results(base_dir):
    """
    Gather Kendall Tau results from multiple sample directories.
    """
    pattern = os.path.join(base_dir, "sample_*", "kendall_tau_results.csv")
    
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No Kendall Tau files found matching: {pattern}")
        return None
    
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
    
    # Average across samples
    combined = pd.concat(dfs)
    avg_tau = combined.groupby('Method')['Avg_Kendall_Tau'].mean()
    
    return avg_tau.sort_values(ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Goodreads Results Gatherer")
    parser.add_argument('--base-dir', '-b', default='consensus_results',
                        help='Base directory containing sample_* subdirectories')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory for summary files (default: base_dir/summary)')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, "summary")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GOODREADS RESULTS SUMMARY")
    print("=" * 70)
    
    # Gather Axiom Satisfaction
    print("\n1. Axiom Satisfaction Results")
    print("-" * 70)
    axiom_results = gather_axiom_satisfaction_results(base_dir)
    if axiom_results:
        summary, frac_true, count_true = axiom_results
        print(summary.to_string())
        
        # Save
        summary.to_csv(os.path.join(output_dir, "axiom_satisfaction_summary.csv"))
        frac_true.to_csv(os.path.join(output_dir, "axiom_satisfaction_fractions.csv"))
        print(f"\nSaved to: {output_dir}/axiom_satisfaction_*.csv")
    
    # Gather Diversity Results
    print("\n2. Diversity Results")
    print("-" * 70)
    coverage_avg, percentage_avg = gather_diversity_results(base_dir)
    if coverage_avg is not None:
        print("\nAverage Coverage by Group:")
        print(coverage_avg.to_string())
        print("\nAverage Percentage by Group:")
        print(percentage_avg.to_string())
        
        # Save
        coverage_avg.to_csv(os.path.join(output_dir, "diversity_coverage_avg.csv"))
        percentage_avg.to_csv(os.path.join(output_dir, "diversity_percentage_avg.csv"))
        print(f"\nSaved to: {output_dir}/diversity_*.csv")
    
    # Gather Kendall Tau Results
    print("\n3. Kendall Tau Results")
    print("-" * 70)
    tau_results = gather_kendall_tau_results(base_dir)
    if tau_results is not None:
        print("\nAverage Kendall Tau by Method:")
        print(tau_results.to_string())
        
        # Save
        tau_results.to_csv(os.path.join(output_dir, "kendall_tau_avg.csv"))
        print(f"\nSaved to: {output_dir}/kendall_tau_avg.csv")
    
    print("\n" + "=" * 70)
    print(f"All summary files saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

