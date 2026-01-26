import os
import csv
import glob
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import argparse

from utils.io import load_sampled_preferences, load_consensus_ranking

"""
Goodreads Kendall Tau Calculator
================================
Evaluates how well consensus rankings preserve individual user preferences
by computing the average Kendall Tau correlation.

Kendall Tau ranges from:
  - 1.0: Identical orderings
  - 0.0: Uncorrelated
  - -1.0: Completely reversed
"""

# =============================================================================
# Data Loading
# =============================================================================

def load_user_lists(csv_path):
    """
    Loads user lists from recommendations.csv.
    Returns: { 'User_ID': [item1, item2, item3...] }
    Items are sorted by the estimated rating (descending).
    """
    print(f"Loading user lists from {csv_path}...")
    user_items = defaultdict(list)
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row['User_ID']
                iid = row['Book_ID']
                try:
                    score = float(row['Estimated_Rating'])
                except ValueError:
                    continue
                user_items[uid].append((iid, score))
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return {}

    user_rankings = {}
    for uid, items in user_items.items():
        items.sort(key=lambda x: x[1], reverse=True)
        user_rankings[uid] = [iid for iid, score in items]
        
    print(f"Loaded {len(user_rankings)} user lists.")
    
    return user_rankings


def load_consensus_ranking_map(file_path):
    """
    Loads a consensus ranking file.
    Returns a Dictionary: { 'ItemID': Rank_Integer }
    """
    rank_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    rank = int(parts[0])
                    item_id = parts[1]
                    rank_map[item_id] = rank
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
        
    return rank_map


# =============================================================================
# Kendall Tau Calculation
# =============================================================================

def calculate_average_tau(user_rankings, consensus_map):
    """
    Calculates the average Kendall Tau across all users.
    
    Method:
    1. For each user, extract their list of items.
    2. Look up the rank of these items in the Consensus Map.
    3. Compare the User's Rank (0, 1, 2...) vs Consensus Rank.
    """
    taus = []
    
    for uid, user_list in user_rankings.items():
        if len(user_list) < 2:
            continue
            
        user_ranks = list(range(len(user_list)))
        default_rank = 100000 
        consensus_ranks = [consensus_map.get(str(item), default_rank) for item in user_list]
        
        tau, _ = kendalltau(user_ranks, consensus_ranks)
        
        if not np.isnan(tau):
            taus.append(tau)
            
    return np.mean(taus) if taus else 0.0


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Goodreads Kendall Tau Calculator")
    parser.add_argument('--pref', '-p', default='recommendations.csv', help='Path to user preferences (CSV or PKL)')
    parser.add_argument('--agg', '-a', default='consensus_results', help='Directory containing consensus TXT files')
    parser.add_argument('--output', '-o', default=None, help='Output CSV file for results')
    args = parser.parse_args()

    # 1. Load User Data - detect file type
    if args.pref.endswith('.pkl'):
        # Use sampled preferences loader
        rankings = load_sampled_preferences(args.pref)
        rankings = rankings.astype({'User_ID': str})
        user_rankings = rankings.set_index("User_ID")["Ranked_Items"].to_dict()
        print(f"Loaded {len(user_rankings)} user rankings from PKL.")
    else:
        # Use CSV loader
        user_rankings = load_user_lists(args.pref)
    
    if not user_rankings:
        print("No user rankings loaded!")
        return

    # 2. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    print(f"\nComparing {len(consensus_files)} consensus methods against {len(user_rankings)} users...")
    print("=" * 70)
    print(f"{'Method':<20} | {'Avg Kendall Tau':<15} | {'Interpretation'}")
    print("-" * 70)

    results = []

    # 3. Iterate over each method
    for file_path in consensus_files:
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        
        consensus_map = load_consensus_ranking_map(file_path)
        
        if not consensus_map:
            continue
            
        avg_tau = calculate_average_tau(user_rankings, consensus_map)
        results.append((method_name, avg_tau))

    # 4. Sort by highest correlation
    results.sort(key=lambda x: x[1], reverse=True)

    # 5. Print Table
    for name, score in results:
        if score > 0.5: interp = "High Agreement"
        elif score > 0.1: interp = "Positive"
        elif score > -0.1: interp = "Uncorrelated"
        else: interp = "Negative/Inverse"
        
        print(f"{name:<20} | {score:13.4f}   | {interp}")

    print("=" * 70)
    print("Note: Kendall Tau ranges from 1.0 (Identical) to -1.0 (Completely Reversed).")
    print("A higher score means the consensus ranking preserves the individual user preferences better.")
    
    # 6. Save results if output specified
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results, columns=['Method', 'Avg_Kendall_Tau'])
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


def run_on_sampled(pref_path, agg_path, output_path=None):
    """
    Run Kendall Tau calculation on sampled preferences.
    
    Args:
        pref_path: Path to sampled_rankings.pkl
        agg_path: Directory containing consensus TXT files
        output_path: Optional output CSV path
    """
    # Load sampled preferences
    rankings = load_sampled_preferences(pref_path)
    rankings = rankings.astype({'User_ID': str})
    user_rankings = rankings.set_index("User_ID")["Ranked_Items"].to_dict()
    
    # Find consensus files
    consensus_files = glob.glob(os.path.join(agg_path, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {agg_path}")
        return []

    print(f"\nComparing {len(consensus_files)} methods against {len(user_rankings)} users...")
    print("=" * 70)
    print(f"{'Method':<20} | {'Avg Kendall Tau':<15} | {'Interpretation'}")
    print("-" * 70)

    results = []

    for file_path in consensus_files:
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        consensus_map = load_consensus_ranking_map(file_path)
        
        if not consensus_map:
            continue
            
        avg_tau = calculate_average_tau(user_rankings, consensus_map)
        results.append((method_name, avg_tau))

    results.sort(key=lambda x: x[1], reverse=True)

    for name, score in results:
        if score > 0.5: interp = "High Agreement"
        elif score > 0.1: interp = "Positive"
        elif score > -0.1: interp = "Uncorrelated"
        else: interp = "Negative/Inverse"
        
        print(f"{name:<20} | {score:13.4f}   | {interp}")

    print("=" * 70)
    
    if output_path:
        import pandas as pd
        df = pd.DataFrame(results, columns=['Method', 'Avg_Kendall_Tau'])
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()



