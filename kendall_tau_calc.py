import os
import csv
import glob
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import argparse

# =============================================================================
# 1. Data Loading
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
                iid = row['Movie_ID']
                try:
                    score = float(row['Estimated_Rating'])
                except ValueError:
                    continue
                user_items[uid].append((iid, score))
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return {}

    # Sort and strip scores to get pure ranked lists
    user_rankings = {}
    for uid, items in user_items.items():
        # Sort desc by score
        items.sort(key=lambda x: x[1], reverse=True)
        # Keep only item IDs
        user_rankings[uid] = [iid for iid, score in items]
        if uid == '5412': print(len(user_rankings[uid]))
        
    print(f"Loaded {len(user_rankings)} user lists.")
    
    return user_rankings

def load_consensus_ranking(file_path):
    """
    Loads a consensus ranking file (rank item score).
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
                    # Format: Rank ItemID Score
                    # We only care about the ItemID and its Rank (order)
                    rank = int(parts[0])
                    item_id = parts[1]
                    rank_map[item_id] = rank
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
        
    return rank_map

# =============================================================================
# 2. Kendall Tau Calculation
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
            continue  # Cannot calc distance for list of 0 or 1 items
            
        # 1. Define User's defined order (ground truth for this calculation)
        user_ranks = list(range(len(user_list)))
        
        # 2. Get the Consensus ranks for these specific items
    
        default_rank = 100000 
        consensus_ranks = [consensus_map.get(item, default_rank) for item in user_list]
        
        # 3. Calculate Kendall Tau
       
        tau, _ = kendalltau(user_ranks, consensus_ranks)
        
        if not np.isnan(tau):
            taus.append(tau)
            
    return np.mean(taus) if taus else 0.0

# =============================================================================
# 3. Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--users', '-u', default='recommendations.csv', help='Path to user recommendations CSV')
    parser.add_argument('--consensus', '-c', default='consensus_results', help='Directory containing consensus TXT files')
    args = parser.parse_args()

    # 1. Load User Data
    user_rankings = load_user_lists(args.users)
    if not user_rankings:
        return

    # 2. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.consensus, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.consensus}")
        return

    print(f"\nComparing {len(consensus_files)} consensus methods against {len(user_rankings)} users...")
    print("=" * 70)
    print(f"{'Method':<20} | {'Avg Kendall Tau':<15} | {'Interpretation'}")
    print("-" * 70)

    results = []

    # 3. Iterate over each method
    for file_path in consensus_files:
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load the ranking
        consensus_map = load_consensus_ranking(file_path)
        
        if not consensus_map:
            continue
            
        # Calculate Metric
        avg_tau = calculate_average_tau(user_rankings, consensus_map)
        results.append((method_name, avg_tau))

    # 4. Sort by highest correlation
    results.sort(key=lambda x: x[1], reverse=True)

    # 5. Print Table
    for name, score in results:
        # Quick visual interpretation
        interp = ""
        if score > 0.5: interp = "High Agreement"
        elif score > 0.1: interp = "Positive"
        elif score > -0.1: interp = "Uncorrelated"
        else: interp = "Negative/Inverse"
        
        print(f"{name:<20} | {score:13.4f}   | {interp}")

    print("=" * 70)
    print("Note: Kendall Tau ranges from 1.0 (Identical) to -1.0 (Completely Reversed).")
    print("A higher score means the consensus ranking preserves the individual user preferences better.")

if __name__ == "__main__":
    main()