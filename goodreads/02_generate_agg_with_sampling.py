import numpy as np
import csv
import os
from collections import defaultdict
import argparse
import sys
from tqdm import tqdm 
import pandas as pd 
import pickle 

from utils.vanilla_aggregation_methods import * 
from utils.fair_aggregation_methods import *

"""
Goodreads Rank Aggregation with Sampling
========================================
This script:
1. Loads full user rankings from recommendations.csv
2. Samples subsets of users and items
3. Applies multiple rank aggregation methods
4. Saves consensus rankings for evaluation

Output structure:
    consensus_results/
    ├── sample_0/
    │   ├── sampled_rankings.pkl
    │   ├── sampled_items.pkl
    │   ├── sampled_users.pkl
    │   ├── BordaCount.txt
    │   ├── CombSUM.txt
    │   └── ...
    └── sample_1/
        └── ...
"""

# =============================================================================
# Data Loading
# =============================================================================

def load_rankings_to_list(filepath="recommendations.csv"):
    """
    Load recommendations from CSV file.
    Expected Format: User_ID,Book_ID,Estimated_Rating
    """
    print(f"Loading data from '{filepath}'...")
    
    user_items = defaultdict(list)
    all_items = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if not {'User_ID', 'Book_ID', 'Estimated_Rating'}.issubset(reader.fieldnames):
                print("Error: CSV must have columns: User_ID, Book_ID, Estimated_Rating")
                sys.exit(1)

            for row in tqdm(reader):
                uid = row['User_ID']
                iid = row['Book_ID']
                try:
                    score = float(row['Estimated_Rating'])
                except ValueError:
                    continue 
                
                user_items[uid].append((iid, score))
                all_items.add(iid)

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    rankings = []
    for uid, items_scores in user_items.items():
        items_scores.sort(key=lambda x: x[1], reverse=True)
        rankings.append([iid for iid, score in items_scores])
    
    print(f"Loaded rankings for {len(rankings)} users.")
    print(f"Total unique items found: {len(all_items)}")
    
    return rankings, all_items


def load_rankings_to_df(filepath="recommendations.csv"):
    """
    Load recommendations from CSV into Pandas.

    Expected columns:
      - User_ID
      - Book_ID
      - Estimated_Rating

    Returns:
      rankings_df: DataFrame with columns [User_ID, Ranked_Items]
      all_items: set of unique Book_IDs
    """
    print(f"Loading data from '{filepath}'...")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    required = {"User_ID", "Book_ID", "Estimated_Rating"}
    if not required.issubset(df.columns):
        print("Error: CSV must have columns: User_ID, Book_ID, Estimated_Rating")
        sys.exit(1)

    df["Estimated_Rating"] = pd.to_numeric(df["Estimated_Rating"], errors="coerce")
    df = df.dropna(subset=["Estimated_Rating"])

    df = df.sort_values(["User_ID", "Estimated_Rating"], ascending=[True, False])

    rankings_df = (
        df.groupby("User_ID")["Book_ID"]
          .apply(list)
          .reset_index(name="Ranked_Items")
    )

    all_items = set(df["Book_ID"].unique())

    print(f"Loaded rankings for {len(rankings_df)} users.")
    print(f"Total unique items found: {len(all_items)}")

    return rankings_df, all_items


# =============================================================================
# Helper Functions
# =============================================================================

def format_sampled_rankings(sampled_preferences):
    """Convert sampled preferences DataFrame to list of rankings."""
    rankings = []
    for _, row in sampled_preferences.iterrows():
        rankings.append(row['Ranked_Items'])
    return rankings


# =============================================================================
# Sampling Logic
# =============================================================================

def generate_sample_sets(n_samples, n_users, n_items, all_items, preferences, seed_base=0): 
    """Generate multiple sample sets of users and items."""
    np.random.seed(seed_base)
    all_users = list(preferences['User_ID'].unique())
    dfs, items, users = [], [], []
    
    for seed in range(n_samples):
        np.random.seed(seed_base + seed)
        user_idx = np.random.choice(len(all_users), size=n_users, replace=False)
        item_idx = np.random.choice(list(all_items), size=n_items, replace=False)
        
        sampled_pref = preferences[preferences['User_ID'].isin([all_users[i] for i in user_idx])].copy()
        sampled_pref['Ranked_Items'] = sampled_pref['Ranked_Items'].apply(
            lambda x: [item for item in x if item in item_idx]
        )
        
        dfs.append(sampled_pref)
        items.append(item_idx)
        users.append(user_idx)
        
    return dfs, items, users


# =============================================================================
# Aggregation Methods
# =============================================================================

VANILLA_METHODS = {
    'CombMIN': comb_min, 'CombMAX': comb_max, 'CombSUM': comb_sum,
    'CombANZ': comb_anz, 'CombMNZ': comb_mnz,
    'MC1': mc1, 'MC2': mc2, 'MC3': mc3, 'MC4': mc4,
    'BordaCount': borda_count, 'Dowdall': dowdall,
    'Median': median_rank, 'Mean': mean_rank,
    'RRF': rrf, 'iRANK': irank, 'ER': er,
    'PostNDCG': postndcg, 'CG': cg, 'DIBRA': dibra
}

FAIR_METHODS = {
    'KuhlmanConsensus': Consensus,
    'FairMedian': FairILP, 
}


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Goodreads Rank Aggregation with Sampling")
    parser.add_argument('--input', '-i', default='recommendations.csv', help='Input CSV file')
    parser.add_argument('--outdir', '-o', default='consensus_results', help='Output directory')
    parser.add_argument('--group-file', '-g', default='data/goodreads/item_groups.pkl', help='Item groups pickle file')
    parser.add_argument('--user-sample-size', '-us', type=int, default=10, help='Number of users to sample')
    parser.add_argument('--item-sample-size', '-is', type=int, default=20, help='Number of items to sample')
    parser.add_argument('--n-samples', '-n', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print(args)
    
    print("=" * 60)
    print("GOODREADS RANK AGGREGATION")
    print("=" * 60)

    # 1. Create Output Directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created directory: {args.outdir}")
    else:
        print(f"Using existing directory: {args.outdir}")

    # 2. Load Data
    rankings, all_items = load_rankings_to_df(args.input)
    
    # Load group file if exists
    group_df = None
    if os.path.exists(args.group_file):
        group_df = pickle.load(open(args.group_file, 'rb'))
        print(f"Loaded item groups from {args.group_file}")
    else:
        print(f"Warning: Group file '{args.group_file}' not found. Fair methods will be skipped.")
    
    # 3. Generate Sample Sets
    sampled_rankings, sampled_items, sampled_users = generate_sample_sets(
        n_samples=args.n_samples, 
        n_users=args.user_sample_size, 
        n_items=args.item_sample_size, 
        all_items=list(all_items), 
        preferences=rankings,
        seed_base=args.seed
    )
    formatted_sampled_rankings = [format_sampled_rankings(sr) for sr in sampled_rankings]
    
    # 4. Process each sample
    for seed in range(args.n_samples):
        write_dir = os.path.join(args.outdir, f"sample_{seed}")
        os.makedirs(write_dir, exist_ok=True)
            
        # Save sampled data
        pickle.dump(sampled_rankings[seed], open(os.path.join(write_dir, "sampled_rankings.pkl"), 'wb'))
        pickle.dump(sampled_items[seed], open(os.path.join(write_dir, "sampled_items.pkl"), 'wb'))
        pickle.dump(sampled_users[seed], open(os.path.join(write_dir, "sampled_users.pkl"), 'wb'))
        
        total = len(VANILLA_METHODS) + (len(FAIR_METHODS) if group_df is not None else 0)
        print(f"\n[Sample {seed}] Processing {total} methods...")
        
        # Run vanilla methods
        for i, (name, method) in enumerate(VANILLA_METHODS.items(), 1):
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            
            result = method(formatted_sampled_rankings[seed], sampled_items[seed])
            
            file_name = f"{name}.txt"
            file_path = os.path.join(write_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Method: {name}\n")
                f.write(f"# Rank ItemID Score\n")
                for rank, (item, score) in enumerate(result, 1):
                    f.write(f"{rank} {item} {score:.6f}\n")
            
            print(f"-> Saved to {file_path}")
        
        # Run fair methods if group data available
        if group_df is not None:
            alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes = process_for_fair_ranking(
                sampled_items[seed], group_df, formatted_sampled_rankings[seed]
            )
            
            for i, (name, method) in enumerate(FAIR_METHODS.items(), len(VANILLA_METHODS)+1):
                print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
                
                result = method(alphas, betas, ranks_for_fairness, attributes_map, num_attributes)
                result = [idx_to_item[i] for i in result]
                
                file_name = f"{name}.txt"
                file_path = os.path.join(write_dir, file_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Method: {name}\n")
                    for rank, item in enumerate(result, 1):
                        f.write(f"{rank} {item}\n")
                
                print(f"-> Saved to {file_path}")

    print("\n" + "=" * 60)
    print("AGGREGATION COMPLETE!")
    print(f"All files are located in: {args.outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()



