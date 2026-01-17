import numpy as np
import csv
import os
from collections import defaultdict
import argparse
import sys
from tqdm import tqdm 
import ipdb 
import pandas as pd 
import pickle 

from utils.vanilla_aggregation_methods import * 
from utils.fair_aggregation_methods import *

# =============================================================================
# Data Loading
# =============================================================================

def load_rankings_to_list(filepath="recommendations.csv"):
    """
    Load recommendations from CSV file.
    Expected Format: User_ID,Movie_ID,Estimated_Rating
    """
    print(f"Loading data from '{filepath}'...")
    
    user_items = defaultdict(list)
    all_items = set()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check headers
            if not {'User_ID', 'Movie_ID', 'Estimated_Rating'}.issubset(reader.fieldnames):
                print("Error: CSV must have columns: User_ID, Movie_ID, Estimated_Rating")
                sys.exit(1)

            for row in tqdm(reader):
                uid = row['User_ID']
                iid = row['Movie_ID']
                try:
                    score = float(row['Estimated_Rating'])
                except ValueError:
                    continue 
                
                user_items[uid].append((iid, score))
                all_items.add(iid)

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    # Convert to sorted lists of items
    rankings = []
    for uid, items_scores in user_items.items():
        # Sort by score descending
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
      - Movie_ID
      - Estimated_Rating

    Returns:
      rankings_df: DataFrame with columns [User_ID, Ranked_Items]
                   Ranked_Items is a list sorted by Estimated_Rating desc
      all_items: set of unique Movie_IDs
      long_df: cleaned long-form DataFrame
               columns [User_ID, Movie_ID, Estimated_Rating]
    """
    print(f"Loading data from '{filepath}'...")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    required = {"User_ID", "Movie_ID", "Estimated_Rating"}
    if not required.issubset(df.columns):
        print("Error: CSV must have columns: User_ID, Movie_ID, Estimated_Rating")
        sys.exit(1)

    # Coerce rating to numeric, drop bad rows
    df["Estimated_Rating"] = pd.to_numeric(df["Estimated_Rating"], errors="coerce")
    df = df.dropna(subset=["Estimated_Rating"])

    # Sort by user, then rating descending
    df = df.sort_values(["User_ID", "Estimated_Rating"], ascending=[True, False])

    # Build rankings per user
    rankings_df = (
        df.groupby("User_ID")["Movie_ID"]
          .apply(list)
          .reset_index(name="Ranked_Items")
    )

    all_items = set(df["Movie_ID"].unique())

    print(f"Loaded rankings for {len(rankings_df)} users.")
    print(f"Total unique items found: {len(all_items)}")

    return rankings_df, all_items

# =============================================================================
# Helper Functions
# =============================================================================

def format_sampled_rankings(sampled_preferences):
    """
    Convert sampled preferences DataFrame to list of rankings.
    """
    rankings = []
    for _, row in sampled_preferences.iterrows():
        rankings.append(row['Ranked_Items'])
    return rankings

# =============================================================================
# Sampling Logic
# =============================================================================

def generate_sample_sets(n_samples, n_users, n_items, all_items, preferences): 
    all_users = list(preferences['User_ID'].unique())
    dfs, items, users = [], [], []
    for seed in range(n_samples):
        user_idx = np.random.choice(len(all_users), size=n_users, replace=False)
        item_idx = np.random.choice(len(all_items), size=n_items, replace=False)
        sampled_pref = preferences[preferences['User_ID'].isin([all_users[i] for i in user_idx])]
        sampled_pref['Ranked_Items'] = sampled_pref['Ranked_Items'].apply(lambda x: [item for item in x if item in item_idx])
        dfs.append(sampled_pref)
        items.append(item_idx)
        users.append(user_idx)
    return dfs, items, users


# =============================================================================
# Main Execution
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

FAIR_METHODS = { 'KuhlmanConsensus': Consensus,
                'FairMedian': FairILP, 
                }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='recommendations.csv', help='Input CSV file')
    parser.add_argument('--outdir', '-o', default='consensus_results', help='Output directory')
    parser.add_argument('--group-file', '-g', default='data/ml-1m/item_groups.pkl', help='Group attributes CSV file')
    parser.add_argument('--top-k', '-k', type=int, default=3453, help='Items to save per file')
    parser.add_argument('--user-sample-size', '-us', type=int, default=10, help='Number of users to sample')
    parser.add_argument('--item-sample-size', '-is', type=int, default=20, help='Number of items to sample')
    parser.add_argument('--n-samples', '-n', type=int, default=1, help='Number of samples to generate')
    args = parser.parse_args()

    print("=" * 60)
    print("Rank Aggregation (Separate Output Files)")
    print("=" * 60)

    # 1. Create Output Directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created directory: {args.outdir}")
    else:
        print(f"Using existing directory: {args.outdir}")

    # 2. Load Data
    # ipdb.set_trace() 
    rankings, all_items = load_rankings_to_df(args.input) #load_rankings_to_list(args.input)
    group_df = pickle.load(open(args.group_file, 'rb'))
    
    # 3. Generate Sample Sets
    sampled_rankings, sampled_items, sampled_users = generate_sample_sets(n_samples=args.n_samples, n_users=args.user_sample_size, n_items=args.item_sample_size, all_items=list(all_items), preferences=rankings)
    formatted_sampled_rankings  = [format_sampled_rankings(sampled_ranking) for sampled_ranking in sampled_rankings]
    
    # 3. Process each method and save immediately
    for seed in range(args.n_samples):
        os.makedirs(os.path.join(args.outdir, f"sample_{seed}"), exist_ok=True)
        write_dir = os.path.join(args.outdir, f"sample_{seed}")
        
        pickle.dump(sampled_rankings[seed], open(os.path.join(write_dir, "sampled_rankings.pkl"), 'wb'))
        pickle.dump(sampled_items[seed], open(os.path.join(write_dir, "sampled_items.pkl"), 'wb'))
        pickle.dump(sampled_users[seed], open(os.path.join(write_dir, "sampled_users.pkl"), 'wb'))
        
        total = len(VANILLA_METHODS) + len(FAIR_METHODS)
        print(f"\nProcessing {total} methods...")
        
        for i, (name, method) in enumerate(VANILLA_METHODS.items(), 1):
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            
            # Calculate Ranking
            result = method(formatted_sampled_rankings[seed], sampled_items[seed])
            # ipdb.set_trace() 
            # Construct Filename
            file_name = f"{name}.txt"
            file_path = os.path.join(write_dir, file_name)
            
            # Write to File
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Method: {name}\n")
                f.write(f"# Rank ItemID Score\n")
                for rank, (item, score) in enumerate(result, 1):
                    f.write(f"{rank} {item} {score:.6f}\n")
            
            print(f"-> Saved to {file_path}")
            
        alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes = process_for_fair_ranking(sampled_items[seed], group_df, formatted_sampled_rankings[seed])
        for i, (name, method) in enumerate(FAIR_METHODS.items(), len(VANILLA_METHODS)+1):
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            
            # Calculate Ranking
            result = method(alphas, betas, ranks_for_fairness, attributes_map, num_attributes)
            result = [idx_to_item[i] for i in result]
            # ipdb.set_trace() 
            # Construct Filename
            file_name = f"{name}.txt"
            file_path = os.path.join(write_dir, file_name)
            
            # Write to File
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Method: {name}\n")
                # f.write(f"# Rank ItemID Score\n")
                for rank, item, in enumerate(result, 1):
                    f.write(f"{rank} {item}\n")
            
            print(f"-> Saved to {file_path}")


        print("\n" + "=" * 60)
        print("Aggregation Complete!")
        print(f"All files are located in: ./{args.outdir}/")
        print("=" * 60)

if __name__ == "__main__":
    main()