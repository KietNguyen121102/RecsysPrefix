import numpy as np
import csv
import os
from collections import defaultdict
import argparse
import sys
from tqdm import tqdm 
import ipdb 

from utils.aggregation_methods import * 
# =============================================================================
# Data Loading
# =============================================================================

def load_rankings_from_csv(filepath="recommendations.csv"):
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

# =============================================================================
# Helper Functions
# =============================================================================


# =============================================================================
# Sampling Logic
# =============================================================================

def generate_sample_sets(n_samples, n_users, n_items, preferences): 
    all_users = list(preferences['User_ID'].unique())
    all_items = list(preferences['Movie_ID'].unique())
    slices = [] 
    for seed in range(n_samples):
        user_idx = np.random.choice(len(all_users), size=n_users, replace=False)
        item_idx = np.random.choice(len(all_items), size=n_items, replace=False)
        sampled_pref = preferences[preferences['User_ID'].isin([all_users[i] for i in user_idx]) & preferences['Movie_ID'].isin([all_items[i] for i in item_idx])]
        slices.append(sampled_pref)
    return slices  


# =============================================================================
# Main Execution
# =============================================================================

ALL_METHODS = {
    'CombMIN': comb_min, 'CombMAX': comb_max, 'CombSUM': comb_sum,
    'CombANZ': comb_anz, 'CombMNZ': comb_mnz,
    'MC1': mc1, 'MC2': mc2, 'MC3': mc3, 'MC4': mc4,
    'BordaCount': borda_count, 'Dowdall': dowdall,
    'Median': median_rank, 'Mean': mean_rank,
    'RRF': rrf, 'iRANK': irank, 'ER': er,
    'PostNDCG': postndcg, 'CG': cg, 'DIBRA': dibra
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='recommendations.csv', help='Input CSV file')
    parser.add_argument('--outdir', '-o', default='consensus_results', help='Output directory')
    parser.add_argument('--top-k', '-k', type=int, default=3453, help='Items to save per file')
    parser.add_argument('--user-sample-size', '-us', type=int, default=100, help='Number of usersto sample')
    parser.add_argument('--item-sample-size', '-is', type=int, default=100, help='Number of usersto sample')
    parser.add_argument('--n-samples', '-n', type=int, default=10, help='Number of samples to generate')
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
    ipdb.set_trace() 
    rankings, all_items = load_rankings_from_csv(args.input)

    # 3. Process each method and save immediately
    total = len(ALL_METHODS)
    print(f"\nProcessing {total} methods...")
    
    for i, (name, method) in enumerate(ALL_METHODS.items(), 1):
        print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
        
        # Calculate Ranking
        result = method(rankings, all_items)
        
        # Construct Filename
        file_name = f"{name}.txt"
        file_path = os.path.join(args.outdir, file_name)
        
        # Write to File
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Method: {name}\n")
            f.write(f"# Rank ItemID Score\n")
            for rank, (item, score) in enumerate(result[:args.top_k], 1):
                f.write(f"{rank} {item} {score:.6f}\n")
        
        print(f"-> Saved to {file_path}")

    print("\n" + "=" * 60)
    print("Aggregation Complete!")
    print(f"All files are located in: ./{args.outdir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()