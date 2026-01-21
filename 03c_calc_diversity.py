import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import argparse
import ipdb 
from tqdm import tqdm 
import pandas as pd 
import math 
import pickle 
import glob 
import os 
import time 

from utils.diversity_metrics import calculate_pop_group_item_diversity
from utils.logging import write_div_metric_dict_as_rows

# =============================================================================
# 1. Helper Functions
# =============================================================================




# =============================================================================
# 2. Data Loading
# =============================================================================

def load_consensus_ranking(file_path):
    """
    Loads a consensus ranking file (rank item score).
    Returns a Dictionary: { 'ItemID': Rank_Integer }
    """
    rank_map = {}
    item_id_list = []
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
                    item_id_list.append(int(item_id)) 
                    rank_map[item_id] = rank
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
        
    return item_id_list

def load_sampled_preferences(file_path):
    """
    Loads sampled user preferences from a CSV file.
    Expects columns: User_ID, Movie_ID, Estimated_Rating
    """
    # ipdb.set_trace() 
    preferences = pickle.load(open(file_path, 'rb')) #.explode('Ranked_Items').reset_index(drop=True)
    return preferences

# =============================================================================
# 3. Main Pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', '-a', default='/u/rsalgani/2024-2025/RecsysPrefix/consensus_results/sample_0', help='Directory containing consensus TXT files')
    parser.add_argument('--pref', '-p', default='/u/rsalgani/2024-2025/RecsysPrefix/consensus_results/sample_0/sampled_rankings.pkl', help='Path to user recommendations CSV')
    parser.add_argument('--group-file', '-g', default='data/ml-1m/item_groups.pkl', help='Group attributes CSV file')
    args = parser.parse_args()

    print("=" * 70)
    print(f"(1) Loading data")
    print("-" * 70)
    
    # 1. Load User-Level Fully Ordered Preference Lists Data
    preferences = load_sampled_preferences(args.pref)
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    group_df = pickle.load(open(args.group_file, 'rb'))
    
    print(f"(2) Loading Consensus Files")
    print("-" * 70)
    # 3. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    # ipdb.set_trace()
    # print(f"(2) Running Preprocessing")
    print("-" * 70)
    print('RUN STATS')
    print("-" * 70)
    print("Number of Voters:", number_voters)
    print("Number of Candidates:", number_candidates)
    print("Number of methods to evaluate:", len(consensus_files))
    print("-" * 70)
    
    print(f"(3) Calculating Population Group Diversity Metrics")
    cvg, pct = [], [] 
    for idx, file_path in enumerate(consensus_files):
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"({idx}/{len(consensus_files)}) Method:", method_name)
        
        # Load the ranking
        committee = load_consensus_ranking(file_path)
        if not committee:
            continue
        coverage_results, percentage_results = calculate_pop_group_item_diversity(
        committee, group_df, k=5
    )
        cvg.append(coverage_results)
        pct.append(percentage_results)
        
    cvg_df = pd.DataFrame(cvg)
    pct_df = pd.DataFrame(pct)    
    pickle.dump(cvg_df, open(os.path.join(args.agg, 'pop_group_coverage_diversity.pkl'), 'wb'))
    pickle.dump(pct_df, open(os.path.join(args.agg, 'pop_group_percentage_diversity.pkl'), 'wb'))
    print("Diversity metrics saved.") 
main() 