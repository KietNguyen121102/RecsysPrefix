import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import argparse
from tqdm import tqdm 
import pandas as pd 
import math 
import pickle 
import glob 
import os 
import time 

from utils.diversity_metrics import calculate_pop_group_item_diversity
from utils.metric_logging import write_div_metric_dict_as_rows
from utils.io import load_consensus_ranking, load_sampled_preferences

"""
Goodreads Diversity Calculator
==============================
Evaluates diversity of consensus rankings across popularity groups.

Metrics computed:
- Coverage: Fraction of items from each group represented in top-k
- Percentage: Fraction of top-k that belongs to each group
"""

# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Goodreads Diversity Calculator")
    parser.add_argument('--agg', '-a', default='consensus_results/sample_0', 
                        help='Directory containing consensus TXT files')
    parser.add_argument('--pref', '-p', default='consensus_results/sample_0/sampled_rankings.pkl', 
                        help='Path to sampled preferences pickle')
    parser.add_argument('--group-file', '-g', default='data/goodreads/item_groups.pkl', 
                        help='Item groups pickle file')
    parser.add_argument('--top-k', '-k', type=int, default=10,
                        help='Top-k items to evaluate')
    args = parser.parse_args()

    print("=" * 70)
    print("GOODREADS DIVERSITY EVALUATION")
    print("=" * 70)
    print(f"(1) Loading data")
    print("-" * 70)
    
    # 1. Load User-Level Fully Ordered Preference Lists Data
    preferences = load_sampled_preferences(args.pref)
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    
    # Load item groups
    if not os.path.exists(args.group_file):
        print(f"Error: Group file '{args.group_file}' not found.")
        print("Run utils/generate_groups.py first to create item groups.")
        return
    
    group_df = pickle.load(open(args.group_file, 'rb'))
    
    print(f"(2) Loading Consensus Files")
    print("-" * 70)
    
    # Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    print("-" * 70)
    print('RUN STATS')
    print("-" * 70)
    print("Number of Voters:", number_voters)
    print("Number of Candidates:", number_candidates)
    print("Number of methods to evaluate:", len(consensus_files))
    print("-" * 70)
    
    print(f"(3) Calculating Population Group Diversity Metrics")
    cvg, pct = [], [] 
    method_names = []
    
    for idx, file_path in enumerate(consensus_files):
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"({idx+1}/{len(consensus_files)}) Method: {method_name}")
        method_names.append(method_name)
        
        # Load the ranking
        committee = load_consensus_ranking(file_path)
        if not committee:
            continue
            
        coverage_results, percentage_results = calculate_pop_group_item_diversity(
            committee, group_df, k=args.top_k
        )
        cvg.append(coverage_results)
        pct.append(percentage_results)
    
    # Create result DataFrames
    cvg_df = pd.DataFrame(cvg, index=method_names)
    pct_df = pd.DataFrame(pct, index=method_names)
    
    # Save results
    pickle.dump(cvg_df, open(os.path.join(args.agg, 'pop_group_coverage_diversity.pkl'), 'wb'))
    pickle.dump(pct_df, open(os.path.join(args.agg, 'pop_group_percentage_diversity.pkl'), 'wb'))
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"COVERAGE DIVERSITY (Top-{args.top_k})")
    print("-" * 70)
    print(cvg_df.to_string())
    
    print("\n" + "=" * 70)
    print(f"PERCENTAGE DIVERSITY (Top-{args.top_k})")
    print("-" * 70)
    print(pct_df.to_string())
    
    print("\n" + "=" * 70)
    print(f"Diversity metrics saved to {args.agg}")
    print("=" * 70)


if __name__ == "__main__":
    main()

