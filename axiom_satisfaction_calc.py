import os
import csv
import glob
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import argparse
import ipdb 
from tqdm import tqdm 
import pandas as pd 
import math 

# =============================================================================
# 1. Helper Functions
# =============================================================================
def preprocess_preferences(preferences):
    # groupby once instead of filtering many times
    user_to_set = preferences.groupby("User_ID")["Movie_ID"].apply(set).to_dict()
    movie_to_users = preferences.groupby("Movie_ID")["User_ID"].apply(list).to_dict()
    n_users = len(user_to_set)
    all_candidates = list(movie_to_users.keys())
    return user_to_set, movie_to_users, n_users, all_candidates



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

# =============================================================================
# 3. Axiom Checks
# =============================================================================

def JR_check_satisfaction_given_committee(proposed_committee, user_to_set, movie_to_users, n_users):
    """
    Efficient JR check using precomputed maps.
    """
    k = len(proposed_committee)
    if k == 0:
        return False

    committee_set = set(proposed_committee)
    threshold = math.ceil(n_users / k)

    for candidate, approving_users in tqdm(movie_to_users.items(), total=len(movie_to_users)):
        # count approving voters whose approval set is disjoint from committee
        counter = 0
        for u in approving_users:
            if user_to_set[u].isdisjoint(committee_set):
                counter += 1
                if counter == threshold:
                    return False
    return True

def PJR_check_satisfaction_given_committee(proposed_committee, partial_lists, l_cohesive, n, k, verbose=False):
    for l in range(1, k+1): #iterate through l, increasing from 1
            voter_sets = l_cohesive[l]['voter_sets']
            candidate_sets = l_cohesive[l]['candidate_sets']
            for i in range(len(candidate_sets)): #go through all bicliques found in the graph
                voter_group, candidate_group = voter_sets[i], candidate_sets[i] #cohesive groups of voters agreeing on specific group of cancidates
                if len(voter_group) >= l*(n/k) and len(candidate_group) >= l: #we need voter group of size ln/k agreeing on l-sized group of candidates
                    approval_set = partial_lists[partial_lists['Voter Name'].isin(voter_group)]['Item Code'].unique(
                    ).tolist() #find the union of candidates that they all agree on 
                    if len(np.intersect1d(approval_set, proposed_committee)) < l: #if less than l of them in the committee W
                        return False       
    return True
# =============================================================================
# 3. Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--users', '-u', default='recommendations.csv', help='Path to user recommendations CSV')
    parser.add_argument('--agg', '-a', default='consensus_results', help='Directory containing consensus TXT files')
    parser.add_argument('--pref', '-p', default='recommendations.csv', help='Path to user recommendations CSV')
    args = parser.parse_args()

    # 1. Load User-Level Fully Ordered Preference Lists Data
    preferences = pd.read_csv(args.pref)
    number_voters = len(preferences['User_ID'].unique())
    number_candidates = len(preferences['Movie_ID'].unique())
    
    user_to_set, movie_to_users, n_users, all_candidates = preprocess_preferences(preferences)

    # 2. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    print("=" * 70)
    print(f"{'Method':<20} | {'Avg Kendall Tau':<15} | {'Interpretation'}")
    print("-" * 70)

    results = []

    # 3. Iterate over each method
    for file_path in consensus_files[:3]:
        satisfaction = {}
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load the ranking
        committee = load_consensus_ranking(file_path)
        if not committee:
            continue
            
        # Calculate Metric
        satisfaction['JR'] = JR_check_satisfaction_given_committee(
            committee,
            user_to_set=user_to_set,
            movie_to_users=movie_to_users,
            n_users=n_users,
        )
        results.append((method_name, satisfaction))

    # 4. Print Table
    for rank_agg_method, axiom in results:        
        print(f"{rank_agg_method:<20} | {axiom}")

    print("=" * 70)
    
if __name__ == "__main__":
    main()