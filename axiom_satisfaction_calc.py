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
from cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
# from cohesive_group_search2 import maximal_bicliques_implicit_gc, add_subsets_fast

# =============================================================================
# 1. Helper Functions
# =============================================================================
def preprocess_for_JR(approvals):
    # groupby once instead of filtering many times
    user_to_set = approvals.groupby("User_ID")["Movie_ID"].apply(set).to_dict()
    movie_to_users = approvals.groupby("Movie_ID")["User_ID"].apply(list).to_dict()
    n_users = len(user_to_set)
    all_candidates = list(movie_to_users.keys())
    return user_to_set, movie_to_users, n_users, all_candidates

def prune_satisfied_for_EJR(partial_lists, proposed_committee, l):
    for voter in partial_lists['User_ID'].unique().tolist(): #look through each voter
        approval_set = partial_lists[partial_lists['User_ID'] == voter]['Movie_ID'].unique(
        ).tolist() #find the candidates he approves of
        if len(np.intersect1d(approval_set, proposed_committee)) >= l: #if the voter is satisfied
            partial_lists = partial_lists[partial_lists['User_ID'] != voter] #prune the voter 
    return partial_lists.reset_index(drop=True)

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

def PJR_check_satisfaction_given_committee(proposed_committee, partial_lists, l_cohesive):
    n, k = len(partial_lists['User_ID'].unique()), len(proposed_committee)
    for l in range(1, k+1): #iterate through l, increasing from 1
            voter_sets = l_cohesive[l]['voter_sets']
            candidate_sets = l_cohesive[l]['candidate_sets']
            for i in range(len(candidate_sets)): #go through all bicliques found in the graph
                voter_group, candidate_group = voter_sets[i], candidate_sets[i] #cohesive groups of voters agreeing on specific group of cancidates
                if len(voter_group) >= l*(n/k) and len(candidate_group) >= l: #we need voter group of size ln/k agreeing on l-sized group of candidates
                    approval_set = partial_lists[partial_lists['User_ID'].isin(voter_group)]['Movie_ID'].unique(
                    ).tolist() #find the union of candidates that they all agree on 
                    if len(np.intersect1d(approval_set, proposed_committee)) < l: #if less than l of them in the committee W
                        return False       
    return True

def EJR_check_satisfaction_given_committee(proposed_committee, partial_lists): 
    n, k = len(partial_lists['User_ID'].unique()), len(proposed_committee)
    for l in tqdm(range(1, k+1)): #iterate through l, increasing from 1
        unsatisfied_voter_set = prune_satisfied_for_EJR(partial_lists, proposed_committee, l)
        voter_sets, candidate_sets = find_maximal_cohesive_groups(unsatisfied_voter_set, committee_size=k)
        # voter_sets, cand_sets = find_maximal_cohesive_groups_groupby(partial_lists)
        for v in voter_sets: 
            if len(v) >= l*(n/k): 
                return False
    return True


def generate_samples(): 
    return 0 

# =============================================================================
# 3. Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', '-a', default='consensus_results', help='Directory containing consensus TXT files')
    parser.add_argument('--pref', '-p', default='recommendations.csv', help='Path to user recommendations CSV')
    parser.add_argument('--committee_size', '-k', type=int, default=50, help='Size of the committee to select')
    args = parser.parse_args()

    print("=" * 70)
    print(f"(1) Loading data")
    print("-" * 70)
    
    # 1. Load User-Level Fully Ordered Preference Lists Data
    preferences = pd.read_csv(args.pref)
    approvals = preferences[preferences['Estimated_Rating'] >= 4].reset_index(drop=True)  # assuming ratings are from 1 to 5
    number_voters = len(approvals['User_ID'].unique())
    number_candidates = len(approvals['Movie_ID'].unique())
    
    print(f"(2) Running Preprocessing")
    print("-" * 70)
    # 2. Preprocess preferences
    user_to_set, movie_to_users, n_users, all_candidates = preprocess_for_JR(approvals)
    voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(approvals, committee_size=args.committee_size, number_voters=number_voters)
    
    print(f"(3) Loading Consensus Files")
    print("-" * 70)
    # 3. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    print(f"(4) Calculating Axiom Satisfaction")
    results = []
    # 4. Iterate over each method
    for file_path in consensus_files[:1]:
        satisfaction = {}
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        print("Method:", method_name)
        # Load the ranking
        committee = load_consensus_ranking(file_path)
        if args.committee_size < len(committee):
            committee = committee[:args.committee_size]
        if not committee:
            continue
            
        # Calculate Satisfaction Over Axioms
        satisfaction['JR'] = JR_check_satisfaction_given_committee(committee, user_to_set=user_to_set, movie_to_users=movie_to_users, n_users=n_users)
        satisfaction['PJR'] = PJR_check_satisfaction_given_committee(committee, partial_lists=preferences, l_cohesive=l_cohesive)
        satisfaction['EJR'] = EJR_check_satisfaction_given_committee(committee, preferences)
        print(f"Satisfaction: {satisfaction}")
        results.append((method_name, satisfaction))

    # 5. Print Table
    print("\n" + "=" * 60)
    print(f"{'Method':<20} | {'JR':^5} | {'PJR':^5} | {'EJR':^5}")
    print("-" * 60)

    for method, jr, pjr, ejr in results:
        def mark(x): return "✓" if x else "✗"
        print(f"{method:<20} | {mark(jr):^5} | {mark(pjr):^5} | {mark(ejr):^5}")

    print("=" * 60)
    
if __name__ == "__main__":
    main()