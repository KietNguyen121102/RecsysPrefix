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
import pickle 
from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
from utils.axiom_checks import JR_check_satisfaction_given_committee, PJR_check_satisfaction_given_committee, EJR_check_satisfaction_given_committee
from utils.io import load_consensus_ranking, load_sampled_preferences
# from cohesive_group_search2 import maximal_bicliques_implicit_gc, add_subsets_fast

# =============================================================================
# 1. Helper Functions
# =============================================================================
def preprocess_for_JR(approvals):
    # groupby once instead of filtering many times
    user_to_set = approvals.groupby("User_ID")["Ranked_Items"].apply(set).to_dict()
    movie_to_users = approvals.groupby("Ranked_Items")["User_ID"].apply(list).to_dict()
    n_users = len(user_to_set)
    all_candidates = list(movie_to_users.keys())
    return user_to_set, movie_to_users, n_users, all_candidates
# =============================================================================
# 3. Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', '-a', default='consensus_results', help='Directory containing consensus TXT files')
    parser.add_argument('--pref', '-p', default='recommendations.csv', help='Path to user recommendations CSV')
    # parser.add_argument('--committee_size', '-k', type=int, default=50, help='Size of the committee to select')
    args = parser.parse_args()

    print("=" * 70)
    print(f"(1) Loading data")
    print("-" * 70)
    
    # 1. Load User-Level Fully Ordered Preference Lists Data
    preferences = load_sampled_preferences(args.pref)
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    
    print(f"(2) Loading Consensus Files")
    print("-" * 70)
    # 3. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    
    
    print("-" * 70)
    print(f'RUN STATS for: {args.agg}')
    print("-" * 70)
    print("Number of Voters:", number_voters)
    print("Number of Candidates:", number_candidates)
    print("Number of methods to evaluate:", len(consensus_files))
    print("-" * 70)
    
    print(f"(4) Calculating Axiom Satisfaction")
    results = []
    
    # 4. Iterate over each method
    for idx, file_path in enumerate(consensus_files):
        satisfaction = {'JR':[], 'PJR':[], 'EJR':[]}
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"({idx}/{len(consensus_files)}) Method:", method_name)
        
        # if os.path.exists('/data2/rsalgani/Prefix/ml-1m/agg_files/sample_9/axiom_satisfaction_results.pkl'): 
        #     print("Skipping existing results...")
        #     continue
        
        # Load the ranking
        committee = load_consensus_ranking(file_path)
        if not committee:
            continue
        
        for prefix_idx in tqdm(range(len(committee))):
            preferences_at_prefix = (
                    preferences
                    .assign(Ranked_Items=lambda df:
                            df['Ranked_Items'].apply(lambda x: x[:prefix_idx + 1]))
                    .explode('Ranked_Items')
                    .reset_index(drop=True)
                )
            # ipdb.set_trace()
            # Calculate Satisfaction Over Axioms
            # user_to_set, movie_to_users, n_users, all_candidates = preprocess_for_JR(preferences)
            voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(preferences_at_prefix, committee_size=prefix_idx+1, number_voters=number_voters)
            satisfaction['JR'].append(JR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, all_candidates=all_candidates, n=number_voters, k=prefix_idx+1))
            satisfaction['PJR'].append(PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive))
            satisfaction['EJR'].append(EJR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix)) 
            
        # print(f"Satisfaction: {satisfaction}")
        results.append((method_name, satisfaction))
    # 5. Print Table
    print("\n" + "=" * 60)
    print(f"{'Method':<20} | {'JR':^5} | {'PJR':^5} | {'EJR':^5}")
    print("-" * 60)

    for method, satisfaction in results:
        jr, pjr, ejr = satisfaction['JR'], satisfaction['PJR'], satisfaction['EJR']
        def mark(x): return "✓" if all(x) else "✗"
        print(f"{method:<20} | {mark(jr):^5} | {mark(pjr):^5} | {mark(ejr):^5}")

    print("=" * 60)
    results_df =  pd.DataFrame({
        method: {k: all(v) for k, v in metrics.items()}
            for method, metrics in results
        }).T
    # ipdb.set_trace() 
    pickle.dump(results_df, open(f'{args.agg}/axiom_satisfaction_results.pkl', 'wb')) 

def test(): 
    pref_path = '/u/rsalgani/2024-2025/RecsysPrefix/consensus_results/sample_0/sampled_rankings.pkl'
    agg_path = '/u/rsalgani/2024-2025/RecsysPrefix/consensus_results/sample_0/BordaCount.txt'
    preferences = load_sampled_preferences(pref_path)
    committee = load_consensus_ranking(agg_path)
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    satisfaction = {}
    
    vidx_to_int = {vidx: i for i, vidx in enumerate(preferences['User_ID'].unique())}
    cidx_to_int = {cidx: i for i, cidx in enumerate(all_candidates)}
    
    preferences['User_ID'] = preferences['User_ID'].map(vidx_to_int)
    preferences['Ranked_Items'] = preferences['Ranked_Items'].apply(lambda x: [cidx_to_int[c] for c in x])
    committee = [cidx_to_int[c] for c in committee if c in cidx_to_int]
    
    for prefix_idx in tqdm(range(len(committee))):
        preferences_at_prefix = (
                preferences
                .assign(Ranked_Items=lambda df:
                        df['Ranked_Items'].apply(lambda x: x[:prefix_idx + 1]))
                .explode('Ranked_Items')
                .reset_index(drop=True)
            )
        # ipdb.set_trace()
        # Calculate Satisfaction Over Axioms
        # user_to_set, movie_to_users, n_users, all_candidates = preprocess_for_JR(preferences)
        voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(preferences_at_prefix, committee_size=prefix_idx+1, number_voters=number_voters)
        satisfaction['JR'] = PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive)
        satisfaction['EJR'] = EJR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix) 
        satisfaction['PJR'] = PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive)
        
        print(prefix_idx, satisfaction)

def test2(): 
    preferences = pd.DataFrame({
        'User_ID': [0, 1, 2, 3, 4, 5],
        'Ranked_Items': [[2,0,1,5,3,4], [2,0,1,5,3,4], 
                         [3,0,1,5,2,4], [3,0,1,5,2,4], 
                         [4,0,1,5,3,2], [4,0,1,5,3,2]]
    })
    # preferences = pd.DataFrame({
    #     'User_ID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     'Ranked_Items': [
    #         [5,7,15,3,4],
    #         [5,8,16,3,4],
    #         [5,9,17,3,4],
    #         [5,10,18,3,4],
    #         [5,6,19,3,4],
    #         [5,25,20,1,2],
    #         [5,11,21,1,2],
    #         [5,12,22,1,2],
    #         [5,13,23,1,2],
    #         [5,14,24,1,2],
    #                      ]
    # })
    item_attribute = {0:0, 1:1, 2:1, 3:1, 4:1, 5:2}
    committee = [0,2,3,5,4,1] # SHOULD PASS
    committee = [2,1,5,0,3,4] # SHOULD FAIL 
    # committee = [5, 6, 25, 19, 20] # SHOULD FAIL EJR
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    satisfaction = {}
    
    vidx_to_int = {vidx: i for i, vidx in enumerate(preferences['User_ID'].unique())}
    cidx_to_int = {cidx: i for i, cidx in enumerate(all_candidates)}
    
    # preferences['User_ID'] = preferences['User_ID'].map(vidx_to_int)
    # preferences['Ranked_Items'] = preferences['Ranked_Items'].apply(lambda x: [cidx_to_int[c] for c in x])
    # committee = [cidx_to_int[c] for c in committee if c in cidx_to_int]
    
    for prefix_idx in tqdm(range(len(committee))):
        # print(preferences_at_prefix)
        preferences_at_prefix = (
                preferences
                .assign(Ranked_Items=lambda df:
                        df['Ranked_Items'].apply(lambda x: x[:prefix_idx + 1]))
                .explode('Ranked_Items')
                .reset_index(drop=True)
            )
        # ipdb.set_trace()
        # Calculate Satisfaction Over Axioms
        # user_to_set, movie_to_users, n_users, all_candidates = preprocess_for_JR(preferences)
        voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(preferences_at_prefix, committee_size=prefix_idx+1, number_voters=number_voters)
        # satisfaction['JR'] = JR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix) 
        # satisfaction['PJR'] = PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix) 
        # satisfaction['EJR'] = EJR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix) 
        satisfaction['JR'] = JR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, all_candidates=all_candidates, n=number_voters, k=prefix_idx+1)
        satisfaction['PJR'] = PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive)
        satisfaction['EJR'] = EJR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix)
            
        # satisfaction['PJR'] = PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive)
        print(satisfaction)
def test3(): 
    
    preferences = pd.DataFrame({
        'User_ID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Ranked_Items': [
            [5,7,15,3,4],
            [5,8,16,3,4],
            [5,9,17,3,4],
            [5,10,18,3,4],
            [5,6,19,3,4],
            [5,25,20,1,2],
            [5,11,21,1,2],
            [5,12,22,1,2],
            [5,13,23,1,2],
            [5,14,24,1,2],
                         ]
    })
    
    committee = [5, 6, 25, 19, 20] # SHOULD FAIL EJR
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    satisfaction = {}
    
    for prefix_idx in tqdm(range(len(committee))):
        # print(preferences_at_prefix)
        preferences_at_prefix = (
                preferences
                .assign(Ranked_Items=lambda df:
                        df['Ranked_Items'].apply(lambda x: x[:prefix_idx + 1]))
                .explode('Ranked_Items')
                .reset_index(drop=True)
            )
        # ipdb.set_trace()
        # Calculate Satisfaction Over Axioms
        # user_to_set, movie_to_users, n_users, all_candidates = preprocess_for_JR(preferences)
        voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(preferences_at_prefix, committee_size=prefix_idx+1, number_voters=number_voters)
        satisfaction['JR'] = JR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, all_candidates=all_candidates, n=number_voters, k=prefix_idx+1)
        satisfaction['PJR'] = PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive)
        satisfaction['EJR'] = EJR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix)
            
        # satisfaction['PJR'] = PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive)
        print(satisfaction)
        
if __name__ == "__main__":
    test2()
    test3()  
    # test() 
    # main()