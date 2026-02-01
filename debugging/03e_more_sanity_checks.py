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
import yaml 
import pickle 
from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
from utils.axiom_checks import JR_check_satisfaction_given_committee, PJR_check_satisfaction_given_committee, EJR_check_satisfaction_given_committee
from utils.io import load_consensus_ranking, load_sampled_preferences
from utils.borda_prefix_jr_ilp import *

from utils.io import load_sampled_preferences, load_consensus_ranking, load_rankings_to_df
# =============================================================================
# I/O Utils
# =============================================================================

def test_load_sampled_preferences(): 
    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    print(dataset_cfg)
    preferences = load_sampled_preferences("/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl")
    print(preferences.head())

def test_load_rankings_to_df(): 
    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    print(dataset_cfg)
    rankings_df, all_items = load_rankings_to_df(dataset_cfg)
    print(rankings_df.head())
    print("All items:", all_items)


# =============================================================================
# Axiom Satisfaction Checking
# =============================================================================


def check_satisfaction_one_ranking(committee_path, preferences_path): 
    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    print(dataset_cfg)
    
    print("=" * 70)
    print(f"(1) Loading data")
    print("-" * 70)
    
    # 1. Load User-Level Fully Ordered Preference Lists Data
    preferences = load_sampled_preferences(preferences_path)
    committee = load_consensus_ranking(committee_path)
    
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    method_name = os.path.splitext(os.path.basename(committee_path))[0]
    satisfaction = {'JR': [], 'PJR': [], 'EJR': []}
    results = []
    for prefix_idx in tqdm(range(len(committee))):
        preferences_at_prefix = (
                preferences
                .assign(Ranked_Items=lambda df:
                        df['Ranked_Items'].apply(lambda x: x[:prefix_idx + 1]))
                .explode('Ranked_Items')
                .reset_index(drop=True)
            )
        voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(preferences_at_prefix, committee_size=prefix_idx+1, number_voters=number_voters, data_cfg=dataset_cfg)
        satisfaction['JR'].append(JR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, all_candidates=all_candidates, n=number_voters, k=prefix_idx+1, user_key=dataset_cfg['dataset']['keys']['user_key']))
        satisfaction['PJR'].append(PJR_check_satisfaction_given_committee(committee[:prefix_idx+1], partial_lists=preferences_at_prefix, l_cohesive=l_cohesive, user_key=dataset_cfg['dataset']['keys']['user_key']))
        satisfaction['EJR'].append(EJR_check_satisfaction_given_committee(committee[:prefix_idx+1], preferences_at_prefix, data_cfg=dataset_cfg, user_key=dataset_cfg['dataset']['keys']['user_key']))
    
    results.append((method_name, satisfaction))
    
    for method, satisfaction in results:
        jr, pjr, ejr = satisfaction['JR'], satisfaction['PJR'], satisfaction['EJR']
        def mark(x): return "✓" if all(x) else "✗"
        print(f"{method:<20} | {mark(jr):^5} | {mark(pjr):^5} | {mark(ejr):^5}")

    print("=" * 60)


def check_ilp_and_borda(preferences_path, agg_path): 
    preferences = load_sampled_preferences(preferences_path) 
    borda_ranking = load_consensus_ranking(agg_path+"/BordaCount.txt")
    fair_ranking = load_consensus_ranking(agg_path+"/KuhlmanConsensus.txt")
    
    fair_info = pickle.load(open(agg_path+'fair_ranking_process.pkl', 'rb'))
    alphas, betas, attributes_map, idx_to_item, num_attributes = fair_info['alphas'], fair_info['betas'], fair_info['attributes_map'], fair_info['idx_to_item'], fair_info['num_attributes']
    item_to_idx = {item: idx for idx, item in idx_to_item.items()}
    user_to_idx = dict(zip(preferences['User_ID'].unique(), range(len(preferences['User_ID'].unique()))))
    borda_ranking_to_idx = [item_to_idx[item] for item in borda_ranking]
    n_voters = len(preferences['User_ID'].unique())
    approvals_by_k = {}
    ipdb.set_trace() 
    for prefix_idx in range(len(borda_ranking)):
        k = prefix_idx + 1
        preferences_at_prefix = (
            preferences
            .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
            .explode('Ranked_Items')
            .reset_index(drop=True)
        )
        preferences_at_prefix['Ranked_Items'] = preferences_at_prefix['Ranked_Items'].map(item_to_idx)
        preferences_at_prefix['User_ID'] = preferences_at_prefix['User_ID'].map(user_to_idx)
        
        approvals_k = {}
        for v in preferences_at_prefix["User_ID"].unique():
            approvals_k[v] = (
                preferences_at_prefix[preferences_at_prefix["User_ID"] == v]["Ranked_Items"]
                .unique().tolist()
            )
        approvals_by_k[k] = approvals_k
    ipdb.set_trace() 
    result, _ = ilp_prefix_jr(borda_ranking_to_idx, approvals_by_k, n_voters)
    result = [idx_to_item[i] for i in result]
    print("Borda Result:", borda_ranking)
    print("ILP Result:", result)



# check_satisfaction_one_ranking(
#     committee_path="/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/BordaCount.txt",
#     preferences_path="/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl"
# ) 

# check_satisfaction_one_ranking(
#     committee_path="/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/KuhlmanConsensus.txt",
#     preferences_path="/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl"
# ) 


# check_ilp_and_borda(preferences_path = "/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl", 
                    # agg_path= "/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/")
