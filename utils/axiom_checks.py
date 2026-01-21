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
from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups

# =============================================================================
# 1. Axiom Checks
# =============================================================================

def JR_check_satisfaction_given_committee_old(proposed_committee, user_to_set, movie_to_users, n_users):
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

def JR_check_satisfaction_given_committee(proposed_committee, partial_lists, all_candidates, n, k): 
    # ipdb.set_trace() 
    for candidate in all_candidates:
        counter = 0 
        approving_voters = partial_lists[partial_lists['Ranked_Items']
                                         == candidate]['User_ID'].unique().tolist()
        for voter in approving_voters: 
            approval_set = partial_lists[partial_lists['User_ID'] ==  voter]['Ranked_Items'].unique().tolist()
            if len(np.intersect1d(approval_set, proposed_committee)) == 0: counter += 1 
        if counter == n/k: 
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
                    # ipdb.set_trace() 
                    approval_set = partial_lists[partial_lists['User_ID'].isin(voter_group)]['Ranked_Items'].unique(
                    ).tolist() #find the union of candidates that they all agree on 
                    if len(np.intersect1d(approval_set, proposed_committee)) < l: #if less than l of them in the committee W
                        return False       
    return True


def prune_satisfied_for_EJR(partial_lists, proposed_committee, l):
    for voter in partial_lists['User_ID'].unique().tolist(): #look through each voter
        approval_set = partial_lists[partial_lists['User_ID'] == voter]['Ranked_Items'].unique(
        ).tolist() #find the candidates he approves of
        if len(np.intersect1d(approval_set, proposed_committee)) >= l: #if the voter is l-satisfied
            partial_lists = partial_lists[partial_lists['User_ID'] != voter] #prune that voter 
    return partial_lists.reset_index(drop=True)

def EJR_check_satisfaction_given_committee(proposed_committee, partial_lists): 
    # ipdb.set_trace() 
    n, k = len(partial_lists['User_ID'].unique()), len(proposed_committee)
    # for l in tqdm(range(1, k+1)): #iterate through l, increasing from 1
    for l in range(1, k+1):
        unsatisfied_voter_set = prune_satisfied_for_EJR(partial_lists, proposed_committee, l)
        voter_sets, candidate_sets = find_maximal_cohesive_groups(unsatisfied_voter_set, committee_size=k)
        # voter_sets, cand_sets = find_maximal_cohesive_groups_groupby(partial_lists)
        for idx, v in enumerate(voter_sets): 
            if len(v) >= l*(n/k) and len(candidate_sets[idx]) == l: 
                return False
    return True
