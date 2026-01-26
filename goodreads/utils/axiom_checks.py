import os
import csv
import glob
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import argparse
from tqdm import tqdm 
import pandas as pd 
import math 
from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups

# =============================================================================
# Axiom Checks
# =============================================================================

def JR_check_satisfaction_given_committee(proposed_committee, partial_lists, all_candidates, n, k): 
    """
    Check if Justified Representation (JR) is satisfied.
    
    JR is violated if there exists a candidate c and a group of voters V' such that:
    - |V'| >= n/k
    - All voters in V' approve c
    - No voter in V' has any approved candidate in the committee
    """
    for candidate in all_candidates:
        counter = 0 
        approving_voters = partial_lists[partial_lists['Ranked_Items']
                                         == candidate]['User_ID'].unique().tolist()
        for voter in approving_voters: 
            approval_set = partial_lists[partial_lists['User_ID'] == voter]['Ranked_Items'].unique().tolist()
            if len(np.intersect1d(approval_set, proposed_committee)) == 0: 
                counter += 1 
        if counter >= n/k: 
            return False             
    return True


def PJR_check_satisfaction_given_committee(proposed_committee, partial_lists, l_cohesive):
    """
    Check if Proportional Justified Representation (PJR) is satisfied.
    
    PJR requires that for any l-cohesive group (a group of size >= l*n/k 
    that jointly approves at least l candidates), the committee must contain
    at least l candidates from their joint approval set.
    """
    n, k = len(partial_lists['User_ID'].unique()), len(proposed_committee)
    for l in range(1, k+1):
        voter_sets = l_cohesive[l]['voter_sets']
        candidate_sets = l_cohesive[l]['candidate_sets']
        for i in range(len(candidate_sets)):
            voter_group, candidate_group = voter_sets[i], candidate_sets[i]
            if len(voter_group) >= l*(n/k) and len(candidate_group) >= l:
                approval_set = partial_lists[partial_lists['User_ID'].isin(voter_group)]['Ranked_Items'].unique().tolist()
                if len(np.intersect1d(approval_set, proposed_committee)) < l:
                    return False       
    return True


def prune_satisfied_for_EJR(partial_lists, proposed_committee, l):
    """
    Remove voters who are already l-satisfied (have at least l approved 
    candidates in the committee).
    """
    for voter in partial_lists['User_ID'].unique().tolist():
        approval_set = partial_lists[partial_lists['User_ID'] == voter]['Ranked_Items'].unique().tolist()
        if len(np.intersect1d(approval_set, proposed_committee)) >= l:
            partial_lists = partial_lists[partial_lists['User_ID'] != voter]
    return partial_lists.reset_index(drop=True)


def EJR_check_satisfaction_given_committee(proposed_committee, partial_lists): 
    """
    Check if Extended Justified Representation (EJR) is satisfied.
    
    EJR requires that for any l-cohesive group, at least one voter in the group
    must be l-satisfied (have at least l approved candidates in the committee).
    """
    n, k = len(partial_lists['User_ID'].unique()), len(proposed_committee)
    for l in range(1, k+1):
        unsatisfied_voter_set = prune_satisfied_for_EJR(partial_lists, proposed_committee, l)
        voter_sets, candidate_sets = find_maximal_cohesive_groups(unsatisfied_voter_set, committee_size=k)
        for idx, v in enumerate(voter_sets): 
            if len(v) >= (l*n)/k and len(candidate_sets[idx]) >= l: 
                return False
    return True

