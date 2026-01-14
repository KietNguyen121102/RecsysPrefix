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
from cohesive_group_search import find_maximal_cohesive_groups


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', '-a', default='consensus_results', help='Directory containing consensus TXT files')
    parser.add_argument('--pref', '-p', default='recommendations.csv', help='Path to user recommendations CSV')
    args = parser.parse_args()
    preferences = pd.read_csv(args.pref)
     
    voter_sets, candidate_sets, U, V = find_maximal_cohesive_groups(preferences,  '')
    ipdb.set_trace() 
main() 