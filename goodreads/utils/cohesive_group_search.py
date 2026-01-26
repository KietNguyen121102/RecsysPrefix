import sys
from collections import defaultdict, deque
import argparse
import time
import numpy as np
import pandas as pd
from itertools import permutations, combinations
from tqdm import tqdm 
import math 


def read_edgelist_from_csv(path):
    df = pd.read_csv(path)
    U = set()
    V = set()
    edges = set()
    voters = df['User_ID'].apply(lambda x: f"V_{x}")
    candidates = df['Ranked_Items'].apply(lambda x: f"C_{x}")
    for i in range(len(voters)):
        u = voters.iloc[i]
        v = candidates.iloc[i]
        U.add(u)
        V.add(v)
        edges.add((u, v))
    return U, V, edges


def read_edgelist_from_df(df):
    U = set()
    V = set()
    edges = set()
    voters = df['User_ID'].apply(lambda x: f"V_{x}")
    candidates = df['Ranked_Items'].apply(lambda x: f"C_{x}")
    for i in range(len(voters)):
        u = voters.iloc[i]
        v = candidates.iloc[i]
        U.add(u)
        V.add(v)
        edges.add((u, v))
    return U, V, edges


def read_aggregate_list(path):
    df = pd.read_csv(path)
    candidates = df.iloc[:, 2].apply(lambda x: f"C_{x}")
    return list(candidates)


def build_clique_extended_graph(U, V, edges):
    """
    Build adjacency dictionary for GC: nodes = U union V.
    Edges = original edges + all-pairs edges inside U + all-pairs edges inside V.
    Returns: dict node -> set(neighbors)
    """
    GC = dict()
    for u in U:
        neighbors = set(U)
        neighbors.remove(u)
        for (_, v) in filter(lambda e: e[0] == u, edges):
            neighbors.add(v)
        GC[u] = neighbors

    neighU = defaultdict(set)
    neighV = defaultdict(set)
    for u, v in edges:
        neighU[u].add(v)
        neighV[v].add(u)

    for u in U:
        GC[u] = (set(U) - {u}) | neighU[u]
    for v in V:
        GC[v] = (set(V) - {v}) | neighV[v]
    return GC


def degeneracy_ordering(adj):
    """
    Return degeneracy ordering of nodes (smallest-first removal).
    adj: dict node->set(neighbors)
    Returns: list order (nodes), and core number mapping.
    """
    n = len(adj)
    deg = {u: len(adj[u]) for u in adj}
    maxdeg = max(deg.values()) if deg else 0
    bins = [deque() for _ in range(maxdeg+1)]
    for u, d in deg.items():
        bins[d].append(u)
    order = []
    core = dict()
    curr_deg = 0
    removed = set()
    for k in range(n):
        i = 0
        while i <= maxdeg and not bins[i]:
            i += 1
        if i > maxdeg:
            break
        u = bins[i].popleft()
        order.append(u)
        core[u] = i
        removed.add(u)
        for w in list(adj[u]):
            if w in removed:
                continue
            d_old = deg[w]
            deg[w] -= 1
            bins[d_old].remove(w)
            bins[d_old-1].append(w)
    return order[::-1], core


def bron_kerbosch_pivot(adj, R, P, X, output_callback):
    """
    adj: dict node->set(neighbors)
    R, P, X: sets
    output_callback(R) called when R is maximal clique
    """
    if not P and not X:
        output_callback(R)
        return
    Px = P | X
    max_int = -1
    pivot = None
    for u in Px:
        inter = len(P & adj[u])
        if inter > max_int:
            max_int = inter
            pivot = u
    for v in list(P - adj.get(pivot, set())):
        Nv = adj[v]
        bron_kerbosch_pivot(adj, R | {v}, P & Nv, X & Nv, output_callback)
        P.remove(v)
        X.add(v)


def maximal_bicliques_from_gc(GC_adj, Uset, Vset):
    """
    Enumerate maximal cliques of GC (which map to maximal bicliques of original G).
    For each maximal clique C, output C_U and C_V (split by membership).
    """
    final_voters, final_candidates = [], []
    order, core = degeneracy_ordering(GC_adj)
    nodes_pos = {node: i for i, node in enumerate(order)}
    
    for v in order:
        N_v = GC_adj[v]
        P = {w for w in N_v if nodes_pos[w] > nodes_pos[v]}
        X = {w for w in N_v if nodes_pos[w] < nodes_pos[v]}
        R = {v}

        def cb(Rclique):
            CU = set([int(x.strip('V_')) for x in Rclique if x in Uset])
            CV = set([int(x.strip('C_')) for x in Rclique if x in Vset])
            if CU and CV:
                final_voters.append(CU)
                final_candidates.append(CV)

        bron_kerbosch_pivot(GC_adj, R, P, X, cb)
    return final_voters, final_candidates


def add_subsets(voter_sets, candidate_sets, k, n):
    cohesive_voter_blocks = []
    cohesive_candidate_blocks = []
    l_cohesive = {} 
    for l in range(1, k+1):
        final_candidate_sets = []
        final_voter_sets = []
        for i in range(len(candidate_sets)):
            if len(candidate_sets[i]) >= l and len(voter_sets[i]) >= (l*n)/k:
                cohesive_voter_blocks.append(voter_sets[i])
                cohesive_candidate_blocks.append(candidate_sets[i])
                
                voter_subsets = list(
                    combinations(voter_sets[i], r=int(math.ceil((l*n)/k))))
                
                final_voter_sets.extend(voter_subsets)
                final_candidate_sets.extend([candidate_sets[i]] * len(voter_subsets))
        final_voter_sets = [list(s) for s in final_voter_sets]
        final_candidate_sets = [list(s) for s in final_candidate_sets]
        l_cohesive[l] = {'voter_sets': final_voter_sets,
                         'candidate_sets': final_candidate_sets}
    return l_cohesive


def find_maximal_cohesive_groups(partial_lists, committee_size):
    U, V, edges = read_edgelist_from_df(partial_lists)
    GC = build_clique_extended_graph(U, V, edges)
    voter_sets, candidate_sets = maximal_bicliques_from_gc(GC, U, V)
    return voter_sets, candidate_sets


def find_all_cohesive_groups(partial_lists, committee_size, number_voters):
    U, V, edges = read_edgelist_from_df(partial_lists)
    GC = build_clique_extended_graph(U, V, edges)
    voter_sets, candidate_sets = maximal_bicliques_from_gc(GC, U, V) 
    l_cohesive = add_subsets(voter_sets, candidate_sets,
                             committee_size, number_voters)
    return voter_sets, candidate_sets, l_cohesive

