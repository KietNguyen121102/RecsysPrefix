import cvxpy as cp
import numpy as np
import random
import math
import time
from collections import deque


# TOPK = 4  # This will be set dynamically
TOPK = 4

def Kendall_Tau_Dist(first, second):
    mappedrank = []
    for i in range(len(second)):
        mappedrank.append(first.index(second[i]))
    cost, blank = mergesort(mappedrank)
    return cost

# mergesort to compute distance in nlogn time
# input: A single ranking
# output: Kendall tau distance to the ranking 1, 2, ..., n
def mergesort(ranking):
    if len(ranking) <= 1:
        return 0, ranking
    leftsum, leftrank = mergesort(ranking[:len(ranking)//2])
    rightsum, rightrank = mergesort(ranking[len(ranking)//2:])
    csum = leftsum + rightsum
    leftindex = 0
    rightindex = 0
    outrank = []
    while leftindex < len(leftrank) and rightindex < len(rightrank):
        if leftrank[leftindex] < rightrank[rightindex]:
            outrank.append(leftrank[leftindex])
            leftindex += 1
        else:
            outrank.append(rightrank[rightindex])
            csum += len(leftrank) - leftindex
            rightindex += 1
    if leftindex < len(leftrank):
        outrank += leftrank[leftindex:]
    if rightindex < len(rightrank):
        outrank += rightrank[rightindex:]
    return csum, outrank
    
def Get_Objective_Value(query, rankings):
    median_cost = 0
    for rank in rankings:
        median_cost += Kendall_Tau_Dist(query, rank)
    return median_cost

# helper function to return the weighted tournament corresponding to the rank aggregation problem
def Get_Frac_Tournament(rankings):
    element_count = len(rankings[0])
    frac_tournament = np.ndarray((element_count, element_count))
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = 0

    for ranking in rankings:
        for i in range(len(ranking)):
            for j in range(i+1, len(ranking)):
                frac_tournament[ranking[i]][ranking[j]] += 1
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = frac_tournament[i][j] / len(rankings)

    return frac_tournament
    
# helper function to recover ordering from acyclic tournament
def Topological_Sort(adj):
    n = len(adj)  
    in_degree = [0] * n

    for i in range(n):
        for j in range(n):
            if adj[i][j] > 0.5:
                in_degree[j] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_sort = []

    while queue:
        node = queue.popleft()
        topo_sort.append(node)

        for j in range(n):
            if adj[node][j] > 0.5:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    return topo_sort

def process_for_fair_ranking(candidates, group_df, ranks_for_fairness):
    num_attributes = len(group_df['binned'].unique())
    
    attributes_df = group_df[group_df.item.isin(candidates)][['item', 'binned']]
    attributes_df = attributes_df.sort_values(by='item')
    item_to_idx = dict(zip(attributes_df.item, list(range(len(candidates)))))
    idx_to_item = dict(zip(list(range(len(candidates))), attributes_df.item))
    item_to_attribute = dict(zip(list(range(len(candidates))), attributes_df.binned))
    for rank_idx in range(len(ranks_for_fairness)):
        for pos in range(len(ranks_for_fairness[rank_idx])):
            ranks_for_fairness[rank_idx][pos] = item_to_idx[ranks_for_fairness[rank_idx][pos]]
    alphas = [1.0 / num_attributes] * num_attributes
    betas = [1.0] * num_attributes

    return alphas, betas, ranks_for_fairness, item_to_attribute, idx_to_item, num_attributes 

    
# Input: set of rankings
# Returns: the optimal median ranking by using an ILP
# See "Improved Bounds for Computing Kemeny Rankings" for related information.
def NormalILP(rankings):
    element_count = len(rankings[0])
    
    frac_tournament = Get_Frac_Tournament(rankings)
    
    X = cp.Variable(element_count * element_count, boolean = True)
    constraints = []
    
    # constraints that for every pair, one is before the other
    for i in range(element_count):
        for j in range(element_count):
            coeff = np.zeros(element_count * element_count)
            coeff[i*element_count + j] = 1
            coeff[j*element_count + i] = 1
            constraints += [coeff @ X == 1]
    
    # triangle inequality constraint
    # x_ab + x_bc + x_ca >= 1 for any a, b, c
    for i in range(element_count):
        for j in range(element_count):
            if i == j:
                continue
            for k in range(element_count):
                if i == k or j == k:
                    continue
                coeff = np.zeros(element_count * element_count)
                coeff[i*element_count + j] = 1
                coeff[j*element_count + k] = 1
                coeff[k*element_count + i] = 1
                constraints += [coeff @ X >= 1]
    
    edge_weight_coeff = np.empty(element_count * element_count)
    for i in range(element_count):
        for j in range(element_count):
            edge_weight_coeff[i * element_count + j] = frac_tournament[i][j]
            
    problem = cp.Problem(cp.Minimize(edge_weight_coeff @ X), constraints)
    
    problem.solve(solver = cp.SCIP)

    result = X.value.reshape(element_count, -1)
    for i in range(element_count):
        result[i][i] = 0
    result_tp = [[0]*element_count for i in range(element_count)]
    for i in range(element_count):
        for j in range(element_count):
            if i != j:
                result_tp[i][j] = result[j][i]
    
    topo_sorted = Topological_Sort(result_tp)
    return topo_sorted

# FAIR ILP
# Takes in the fairness parameters, rankings, mapping of elements to attributes
# Returns the optimal fair median ranking
def FairILP(alphas, betas, rankings, id_attribute, num_attributes):

    start_time = time.time()
    element_count = len(rankings[0])

    frac_tournament = Get_Frac_Tournament(rankings)

    # Large constant, must be bigger than 2 * elements
    bigM = element_count * 10

    X = cp.Variable(element_count * element_count, boolean = True)
    constraints = []
    
    # constraints that for every pair, one is before the other
    for i in range(element_count):
        for j in range(element_count):
            coeff = np.zeros(element_count * element_count)
            coeff[i*element_count + j] = 1
            coeff[j*element_count + i] = 1
            constraints += [coeff @ X == 1]
    
    # triangle inequality constraint
    # x_ab + x_bc + x_ca >= 1 for any a, b, c
    for i in range(element_count):
        for j in range(element_count):
            if i == j:
                continue
            for k in range(element_count):
                if i == k or j == k:
                    continue
                coeff = np.zeros(element_count * element_count)
                coeff[i*element_count + j] = 1
                coeff[j*element_count + k] = 1
                coeff[k*element_count + i] = 1
                constraints += [coeff @ X >= 1]

    # Y_a variables to enforce fairness
    # The pair of constraints force it so that Y = 1 if for some i, at least d - 1 - K X_ij are 0
    # Otherwise it is forced to be 0
    Y = cp.Variable(element_count, boolean = True)

    # to be in top-K, the element must be ordered ahead of at least d - K elements.
    largerthan_k = element_count - TOPK - 1
    for i in range(element_count):
        coeff = np.zeros(element_count * element_count)
        for j in range(element_count):
            if i != j:
                coeff[i * element_count + j] = -1
        constraints += [coeff @ X + element_count - 1 >= largerthan_k + 1 - bigM*(1 - Y[i])]
        constraints += [coeff @ X + element_count - 1 <= largerthan_k + bigM * Y[i]]

    # Lower and upper bound constraints per attribute
    for attribute in range(num_attributes):
        coeff = np.zeros(element_count)
        for i in range(element_count):
            if id_attribute[i] == attribute:
                coeff[i] = 1
        lb = math.floor(alphas[attribute] * TOPK)
        ub = math.ceil(betas[attribute] * TOPK)
        constraints += [coeff @ Y >= lb]
        constraints += [coeff @ Y <= ub]
    
    edge_weight_coeff = np.empty(element_count * element_count)
    for i in range(element_count):
        for j in range(element_count):
            edge_weight_coeff[i * element_count + j] = frac_tournament[i][j]
            
    problem = cp.Problem(cp.Minimize(edge_weight_coeff.T @ X), constraints)
    
    problem.solve(solver = cp.SCIP)

    # use topological sorting algo to also get the ordering of elements
    result = X.value.reshape(element_count, -1)
    for i in range(element_count):
        result[i][i] = 0
    result_tp = [[0]*element_count for i in range(element_count)]
    for i in range(element_count):
        for j in range(element_count):
            if i != j:
                result_tp[i][j] = result[j][i]
    
    
    topo_sorted = Topological_Sort(result_tp)
    obj_cost = Get_Objective_Value(topo_sorted, rankings)
    end_time = time.time()
    return topo_sorted

# This implementation of our algorithm uses ILP to solve the two partitions optimally
# Think of this as the 'best case' scenario possible.
# Takes in the fairness parameters, rankings, mapping of elements to attributes
def Consensus(alphas, betas, rankings, id_attribute, num_attributes):
    element_count = len(rankings[0])

    # STEP 1: determining top-k elements
    # Construct weighted tournament, and then sort by indegrees, and take it as following the algorithm in the paper
   
    start_time = time.time()
    frac_tournament = Get_Frac_Tournament(rankings)

    fract_time = time.time()

    # List of lists.
    # List i contains tuples of elements with attribute i
    # tuple is in the form (element id, indegree)
    indegree_attr = []
    
    for attribute in range(num_attributes):
        indegree_attr.append([])
    for i in range(element_count):
        i_attr = id_attribute[i]
        indeg = 0
        for j in range(element_count):
            indeg += frac_tournament[j][i]
        indegree_attr[i_attr].append((i, indeg))
    for attr in range(num_attributes):
        indegree_attr[attr].sort(key = lambda ituple : ituple[1])

    topk_elements = set()
    elements_taken = [0] * num_attributes
    num_taken = 0
    
    # now, we get top k elements following the algo
    # take lower bound first
    # form combined list at same time
    indegree_combined = []
    for attr in range(num_attributes):
        for j in range(math.floor(alphas[attr] * TOPK)):
            topk_elements.add(indegree_attr[attr][j][0])
            elements_taken[attr] += 1
        indegree_combined += indegree_attr[attr][math.floor(alphas[attr] * TOPK):]
    
    # sort combined list, then take while respecting beta upper bounds
    indegree_combined.sort(key = lambda ituple : ituple[1])
    for i in range(len(indegree_combined)):
        if len(topk_elements) >= TOPK:
            break
        element = indegree_combined[i]
        i_attr = id_attribute[element[0]]
        if elements_taken[i_attr] < math.ceil(betas[i_attr] * TOPK):
            elements_taken[i_attr] += 1
            topk_elements.add(element[0])


    # STEP 2, we need to order the top-k.
    # Following the paper, we need to run rank aggregation over the two partitions.
    
    # In this implementation, ILP is used to solve optimally, giving the best case scenario.
    
    # So we need to construct the restricted rankings
    # left is top k, right is the remaining elements
    
    rankings_left = []
    rankings_right = []
    for rank in rankings:
        left_rank = []
        right_rank = []
        for i in rank:
            if i in topk_elements:
                left_rank.append(i)
            else:
                right_rank.append(i)
        rankings_left.append(left_rank)
        rankings_right.append(right_rank)

    # NOTE: Because the elements of the reduced rankings are not a continuous 1 ... k, we need to relabel the elements to be 1 ... k, and save the mapping
    # so we can map the result back to these elements

    left_forward_map = {}
    left_backward_map = {}
    mapped_rankings_left = []
    for i in range(len(rankings_left[0])):
        left_forward_map[rankings_left[0][i]] = i
        left_backward_map[i] = rankings_left[0][i]
    mapped_rankings_left.append([i for i in range(len(rankings_left[0]))])
    for i in range(1, len(rankings_left)):
        mapped_rank = []
        for j in rankings_left[i]:
            mapped_rank.append(left_forward_map[j])
        mapped_rankings_left.append(mapped_rank)

    right_forward_map = {}
    right_backward_map = {}
    mapped_rankings_right = []
    for i in range(len(rankings_right[0])):
        right_forward_map[rankings_right[0][i]] = i
        right_backward_map[i] = rankings_right[0][i]
    mapped_rankings_right.append([i for i in range(len(rankings_right[0]))])
    for i in range(1, len(rankings_right)):
        mapped_rank = []
        for j in rankings_right[i]:
            mapped_rank.append(right_forward_map[j])
        mapped_rankings_right.append(mapped_rank)
    
    # use ILP to solve
    left_topo_sorted = NormalILP(mapped_rankings_left)
    right_topo_sorted = NormalILP(mapped_rankings_right)
    
    # Re-map the topologically sorted elements, to the original elements using the backward maps

    left_original = [left_backward_map[i] for i in left_topo_sorted]
    right_original = [right_backward_map[i] for i in right_topo_sorted]

    output_ranking = left_original + right_original

    # Get objective cost
    obj_cost = Get_Objective_Value(output_ranking, rankings)

    end_time = time.time()
    return output_ranking

