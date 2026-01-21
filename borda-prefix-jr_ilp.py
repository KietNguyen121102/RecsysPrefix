import cvxpy as cp
import numpy as np

def ilp(borda_ranking, l_cohesive):
    # here i'm imagining l_cohesive is built iteratively based on level (k)
    n = len(borda_ranking) # assuming we have rankings of equal size (prefix-JR and Borda)
    borda_dict = {}
    for i in range(len(borda_ranking)): # makes things easier later
        borda_dict[i] = borda_ranking[i]
    
    x = cp.Variable((n,n), boolean=True) # x_{c,p} = 1 if candidate c is in position p, 0 otw
    y = cp.Variable((n,n), boolean=True) # y_{c,d} = 1 if candidate c is ranked above d in ranking, 0 otw

    constraints = []

    # variable constraints
    for c in range(n):
        constraints.append(cp.sum(x[c, :]) == 1) # 1 position per candidate
    for p in range(n):
        constraints.append(cp.sum(x[:, p]) == 1) # 1 candidate per position

    for c in range(n):
        for d in range(n):
            if c != d:
                for p in range(n):
                    for q in range(p + 1, n):
                        constraints.append(y[c,d] >= x[c,p] + x[d, q] - 1) # enforces y_{c,d} to be correct

    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c,d] + y[d,c] == 1)
    
    for c in range(n):
        constraints.append(y[c,c] == 0)
    
    # prefix-JR constraints:
    for k in range(1, n + 1):
        cohesive_groups = l_cohesive['voter_sets']
        alts = l_cohesive['candidate_sets']
        cohesive_k = cohesive_groups[k]
        alts_k = alts[k]
        constraints.append()


# usage:
# get all cohesive groups 
# run ilp, include it in the evaluation pipeline 
# -> sanity check: must satisfy JR always. otherwise something is very wrong