import cvxpy as cp
import numpy as np
import pandas as pd 
import ipdb 
from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
from utils.axiom_checks import JR_check_satisfaction_given_committee, PJR_check_satisfaction_given_committee, EJR_check_satisfaction_given_committee


def ilp(borda_ranking, l_cohesive):
    # here i'm imagining l_cohesive is built iteratively based on level (k)
    ipdb.set_trace() 
    n = len(borda_ranking) # assuming we have rankings of equal size (prefix-JR and Borda)
    borda_dict = {}
    for i in range(len(borda_ranking)): # makes things easier later
        borda_dict[i] = borda_ranking[i]
    
    x = cp.Variable((n,n), boolean=True) # x_{c,p} = 1 if candidate c is in position p, 0 otw
    y = cp.Variable((n,n), boolean=True) # y_{c,d} = 1 if candidate c is ranked above d in ranking, 0 otw

    constraints = []

    # variable constraints
    print("LINE:22", len(constraints))
    for c in range(n): #0
        constraints.append(cp.sum(x[c, :]) == 1) # 1 position per candidate
    print("LINE:25", len(constraints))
    for p in range(n): #6 
        constraints.append(cp.sum(x[:, p]) == 1) # 1 candidate per position

    print("LINE:29", len(constraints))
    # for c in range(n):  
    #     for d in range(n):
    #         if c != d:
    #             for p in range(n):
    #                 for q in range(p + 1, n):
    #                     constraints.append(y[c,d] >= x[c,p] + x[d, q] - 1) # enforces y_{c,d} to be correct
    
    print("LINE:36", len(constraints))
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c,d] + y[d,c] == 1)
    print("LINE:40", len(constraints))
    for c in range(n):
        constraints.append(y[c,c] == 0)
    
    # prefix-JR constraints:
    ipdb.set_trace()
    # for k in range(1, n + 1):
    #     # cohesive_groups = l_cohesive[k]['voter_sets']
    #     # alts = l_cohesive[k]['candidate_sets']
    #     alts = l_cohesive[k] 
    #     # ipdb.set_trace()
    #     # cohesive_k = cohesive_groups
    #     alts_k = alts
    #     try: 
    #         constraints.append(cp.sum([x[c, p] for c in alts_k for p in range(k)]) >= 1)
    #     except Exception as e:
    #         ipdb.set_trace()
    #         print(e)
            

    objective_terms = []

    for c in range(n):
        for d in range(n):
            if c != d:
                if borda_dict[d] < borda_dict[c]:
                    objective_terms.append(y[c,d])
    
    objective = cp.Minimize(cp.sum(objective_terms))

    # solving ilp
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.CBC, verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
   # output ranking
    x_sol = x.value
    ranking = [0] * n
    for c in range(n):
        for p in range(n):
            if x_sol[c, p] > 0.5:  
                ranking[p] = c
                break
    
    return ranking, problem.value


# usage:
# get all cohesive groups 
# run ilp, include it in the evaluation pipeline 
# -> sanity check: must satisfy JR always. otherwise something is very wrong

if __name__ == "__main__":
    num_candidates = 6
    num_voters = 6
    preferences = pd.DataFrame({
        'User_ID': [0, 1, 2, 3, 4, 5],
        'Ranked_Items': [[2,0,1,5,3,4], [2,0,1,5,3,4], [3,0,1,5,2,4], [3,0,1,5,2,4], [4,0,1,5,3,2], [4,0,1,5,3,2]]
    })
    all_candidates = preferences['Ranked_Items'].explode().unique().tolist()
    item_attribute = {0:0, 1:1, 2:1, 3:1, 4:1, 5:2}
    borda_ranking = [0,1,3,2,5,4]
    
    # dict should be {0: 2, 1: 1, 2: 4, 3: 3, 4: 5}
    
    # Two cohesive groups
    # l_cohesive = {1: {'voter_sets': [],
    #                      'candidate_sets': []}}
    cohesive_groups = {}
    satisfaction = {}
    candidates_to_consider = {}
    for prefix_idx in range(len(borda_ranking)):
        preferences_at_prefix = (
                preferences
                .assign(Ranked_Items=lambda df:
                        df['Ranked_Items'].apply(lambda x: x[:prefix_idx + 1]))
                .explode('Ranked_Items')
                .reset_index(drop=True)
            )
        voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(preferences_at_prefix, committee_size=prefix_idx+1, number_voters=num_voters)
        # ipdb.set_trace() 
        
        cohesive_groups[prefix_idx + 1] = {
            'voter_sets': l_cohesive[1]['voter_sets'], 
            'candidate_sets': l_cohesive[1]['candidate_sets']
        }
        
        candidates_to_consider[prefix_idx + 1] = np.unique(np.concatenate(l_cohesive[1]['candidate_sets'])).tolist() if len(l_cohesive[1]['candidate_sets']) > 0 else []
        satisfaction['JR'] = JR_check_satisfaction_given_committee(borda_ranking[:prefix_idx+1], partial_lists=preferences_at_prefix, all_candidates=all_candidates, n=num_voters, k=prefix_idx+1)
        print(satisfaction)
    ipdb.set_trace() 
    ranking, obj_value = ilp(
        borda_ranking, candidates_to_consider
    )
    
    print("Borda ranking order (best to worst):")
    print(f"  {borda_ranking}")

    
    print(f"\nPrefix-JR constrained ranking (best to worst):")
    print(f"  {ranking}")
    
    print(f"\nNumber of disagreements with Borda: {obj_value}")
    
    # Verify prefix-JR constraints
    # print("\nPrefix-JR verification:")
    # for k in range(1, num_candidates + 1):
    #     print(f"  Top {k}:", ranking[:k])
    #     for i, group in enumerate(l_cohesive):
    #         has_member = any(c in ranking[:k] for c in group)
    #         print(f"    Group {i+1} {group}: {'✓' if has_member else '✗'}")