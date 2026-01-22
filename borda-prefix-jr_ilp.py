import cvxpy as cp
import numpy as np
import pandas as pd 
import ipdb 
from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
from utils.axiom_checks import JR_check_satisfaction_given_committee, PJR_check_satisfaction_given_committee, EJR_check_satisfaction_given_committee


def ilp_old(borda_ranking, l_cohesive):
    # here i'm imagining l_cohesive is built iteratively based on level (k)
    ipdb.set_trace() 
    n = len(borda_ranking) # assuming we have rankings of equal size (prefix-JR and Borda)
    # borda_dict = {}
    # for i in range(len(borda_ranking)): # makes things easier later
    #     borda_dict[i] = borda_ranking[i]
    borda_dict = {cand: pos for pos, cand in enumerate(borda_ranking)}

    
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



import cvxpy as cp
import numpy as np

def ilp(borda_ranking, cohesive_groups):
    n = len(borda_ranking)

    # candidate -> Borda position (0 = best)
    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)   # x[c,p]
    y = cp.Variable((n, n), boolean=True)   # y[c,d]
    pos = cp.Variable(n, integer=True)      # pos[c]

    constraints = []

    # Permutation constraints on x
    for c in range(n):
        constraints.append(cp.sum(x[c, :]) == 1)
    for p in range(n):
        constraints.append(cp.sum(x[:, p]) == 1)

    # Define pos[c] = sum_p p * x[c,p]
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(np.arange(n), x[c, :])))

    # y tournament structure
    for c in range(n):
        constraints.append(y[c, c] == 0)
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # Link y to pos via big-M
    # If y[c,d]=1 => pos[c] <= pos[d] - 1
    # If y[c,d]=0 => constraint relaxed
    M = n  # safe big-M
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            # Optional: also enforce the opposite direction when y[c,d]=0
            # i.e., if y[c,d]=0 then pos[c] >= pos[d] + 1
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # Prefix-JR constraints (per cohesive group)
    # cohesive_groups[k]['candidate_sets'] is a list of sets/lists of candidates
    for k in range(1, n + 1):
        cand_sets = cohesive_groups.get(k, {}).get('candidate_sets', [])
        ipdb.set_trace()
        for S in cand_sets:
            constraints.append(
                cp.sum([x[c, p] for c in S for p in range(k)]) >= 1
            )

    # Objective: minimize pairwise disagreements with Borda
    # If Borda wants c above d (borda_pos[c] < borda_pos[d]),
    # then disagreement occurs when y[d,c] = 1.
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])  # Borda says c>d, but model says d>c
    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.CBC, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    # Decode ranking from x
    x_sol = x.value
    ranking = [None] * n
    for c in range(n):
        p = int(np.argmax(x_sol[c, :]))
        ranking[p] = c

    return ranking, problem.value



import cvxpy as cp
import numpy as np
import math

def ilp_prefix_jr(borda_ranking, approvals_by_k, n_voters):
    """
    borda_ranking: list[int] length n (candidate IDs 0..n-1)
    approvals_by_k: dict[int, dict[int, list[int]]]
        approvals_by_k[k][v] = list of candidates voter v approves at prefix k
        (in your setup, that's the voter's top-k ranked items)
    n_voters: int
    """
    n = len(borda_ranking)

    # candidate -> Borda position (0 = best)
    borda_pos = {cand: i for i, cand in enumerate(borda_ranking)}

    x = cp.Variable((n, n), boolean=True)   # x[c,p] = 1 if cand c at position p
    y = cp.Variable((n, n), boolean=True)   # y[c,d] = 1 if c ranked above d
    pos = cp.Variable(n, integer=True)      # pos[c] = position index of candidate c

    constraints = []

    # Permutation constraints on x
    for c in range(n):
        constraints.append(cp.sum(x[c, :]) == 1)
    for p in range(n):
        constraints.append(cp.sum(x[:, p]) == 1)

    # Define pos[c] = sum_p p * x[c,p]
    for c in range(n):
        constraints.append(pos[c] == cp.sum(cp.multiply(np.arange(n), x[c, :])))

    # y tournament structure
    for c in range(n):
        constraints.append(y[c, c] == 0)
    for c in range(n):
        for d in range(c + 1, n):
            constraints.append(y[c, d] + y[d, c] == 1)

    # Link y to pos via big-M (tight, two-sided)
    M = n
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            # if y[c,d]=1 then pos[c] <= pos[d]-1
            constraints.append(pos[c] <= pos[d] - 1 + M * (1 - y[c, d]))
            # if y[c,d]=0 then pos[c] >= pos[d]+1
            constraints.append(pos[c] >= pos[d] + 1 - M * y[c, d])

    # ------------------------------------------------------------
    # Prefix-JR constraints: JR must hold for every prefix k
    # ------------------------------------------------------------
    # z[v,k] = 1 if voter v is represented in top-(k+1) (k is 0-indexed here)
    z = cp.Variable((n_voters, n), boolean=True)

    for k in range(1, n + 1):
        quota = math.ceil(n_voters / k)  # match JR definition cleanly
        topk_positions = range(k)
        approvals_k = approvals_by_k[k]  # dict voter -> list of approved cands at prefix k

        # Link z[v,k-1] to whether top-k contains ANY approved candidate for voter v
        for v in range(n_voters):
            A_vk = approvals_k.get(v, [])
            if len(A_vk) == 0:
                # If voter approves nothing at this prefix (shouldn't happen in your setup),
                # force them unrepresented.
                constraints.append(z[v, k - 1] == 0)
                continue

            # z[v,k-1] <= sum_{a in A_vk} sum_{p<k} x[a,p]
            constraints.append(
                z[v, k - 1] <= cp.sum([x[a, p] for a in A_vk for p in topk_positions])
            )

        # JR condition per your checker:
        # For each candidate c, among voters who approve c, you cannot have quota (or more)
        # who are unrepresented => sum_{v in Vc} (1 - z[v,k-1]) <= quota - 1
        for c in range(n):
            Vc = [v for v in range(n_voters) if c in approvals_k.get(v, [])]
            if len(Vc) < quota:
                # Even if all are unrepresented, can't reach quota -> no JR constraint needed
                continue

            constraints.append(
                cp.sum([1 - z[v, k - 1] for v in Vc]) <= quota - 1
            )

    # Objective: minimize pairwise disagreements with Borda
    obj_terms = []
    for c in range(n):
        for d in range(n):
            if c == d:
                continue
            if borda_pos[c] < borda_pos[d]:
                obj_terms.append(y[d, c])  # disagreement if model says d > c

    objective = cp.Minimize(cp.sum(obj_terms))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        problem.solve(solver=cp.CBC, verbose=False)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    # Decode ranking from x
    x_sol = x.value
    ranking = [None] * n
    for c in range(n):
        p = int(np.argmax(x_sol[c, :]))
        ranking[p] = c

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
    borda_ranking = [5,1, 3, 2, 4,0] # SHOULD FAIL 
    # borda_ranking = [0,1,2,3,4,5] # SHOULD PASS
    # borda_ranking = [2,3,4,0,1,5] # SHOULD PASS
    # borda_ranking = [1,5,4,3,2,0] # SHOULD FAIL
    print(preferences)
    
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
        
        
        
        # for i, v in enumerate(l_cohesive[1]['voter_sets']): 
        #     ipdb.set_trace() 
        #     if len(v) >= num_voters/(prefix_idx+1):
        #         cohesive_groups[prefix_idx + 1]['voter_sets'] = l_cohesive[1]['voter_sets'][i]
        #         cohesive_groups[prefix_idx + 1]['candidate_sets'] = l_cohesive[1]['candidate_sets'][i]
        
        cohesive_groups[prefix_idx + 1] = {}
        cohesive_groups[prefix_idx + 1]['voter_sets'] = []
        cohesive_groups[prefix_idx + 1]['candidate_sets'] = []
        for i, v in enumerate(voter_sets): 
            # ipdb.set_trace() 
            if len(v) >= num_voters/(prefix_idx+1):
                cohesive_groups[prefix_idx + 1]['voter_sets'].append(voter_sets[i])
                cohesive_groups[prefix_idx + 1]['candidate_sets'].append(candidate_sets[i])
                

        # print(cohesive_groups)
        # cohesive_groups[prefix_idx + 1] = {
        #     'voter_sets': l_cohesive[1]['voter_sets'], 
        #     'candidate_sets': l_cohesive[1]['candidate_sets']
        # }
        
        candidates_to_consider[prefix_idx + 1] = np.unique(np.concatenate(l_cohesive[1]['candidate_sets'])).tolist() if len(l_cohesive[1]['candidate_sets']) > 0 else []
        satisfaction['JR'] = JR_check_satisfaction_given_committee(borda_ranking[:prefix_idx+1], partial_lists=preferences_at_prefix, all_candidates=all_candidates, n=num_voters, k=prefix_idx+1)
        print(satisfaction)
    # ipdb.set_trace() 
    
    
    
    # ranking, obj_value = ilp(
    #     borda_ranking, cohesive_groups
    # )
    
    
    approvals_by_k = {}

    for prefix_idx in range(len(borda_ranking)):
        k = prefix_idx + 1
        preferences_at_prefix = (
            preferences
            .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
            .explode('Ranked_Items')
            .reset_index(drop=True)
        )

        approvals_k = {}
        for v in range(num_voters):
            approvals_k[v] = (
                preferences_at_prefix[preferences_at_prefix["User_ID"] == v]["Ranked_Items"]
                .unique().tolist()
            )
        approvals_by_k[k] = approvals_k
        
    ranking, obj_value = ilp_prefix_jr(borda_ranking, approvals_by_k, n_voters=num_voters)
    
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