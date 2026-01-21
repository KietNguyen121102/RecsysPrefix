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
        constraints.append(cp.sum([x[c, p] for c in alts_k for p in range(k)]) >= 1)

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
    num_candidates = 5
    borda_ranking = [2, 1, 4, 3, 5]
    # dict should be {0: 2, 1: 1, 2: 4, 3: 3, 4: 5}
    
    # Two cohesive groups
    l_cohesive = {1: {'voter_sets': [],
                         'candidate_sets': []}}
    
    ranking, obj_value = ilp(
        borda_ranking, l_cohesive, num_candidates
    )
    
    print("Borda ranking order (best to worst):")
    print(f"  {borda_ranking}")

    
    print(f"\nPrefix-JR constrained ranking (best to worst):")
    print(f"  {ranking}")
    
    print(f"\nNumber of disagreements with Borda: {obj_value}")
    
    # Verify prefix-JR constraints
    print("\nPrefix-JR verification:")
    for k in range(1, num_candidates + 1):
        print(f"  Top {k}:", ranking[:k])
        for i, group in enumerate(l_cohesive):
            has_member = any(c in ranking[:k] for c in group)
            print(f"    Group {i+1} {group}: {'✓' if has_member else '✗'}")