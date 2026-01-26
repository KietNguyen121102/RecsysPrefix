import os
import glob
import numpy as np
import argparse
from tqdm import tqdm 
import pandas as pd 
import pickle 

from utils.cohesive_group_search import find_maximal_cohesive_groups, find_all_cohesive_groups
from utils.axiom_checks import JR_check_satisfaction_given_committee, PJR_check_satisfaction_given_committee, EJR_check_satisfaction_given_committee
from utils.io import load_consensus_ranking, load_sampled_preferences

"""
Goodreads Axiom Satisfaction Calculator
=======================================
Evaluates whether consensus rankings satisfy proportionality axioms:

- JR (Justified Representation): No large group of voters with a common
  approved candidate can be completely unrepresented.

- PJR (Proportional JR): l-cohesive groups get at least l representatives.

- EJR (Extended JR): Each voter in an l-cohesive group must be l-satisfied.

This is evaluated at EVERY PREFIX of the committee (prefix-based evaluation).
"""

# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Goodreads Axiom Satisfaction Calculator")
    parser.add_argument('--agg', '-a', default='consensus_results', help='Directory containing consensus TXT files')
    parser.add_argument('--pref', '-p', default='sampled_rankings.pkl', help='Path to sampled preferences')
    args = parser.parse_args()

    print("=" * 70)
    print("GOODREADS AXIOM SATISFACTION EVALUATION")
    print("=" * 70)
    print(f"(1) Loading data")
    print("-" * 70)
    
    # 1. Load User-Level Fully Ordered Preference Lists
    preferences = load_sampled_preferences(args.pref)
    number_voters = len(preferences['User_ID'].unique())
    all_candidates = preferences.explode('Ranked_Items')['Ranked_Items'].unique()
    number_candidates = len(all_candidates)
    
    print(f"(2) Loading Consensus Files")
    print("-" * 70)
    
    # 2. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    print(f'RUN STATS for: {args.agg}')
    print("-" * 70)
    print("Number of Voters:", number_voters)
    print("Number of Candidates:", number_candidates)
    print("Number of methods to evaluate:", len(consensus_files))
    print("-" * 70)
    
    print(f"(3) Calculating Axiom Satisfaction")
    results = []
    
    # 3. Iterate over each method
    for idx, file_path in enumerate(consensus_files):
        satisfaction = {'JR': [], 'PJR': [], 'EJR': []}
        method_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"({idx+1}/{len(consensus_files)}) Method: {method_name}")
        
        # Load the ranking
        committee = load_consensus_ranking(file_path)
        if not committee:
            continue
        
        # Evaluate at each prefix
        for prefix_idx in tqdm(range(len(committee)), desc=f"  Prefixes"):
            preferences_at_prefix = (
                preferences
                .assign(Ranked_Items=lambda df:
                        df['Ranked_Items'].apply(lambda x: x[:prefix_idx + 1]))
                .explode('Ranked_Items')
                .reset_index(drop=True)
            )
            
            # Calculate Satisfaction Over Axioms
            voter_sets, candidate_sets, l_cohesive = find_all_cohesive_groups(
                preferences_at_prefix, 
                committee_size=prefix_idx+1, 
                number_voters=number_voters
            )
            
            satisfaction['JR'].append(
                JR_check_satisfaction_given_committee(
                    committee[:prefix_idx+1], 
                    partial_lists=preferences_at_prefix, 
                    all_candidates=all_candidates, 
                    n=number_voters, 
                    k=prefix_idx+1
                )
            )
            satisfaction['PJR'].append(
                PJR_check_satisfaction_given_committee(
                    committee[:prefix_idx+1], 
                    partial_lists=preferences_at_prefix, 
                    l_cohesive=l_cohesive
                )
            )
            satisfaction['EJR'].append(
                EJR_check_satisfaction_given_committee(
                    committee[:prefix_idx+1], 
                    preferences_at_prefix
                )
            )
            
        results.append((method_name, satisfaction))
    
    # 4. Print Results Table
    print("\n" + "=" * 60)
    print(f"{'Method':<20} | {'JR':^5} | {'PJR':^5} | {'EJR':^5}")
    print("-" * 60)

    for method, satisfaction in results:
        jr, pjr, ejr = satisfaction['JR'], satisfaction['PJR'], satisfaction['EJR']
        def mark(x): return "✓" if all(x) else "✗"
        print(f"{method:<20} | {mark(jr):^5} | {mark(pjr):^5} | {mark(ejr):^5}")

    print("=" * 60)
    
    # 5. Save results
    results_df = pd.DataFrame({
        method: {k: all(v) for k, v in metrics.items()}
        for method, metrics in results
    }).T
    
    output_path = os.path.join(args.agg, 'axiom_satisfaction_results.pkl')
    pickle.dump(results_df, open(output_path, 'wb'))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()



