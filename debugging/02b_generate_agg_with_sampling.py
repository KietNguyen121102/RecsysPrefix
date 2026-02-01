import numpy as np
import csv
import os
from collections import defaultdict
import argparse
import sys
from tqdm import tqdm 
import ipdb 
import pandas as pd 
import pickle 
import yaml

from utils.vanilla_aggregation_methods import *
from utils.fair_aggregation_methods import *
from utils.borda_prefix_jr_ilp import * 
from utils.io import load_rankings_to_df
from utils.sampling_logic import generate_sample_sets, generate_sample_sets_stratified_by_bin

# =============================================================================
# Data Loading
# =============================================================================
# =============================================================================
# Helper Functions
# =============================================================================

# def format_sampled_rankings(sampled_preferences):
#     """
#     Convert sampled preferences DataFrame to list of rankings.
#     """
#     rankings = []
#     for _, row in sampled_preferences.iterrows():
#         rankings.append(row['Ranked_Items'])
#     return rankings

# =============================================================================
# Sampling Logic
# =============================================================================

# def generate_sample_sets(n_samples, n_users, n_items, all_items, preferences): 
#     all_users = list(preferences['User_ID'].unique())
#     dfs, items, users = [], [], []
#     for seed in range(n_samples):
#         user_idx = np.random.choice(len(all_users), size=n_users, replace=False)
#         item_idx = np.random.choice(all_items, size=n_items, replace=False)
#         sampled_pref = preferences[preferences['User_ID'].isin([all_users[i] for i in user_idx])]
#         sampled_pref['Ranked_Items'] = sampled_pref['Ranked_Items'].apply(lambda x: [item for item in x if item in item_idx])
#         dfs.append(sampled_pref)
#         items.append(item_idx)
#         users.append(user_idx)
#     return dfs, items, users

# =============================================================================
# Main Execution
# =============================================================================

VANILLA_METHODS = {
    # 'CombMIN': comb_min, 'CombMAX': comb_max, 'CombSUM': comb_sum,
    # 'CombANZ': comb_anz, 'CombMNZ': comb_mnz,
    # 'MC1': mc1, 'MC2': mc2, 'MC3': mc3, 'MC4': mc4,
    # 'BordaCount': borda_count, 'Dowdall': dowdall,
    # 'Median': median_rank, 'Mean': mean_rank,
    # 'RRF': rrf, 'iRANK': irank, 'ER': er,
    # 'PostNDCG': postndcg, 'CG': cg, 'DIBRA': dibra
    'BordaCount': borda_count,
}

FAIR_METHODS = { 'KuhlmanConsensus': Consensus,
                # 'FairMedian': FairILP, 
                }

OUR_METHODS = {
    'Our_Prefix_ILP': ilp_prefix_jr, 
    # 'Our_Prefix_Fair_ILP': ilp_prefix_jr_plus_fair,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='ml-1m')
    parser.add_argument('--input', '-i', default='recommendations.csv', help='Input CSV file')
    parser.add_argument('--outdir', '-o', default='consensus_results', help='Output directory')
    parser.add_argument('--group-file', '-g', default='data/ml-1m/item_groups.pkl', help='Group attributes CSV file')
    parser.add_argument('--top-k', '-k', type=int, default=3453, help='Items to save per file')
    parser.add_argument('--user-sample-size', '-us', type=int, default=10, help='Number of users to sample')
    parser.add_argument('--item-sample-size', '-is', type=int, default=20, help='Number of items to sample')
    parser.add_argument('--n-samples', '-n', type=int, default=1, help='Number of samples to generate')
    args = parser.parse_args()

    # ipdb.set_trace() 
    
    print(args)
    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    print(dataset_cfg)
    
    
    print("=" * 60)
    print("Rank Aggregation (Separate Output Files)")
    print("=" * 60)

    # 1. Create Output Directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created directory: {args.outdir}")
    else:
        print(f"Using existing directory: {args.outdir}")

    # 2. Load Data 
    rankings, all_items = load_rankings_to_df(dataset_cfg)
    group_df = pickle.load(open(args.group_file, 'rb'))
    
    # 3. Generate Sample Sets
    sampled_rankings, sampled_items, sampled_users = generate_sample_sets(n_samples=args.n_samples, n_users=args.user_sample_size, n_items=args.item_sample_size, all_items=list(all_items), preferences=rankings)
    formatted_sampled_rankings  = [format_sampled_rankings(sampled_ranking) for sampled_ranking in sampled_rankings]
    
    # 3. Process each method and save immediately
    for seed in range(args.n_samples):
        write_dir = os.path.join(args.outdir, f"sample_{seed}")
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)
        
        pickle.dump(sampled_rankings[seed], open(os.path.join(write_dir, "sampled_rankings.pkl"), 'wb'))
        pickle.dump(sampled_items[seed], open(os.path.join(write_dir, "sampled_items.pkl"), 'wb'))
        pickle.dump(sampled_users[seed], open(os.path.join(write_dir, "sampled_users.pkl"), 'wb'))
        
        total = len(VANILLA_METHODS) + len(FAIR_METHODS) + len(OUR_METHODS)
        print(f"\nProcessing {total} methods...")
        
        for i, (name, method) in enumerate(VANILLA_METHODS.items(), 1):
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            
            # Calculate Ranking
            result = method(formatted_sampled_rankings[seed], sampled_items[seed])
            # Construct Filename
            file_name = f"{name}.txt"
            file_path = os.path.join(write_dir, file_name)
            
            # Write to File
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Method: {name}\n")
                f.write(f"# Rank ItemID Score\n")
                for rank, (item, score) in enumerate(result, 1):
                    f.write(f"{rank} {item} {score:.6f}\n")
            
            print(f"-> Saved to {file_path}")
        print(name, result)  
        alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes = process_for_fair_ranking(sampled_items[seed], group_df, formatted_sampled_rankings[seed], dataset_cfg)
        pickle.dump(
            {'alphas': alphas,
            'betas': betas,
            'attributes_map': attributes_map,
            'idx_to_item': idx_to_item,
            'num_attributes': num_attributes}, open(os.path.join(write_dir, "fair_ranking_process.pkl"), 'wb')) 
        
        all_candidates = sampled_rankings[seed]['Ranked_Items'].explode().unique().tolist()
        item_to_idx = dict(zip(all_candidates, range(len(all_candidates))))
        user_to_idx = dict(zip(sampled_rankings[seed]['User_ID'].unique(), range(len(sampled_rankings[seed]['User_ID'].unique()))))
        
        # borda_ranking = [item_to_idx[item] for item in borda_ranking]
        
        
        for i, (name, method) in enumerate(FAIR_METHODS.items(), len(VANILLA_METHODS)+1):
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            
            # ipdb.set_trace() 
            # Calculate Ranking
            try: 
                result = method(alphas, betas, ranks_for_fairness, attributes_map, num_attributes)
                result = [idx_to_item[i] for i in result]
                
                # Construct Filename
                file_name = f"{name}.txt"
                file_path = os.path.join(write_dir, file_name)
                
                # Write to File
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Method: {name}\n")
                    f.write(f"# Rank ItemID Score\n")
                    for rank, item, in enumerate(result, 1):
                        f.write(f"{rank} {item}\n")
                
                print(f"-> Saved to {file_path}")
            except IndexError as e:
                raise FairnessConstraintsInfeasible(
                    "Consensus crashed due to infeasible group lower bounds "
                    "(not enough items in a group to satisfy floor(alpha*TOPK))."
                ) from e
        
        print(name, result)  
        for i, (name, method) in enumerate(OUR_METHODS.items(), len(VANILLA_METHODS)+len(FAIR_METHODS)+1):
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            borda_ranking = borda_count(ranks_for_fairness, list(range(len(idx_to_item))))
            borda_ranking = [x for x, _ in borda_ranking]
            n_voters = len(ranks_for_fairness)
            approvals_by_k = {}
            for prefix_idx in range(len(borda_ranking)):
                k = prefix_idx + 1
                preferences_at_prefix = (
                    sampled_rankings[seed]
                    .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
                    .explode('Ranked_Items')
                    .reset_index(drop=True)
                )
                preferences_at_prefix['Ranked_Items'] = preferences_at_prefix['Ranked_Items'].map(item_to_idx)
                preferences_at_prefix['User_ID'] = preferences_at_prefix['User_ID'].map(user_to_idx)
        
                approvals_k = {}
                for v in range(n_voters):
                    approvals_k[v] = (
                        preferences_at_prefix[preferences_at_prefix["User_ID"] == v]["Ranked_Items"]
                        .unique().tolist()
                    )
                approvals_by_k[k] = approvals_k
            
            
            # Calculate Ranking
            if name == 'Our_Prefix_ILP':
                result, _ = method(borda_ranking, approvals_by_k, n_voters)
        
            if name == 'Our_Prefix_Fair_ILP':
                result, _ = method(borda_ranking, approvals_by_k, n_voters, alphas, betas, k, attributes_map, num_attributes)
        
            result = [idx_to_item[i] for i in result] 
            # Construct Filename
            file_name = f"{name}.txt"
            file_path = os.path.join(write_dir, file_name)
            
            # Write to File
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Method: {name}\n")
                f.write(f"# Rank ItemID Score\n")
                for rank, item, in enumerate(result, 1):
                    f.write(f"{rank} {item}\n")
            
            print(f"-> Saved to {file_path}")
        print(name, result)
        print("\n" + "=" * 60)
        print("Aggregation Complete!")
        print(f"All files are located in: ./{args.outdir}/")
        print("=" * 60)

def test(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/full_recset.csv', help='Input CSV file')
    parser.add_argument('--outdir', '-o', default='/data2/rsalgani/Prefix/ml-1m/agg_files', help='Output directory')
    parser.add_argument('--group-file', '-g', default='data/ml-1m/item_groups.pkl', help='Group attributes CSV file')
    # parser.add_argument('--top-k', '-k', type=int, default=3453, help='Items to save per file')
    parser.add_argument("--dataset", type=str, choices=['ml-1m', 'goodreads'], default='ml-1m')
    parser.add_argument('--user-sample-size', '-us', type=int, default=10, help='Number of users to sample')
    parser.add_argument('--item-sample-size', '-is', type=int, default=20, help='Number of items to sample')
    parser.add_argument('--n-samples', '-n', type=int, default=1, help='Number of samples to generate')
    args = parser.parse_args()

    print(args)
    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    print(dataset_cfg)
    
    print("=" * 60)
    print("Rank Aggregation (Separate Output Files)")
    print("=" * 60)

    # 1. Create Output Directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"Created directory: {args.outdir}")
    else:
        print(f"Using existing directory: {args.outdir}")

    # 2. Load Data
    # ipdb.set_trace() 
    rankings_df, all_items = load_rankings_to_df(dataset_cfg)
    
   
    group_df = pickle.load(open(args.group_file, 'rb'))
    
    # 3. Generate Sample Sets
    sampled_rankings, sampled_items, sampled_users = generate_sample_sets(n_samples=args.n_samples, n_users=args.user_sample_size, n_items=args.item_sample_size, all_items=list(all_items), preferences=rankings_df)
    formatted_sampled_rankings  = [format_sampled_rankings(sampled_ranking) for sampled_ranking in sampled_rankings]
    
    
    
    # 3. Process each method and save immediately
    for seed in range(args.n_samples):
        write_dir = os.path.join(args.outdir, f"sample_{seed}")
        if not os.path.exists(write_dir):
            os.makedirs(write_dir, exist_ok=True)
        
        total = len(VANILLA_METHODS) + len(FAIR_METHODS)
        print(f"\nProcessing {total} methods...")
        
            
        alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes = process_for_fair_ranking(sampled_items[seed], group_df, formatted_sampled_rankings[seed])
       
        for i, (name, method) in enumerate(FAIR_METHODS.items(), len(VANILLA_METHODS)+1):
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            
            # Calculate Ranking
            result = method(alphas, betas, ranks_for_fairness, attributes_map, num_attributes)
            result = [idx_to_item[i] for i in result]
            # ipdb.set_trace() 
            # Construct Filename
            file_name = f"{name}.txt"
            file_path = os.path.join(write_dir, file_name)
            
            # # Write to File
            # with open(file_path, 'w', encoding='utf-8') as f:
            #     f.write(f"# Method: {name}\n")
            #     # f.write(f"# Rank ItemID Score\n")
            #     for rank, item, in enumerate(result, 1):
            #         f.write(f"{rank} {item}\n")
            
            # print(f"-> Saved to {file_path}")

        borda_ranking = borda_count(ranks_for_fairness, list(range(len(idx_to_item))))
        borda_ranking = [x for x, _ in borda_ranking]
        # borda_ranking = [idx_to_item[i] for i in borda_ranking]
        n_voters = len(ranks_for_fairness)
        for i, (name, method) in enumerate(OUR_METHODS.items(), len(VANILLA_METHODS)+len(FAIR_METHODS)+1):
            # ipdb.set_trace() 
            print(f"  [{i:2d}/{total}] Running {name}...", end=" ", flush=True)
            
            approvals_by_k = {}
        
            for prefix_idx in range(len(borda_ranking)):
                k = prefix_idx + 1
                preferences_at_prefix = (
                    sampled_rankings[seed]
                    .assign(Ranked_Items=lambda df: df['Ranked_Items'].apply(lambda x: x[:k]))
                    .explode('Ranked_Items')
                    .reset_index(drop=True)
                )

                approvals_k = {}
                for v in range(n_voters):
                    approvals_k[v] = (
                        preferences_at_prefix[preferences_at_prefix["User_ID"] == v]["Ranked_Items"]
                        .unique().tolist()
                    )
                approvals_by_k[k] = approvals_k
        
            
            # Calculate Ranking
            if name == 'Our_Prefix_ILP':
                result, _ = method(borda_ranking, approvals_by_k, n_voters)
        
            else:
                result, _ = method(borda_ranking, approvals_by_k, n_voters, alphas, betas, k, attributes_map, num_attributes)

            # ipdb.set_trace() 
            result = [idx_to_item[i] for i in result]
            # ipdb.set_trace() 
            # Construct Filename
            file_name = f"{name}.txt"
            file_path = os.path.join(write_dir, file_name)
            
            # # Write to File
            # with open(file_path, 'w', encoding='utf-8') as f:
            #     f.write(f"# Method: {name}\n")
            #     # f.write(f"# Rank ItemID Score\n")
            #     for rank, item, in enumerate(result, 1):
            #         f.write(f"{rank} {item}\n")
            
            # print(f"-> Saved to {file_path}")
            
        
        print("\n" + "=" * 60)
        print("Aggregation Complete!")
        print(f"All files are located in: ./{args.outdir}/")
        print("=" * 60)


if __name__ == "__main__":
    # test() 
    main()