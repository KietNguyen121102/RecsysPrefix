import os
import glob
import numpy as np
from scipy.stats import kendalltau
import argparse
import pickle
import yaml
import multiprocessing as mp
import ipdb 

from utils.io import load_sampled_preferences, format_sampled_rankings_kt, load_consensus_ranking_kt

# =============================================================================
# 2. Kendall Tau Calculation
# =============================================================================

def calculate_average_tau(user_rankings, consensus_map):
    taus = []
    for uid, user_list in user_rankings.items():
        if len(user_list) < 2:
            continue

        user_ranks = list(range(len(user_list)))
        default_rank = 100000
        consensus_ranks = [consensus_map.get(str(item), default_rank) for item in user_list]

        tau, _ = kendalltau(user_ranks, consensus_ranks)
        if not np.isnan(tau):
            taus.append(tau)

    return float(np.mean(taus)) if taus else 0.0

def calculate_average_tau_by_group(user_rankings, user_groups, consensus_map):
    """
    user_rankings: dict {uid: [item1, item2, ...]}
    user_groups:   dict {uid: group_label}  (exactly 3 groups, but works for any)
    consensus_map: dict {item: rank_int}

    Returns:
      overall_mean: float
      group_means:  dict {group_label: float}
      group_counts: dict {group_label: int}  (#users contributing taus)
    """
    taus_all = []
    group_taus = {}

    for uid, user_list in user_rankings.items():
        if len(user_list) < 2:
            continue

        user_ranks = list(range(len(user_list)))
        default_rank = 100000
        consensus_ranks = [consensus_map.get(str(item), default_rank) for item in user_list]

        tau, _ = kendalltau(user_ranks, consensus_ranks)
        if np.isnan(tau):
            continue

        taus_all.append(tau)

        g = user_groups.get(uid, "UNKNOWN")
        group_taus.setdefault(g, []).append(tau)

    overall_mean = float(np.mean(taus_all)) if taus_all else 0.0
    group_means = {g: (float(np.mean(v)) if len(v) else 0.0) for g, v in group_taus.items()}
    group_counts = {g: len(v) for g, v in group_taus.items()}

    return overall_mean, group_means, group_counts

# =============================================================================
# Parallel worker plumbing
# =============================================================================

_GLOBAL_USER_RANKINGS = None
_GLOBAL_USER_GROUPS = None

def _init_worker(user_rankings, user_groups):
    global _GLOBAL_USER_RANKINGS, _GLOBAL_USER_GROUPS
    _GLOBAL_USER_RANKINGS = user_rankings
    _GLOBAL_USER_GROUPS = user_groups
    
# def _score_one_consensus_file(file_path):
#     """Worker function: returns (method_name, avg_tau) or None if bad."""
#     global _GLOBAL_USER_RANKINGS

#     method_name = os.path.splitext(os.path.basename(file_path))[0]
#     consensus_map = load_consensus_ranking_kt(file_path)
#     if not consensus_map:
#         return None

#     avg_tau = calculate_average_tau(_GLOBAL_USER_RANKINGS, consensus_map)
#     return (method_name, avg_tau)

def _score_one_consensus_file(file_path):
    global _GLOBAL_USER_RANKINGS, _GLOBAL_USER_GROUPS

    method_name = os.path.splitext(os.path.basename(file_path))[0]
    consensus_map = load_consensus_ranking_kt(file_path)
    if not consensus_map:
        return None

    overall, by_group, counts = calculate_average_tau_by_group(
        _GLOBAL_USER_RANKINGS, _GLOBAL_USER_GROUPS, consensus_map
    )

    # return a richer record
    return {
        "method": method_name,
        "overall": overall,
        "by_group": by_group,
        "counts": counts,
    }



# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', '-p', default='', help='Path to sampled preferences (csv/pkl depending on your loader)')
    parser.add_argument('--agg', '-c', default='agg_results', help='Directory containing consensus TXT files')
    parser.add_argument('--dataset', '-d', default='ml-1m', choices=['ml-1m', 'goodreads'], help='Dataset name for config loading')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Number of parallel workers for consensus methods')
    parser.add_argument('--no-parallel', action='store_true', help='Disable multiprocessing (debug)')

    args = parser.parse_args()
    print(args)

    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)

    user_key = dataset_cfg['dataset']['keys']['user_key']
    # 1. Load User Data
    # user_rankings = load_sampled_preferences(args.pref)
    # user_rankings = format_sampled_rankings_kt(user_rankings, dataset_cfg)
    rankings = load_sampled_preferences(args.pref) #load_user_lists(args.pref)
    user_groups = pickle.load(open(dataset_cfg['dataset']['user_group_file_path'], 'rb'))
    user_rankings = rankings.merge(user_groups, on=user_key)
    user_groups = user_rankings.set_index(user_key)['entropy_bin'].to_dict()
    user_rankings = user_rankings.set_index(user_key)["Ranked_Items"].to_dict()

    # 2. Find all consensus files
    consensus_files = glob.glob(os.path.join(args.agg, "*.txt"))
    if not consensus_files:
        print(f"No .txt files found in {args.agg}")
        return

    print(f"\nComparing {len(consensus_files)} consensus methods against {len(user_rankings)} pref...")
    print("=" * 70)
    print(f"{'Method':<20} | {'Avg Kendall Tau':<15} | {'Interpretation'}")
    print("-" * 70)

    # 3. Score methods (parallel)
    results = []
    if args.no_parallel or args.workers <= 1:
        for fp in consensus_files:
            out = _score_one_consensus_file(fp) if args.no_parallel else None  # safe fallback
            if args.no_parallel:
                if out is not None:
                    results.append(out)
            else:
                # if workers <= 1, just run directly without global init
                method_name = os.path.splitext(os.path.basename(fp))[0]
                consensus_map = load_consensus_ranking_kt(fp)
                if not consensus_map:
                    continue
                avg_tau = calculate_average_tau(user_rankings, consensus_map)
                results.append((method_name, avg_tau))
    else:
        # Use "fork" on Linux by default; on some systems you might want "spawn"
        # with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(user_rankings,)) as pool:
        with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(user_rankings, user_groups)) as pool:
            for out in pool.imap_unordered(_score_one_consensus_file, consensus_files, chunksize=1):
                if out is not None:
                    results.append(out)

    
    results.sort(key=lambda d: d["overall"], reverse=True)

    # Print
    for r in results:
        name = r["method"]
        score = r["overall"]
        by_group = r["by_group"]
        counts = r["counts"]

        print(f"{name:<20} | overall={score:0.4f} | "
            + " ".join([f"{g}={by_group.get(g, 0.0):0.4f}(n={counts.get(g,0)})" for g in sorted(by_group.keys())]))

    # 6. Save results
    out_path = os.path.join(args.agg, "kendall_results_by_group.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results -> {out_path}")

    print("=" * 70)
    print("Note: Kendall Tau ranges from 1.0 (Identical) to -1.0 (Completely Reversed).")
    print("A higher score means the consensus ranking preserves the individual user preferences better.")

def test(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', '-p', default='/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/sampled_rankings.pkl', help='Path to sampled preferences (csv/pkl depending on your loader)')
    parser.add_argument('--agg', '-c', default='/data2/rsalgani/Prefix/ml-1m/agg_files/sample_0/', help='Directory containing consensus TXT files')
    parser.add_argument('--dataset', '-d', default='ml-1m', choices=['ml-1m', 'goodreads'], help='Dataset name for config loading')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Number of parallel workers for consensus methods')
    parser.add_argument('--no-parallel', action='store_true', help='Disable multiprocessing (debug)')

    args = parser.parse_args()
    print(args)

    with open(f"/u/rsalgani/2024-2025/RecsysPrefix/data/{args.dataset}/params.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    ipdb.set_trace()
    # 1. Load User Data
    rankings = load_sampled_preferences(args.pref) #load_user_lists(args.pref)
    user_groups = pickle.load(open(dataset_cfg['dataset']['user_group_file_path'], 'rb'))
    user_rankings = rankings.merge(user_groups, on=user_key)
    user_groups = user_rankings.set_index(user_key)['entropy_bin'].to_dict()
    user_rankings = user_rankings.set_index(user_key)["Ranked_Items"].to_dict()


if __name__ == "__main__":
    main()
    # test() 
