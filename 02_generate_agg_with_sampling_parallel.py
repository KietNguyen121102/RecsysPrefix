import os
import sys
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.vanilla_aggregation_methods import *
from utils.fair_aggregation_methods import *

# =============================================================================
# Data Loading
# =============================================================================

def load_rankings_to_df(filepath="recommendations.csv"):
    print(f"Loading data from '{filepath}'...")

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    required = {"User_ID", "Movie_ID", "Estimated_Rating"}
    if not required.issubset(df.columns):
        print("Error: CSV must have columns: User_ID, Movie_ID, Estimated_Rating")
        sys.exit(1)

    df["Estimated_Rating"] = pd.to_numeric(df["Estimated_Rating"], errors="coerce")
    df = df.dropna(subset=["Estimated_Rating"])

    df = df.sort_values(["User_ID", "Estimated_Rating"], ascending=[True, False])

    rankings_df = (
        df.groupby("User_ID")["Movie_ID"]
          .apply(list)
          .reset_index(name="Ranked_Items")
    )

    all_items = set(df["Movie_ID"].unique())

    print(f"Loaded rankings for {len(rankings_df)} users.")
    print(f"Total unique items found: {len(all_items)}")

    return rankings_df, all_items

# =============================================================================
# Helper Functions
# =============================================================================

def validate_full_rankings(rankings, candidate_items, context=""):
    """
    Enforce that each ranking is a FULL permutation of candidate_items.

    Checks:
      - correct length
      - no duplicates
      - contains exactly the same items as candidate_items
    """
    cand_list = list(candidate_items)
    cand_set = set(cand_list)
    d = len(cand_list)

    if len(cand_set) != d:
        raise ValueError(f"{context} candidate_items contains duplicates (size={d}, unique={len(cand_set)}).")

    for idx, r in enumerate(rankings):
        if not isinstance(r, (list, tuple)):
            raise TypeError(f"{context} ranking[{idx}] is not a list/tuple: {type(r)}")

        if len(r) != d:
            missing = sorted(cand_set - set(r))
            extra = sorted(set(r) - cand_set)
            raise ValueError(
                f"{context} ranking[{idx}] is partial or wrong length: len={len(r)} expected={d}\n"
                f"  missing={missing[:30]}\n"
                f"  extra={extra[:30]}"
            )

        if len(set(r)) != len(r):
            # show first few duplicates
            seen = set()
            dupes = []
            for x in r:
                if x in seen:
                    dupes.append(x)
                seen.add(x)
            raise ValueError(f"{context} ranking[{idx}] contains duplicates: {dupes[:30]}")

        if set(r) != cand_set:
            missing = sorted(cand_set - set(r))
            extra = sorted(set(r) - cand_set)
            raise ValueError(
                f"{context} ranking[{idx}] does not match candidate set.\n"
                f"  missing={missing[:30]}\n"
                f"  extra={extra[:30]}"
            )

def format_sampled_rankings(sampled_preferences: pd.DataFrame):
    return [row["Ranked_Items"] for _, row in sampled_preferences.iterrows()]

def process_for_fair_ranking_safe(candidates, group_df, ranks_for_fairness, complete=True):
    """
    Safe version:
    - drops candidate items missing from group_df (no label)
    - uses d = number of labeled items (NOT len(candidates))
    - compresses group labels to 0..g-1
    - filters+maps rankings and optionally completes them to full permutations
    """
    # Restrict to candidates that have group labels
    attributes_df = (
        group_df[group_df.item.isin(candidates)][["item", "binned"]]
        .drop_duplicates(subset=["item"])
        .sort_values("item")
        .reset_index(drop=True)
    )

    candidates_set = set(candidates)
    labeled_set = set(attributes_df["item"])
    missing_labels = sorted(candidates_set - labeled_set)
    if missing_labels:
        print(f"[WARN] {len(missing_labels)} sampled items missing group labels; dropping. "
              f"Example: {missing_labels[:10]}")

    # The effective candidate universe is ONLY those with labels
    items = attributes_df["item"].tolist()
    d = len(items)
    if d == 0:
        raise ValueError("No labeled candidates left after filtering by group_df. "
                         "Check group_df.item coverage or your sampled_items.")

    # item -> 0..d-1 and back
    item_to_idx = {item: i for i, item in enumerate(items)}
    idx_to_item = {i: item for item, i in item_to_idx.items()}

    # compress binned values to 0..g-1 for this sample
    group_labels = attributes_df["binned"].tolist()
    uniq_groups = sorted(set(group_labels))
    group_to_gid = {g: i for i, g in enumerate(uniq_groups)}
    num_attributes = len(uniq_groups)

    # idx -> attribute (group id)
    idx_to_attribute = {
        item_to_idx[item]: group_to_gid[g]
        for item, g in zip(items, group_labels)
    }

    # map rankings to idx space, dropping unlabeled candidates
    mapped_rankings = []
    for r in ranks_for_fairness:
        filtered = [x for x in r if x in item_to_idx]
        mapped = [item_to_idx[x] for x in filtered]

        # de-duplicate while preserving order
        seen = set()
        mapped = [x for x in mapped if not (x in seen or seen.add(x))]
        mapped_rankings.append(mapped)

    if complete:
        universe = list(range(d))
        completed = []
        for r in mapped_rankings:
            seen = set(r)
            missing = [x for x in universe if x not in seen]
            completed.append(r + missing)
        mapped_rankings = completed

    alphas = [1.0 / num_attributes] * num_attributes
    betas = [1.0] * num_attributes

    return alphas, betas, mapped_rankings, idx_to_attribute, idx_to_item, num_attributes

def validate_all_candidates_labeled(sampled_items, group_df, context=""):
    labeled = set(group_df["item"].unique())
    missing = sorted(set(sampled_items) - labeled)
    if missing:
        raise ValueError(
            f"{context} Some sampled items are missing group labels in group_df.\n"
            f"Missing count={len(missing)} example={missing[:30]}"
        )

# =============================================================================
# Sampling Logic
# =============================================================================

def generate_sample_sets(n_samples, n_users, n_items, all_items, preferences):
    all_users = list(preferences["User_ID"].unique())
    dfs, items, users = [], [], []
    all_items = np.array(list(all_items))

    for seed in range(n_samples):
        user_idx = np.random.choice(len(all_users), size=n_users, replace=False)
        item_idx = np.random.choice(all_items, size=n_items, replace=False)

        sampled_pref = preferences[preferences["User_ID"].isin([all_users[i] for i in user_idx])].copy()

        # Filter to sampled items
        item_set = set(item_idx.tolist())
        sampled_pref["Ranked_Items"] = sampled_pref["Ranked_Items"].apply(
            lambda x: [item for item in x if item in item_set]
        )

        dfs.append(sampled_pref)
        items.append(item_idx)
        users.append(user_idx)

    return dfs, items, users


VANILLA_METHODS = {
    "CombMIN": comb_min, "CombMAX": comb_max, "CombSUM": comb_sum,
    "CombANZ": comb_anz, "CombMNZ": comb_mnz,
    "MC1": mc1, "MC2": mc2, "MC3": mc3, "MC4": mc4,
    "BordaCount": borda_count, "Dowdall": dowdall,
    "Median": median_rank, "Mean": mean_rank,
    "RRF": rrf, "iRANK": irank, "ER": er,
    "PostNDCG": postndcg, "CG": cg, "DIBRA": dibra
}

FAIR_METHODS = {
    "KuhlmanConsensus": Consensus,
    "FairMedian": FairILP,
}

# =============================================================================
# Worker helpers (top-level for multiprocessing)
# =============================================================================

def _run_vanilla_method(method_name: str, rankings, sampled_items):
    try:
        method = VANILLA_METHODS[method_name]
        result = method(rankings, sampled_items)
        return method_name, result
    except Exception as e:
        raise RuntimeError(f"[VANILLA method {method_name}] failed") from e

def _run_fair_method(method_name: str, alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes):
    try:
        method = FAIR_METHODS[method_name]
        result = method(alphas, betas, ranks_for_fairness, attributes_map, num_attributes)

        # map back to original item ids (raise a helpful error if something is out of range)
        mapped_back = []
        for i in result:
            if i not in idx_to_item:
                raise KeyError(f"Fair method returned index {i} not in idx_to_item keys "
                               f"(expected 0..{max(idx_to_item.keys())}).")
            mapped_back.append(idx_to_item[i])

        return method_name, mapped_back
    except Exception as e:
        raise RuntimeError(f"[FAIR method {method_name}] failed") from e

# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="recommendations.csv")
    parser.add_argument("--outdir", "-o", default="consensus_results")
    parser.add_argument("--group-file", "-g", default="data/ml-1m/item_groups.pkl")
    parser.add_argument("--user-sample-size", "-us", type=int, default=10)
    parser.add_argument("--item-sample-size", "-is", type=int, default=20)
    parser.add_argument("--n-samples", "-n", type=int, default=1)
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                        help="Number of parallel worker processes")
    parser.add_argument("--complete-rankings", action="store_true",
                        help="Force each user ranking to be a full permutation of the sampled candidates "
                             "by appending missing items at the end (recommended for Kemeny/ILP assumptions).")
    args = parser.parse_args()

    print("=" * 60)
    print("Rank Aggregation (Parallel, Safe Fair Mapping)")
    print("=" * 60)
    print(args)

    os.makedirs(args.outdir, exist_ok=True)

    rankings_df, all_items = load_rankings_to_df(args.input)
    group_df = pickle.load(open(args.group_file, "rb"))

    sampled_rankings_dfs, sampled_items, sampled_users = generate_sample_sets(
        n_samples=args.n_samples,
        n_users=args.user_sample_size,
        n_items=args.item_sample_size,
        all_items=list(all_items),
        preferences=rankings_df,
    )
    formatted_sampled_rankings = [format_sampled_rankings(df) for df in sampled_rankings_dfs]

    for seed in range(args.n_samples):
        write_dir = os.path.join(args.outdir, f"sample_{seed}")
        os.makedirs(write_dir, exist_ok=True)

        # Save sampled artifacts
        pickle.dump(sampled_rankings_dfs[seed], open(os.path.join(write_dir, "sampled_rankings.pkl"), "wb"))
        pickle.dump(sampled_items[seed], open(os.path.join(write_dir, "sampled_items.pkl"), "wb"))
        pickle.dump(sampled_users[seed], open(os.path.join(write_dir, "sampled_users.pkl"), "wb"))

        total = len(VANILLA_METHODS) + len(FAIR_METHODS)
        print(f"\n[seed={seed}] Processing {total} methods with jobs={args.jobs}...")

        # Optionally complete rankings (so each user has full permutation of sampled items)
        rankings_seed = formatted_sampled_rankings[seed]
        sampled_items_seed = list(sampled_items[seed])

        # THROW ERROR if partial ranking
        validate_full_rankings(
            rankings_seed,
            sampled_items_seed,
            context=f"[seed={seed}]"
        )
        validate_all_candidates_labeled(sampled_items_seed, group_df, context=f"[seed={seed}]")
        
        lens = [len(r) for r in rankings_seed]
        print(f"[seed={seed}]", "min/mean/max lens:", min(lens), sum(lens)/len(lens), max(lens))
        print("expected:", len(sampled_items_seed))
        assert min(lens) ==  sum(lens)/len(lens) == max(lens) == len(sampled_items_seed)
        
        # Precompute fairness inputs once per seed using SAFE mapping
        alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes = process_for_fair_ranking_safe(
            sampled_items_seed, group_df, rankings_seed, complete=True  # always complete in idx-space
        )

        # Save effective candidates used for FAIR methods (post drop of missing labels)
        effective_items = [idx_to_item[i] for i in range(len(idx_to_item))]
        pickle.dump(effective_items, open(os.path.join(write_dir, "sampled_items_effective.pkl"), "wb"))

        futures = []
        results_vanilla = {}
        results_fair = {}

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            # vanilla methods: use the (optionally completed) rankings + original sampled item ids
            for name in VANILLA_METHODS.keys():
                futures.append(ex.submit(_run_vanilla_method, name, rankings_seed, sampled_items_seed))

            # fair methods: use the safe, relabeled universe
            for name in FAIR_METHODS.keys():
                futures.append(ex.submit(
                    _run_fair_method, name,
                    alphas, betas, ranks_for_fairness, attributes_map, idx_to_item, num_attributes
                ))

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"seed {seed}"):
                name, result = fut.result()  # will raise with method name context

                if name in VANILLA_METHODS:
                    results_vanilla[name] = result
                else:
                    results_fair[name] = result

        # Write vanilla outputs
        for name, result in results_vanilla.items():
            file_path = os.path.join(write_dir, f"{name}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# Method: {name}\n")
                f.write(f"# Rank ItemID Score\n")
                for rank, (item, score) in enumerate(result, 1):
                    f.write(f"{rank} {item} {score:.6f}\n")

        # Write fair outputs
        for name, ranking in results_fair.items():
            file_path = os.path.join(write_dir, f"{name}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# Method: {name}\n")
                for rank, item in enumerate(ranking, 1):
                    f.write(f"{rank} {item}\n")

        print(f"[seed={seed}] Done. Outputs in: {write_dir}")

    print("\n" + "=" * 60)
    print("Aggregation Complete!")
    print(f"All files are located in: {args.outdir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
