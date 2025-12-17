"""
Evaluate Rank Aggregation Methods using Kendall Tau Distance

Measures how well each aggregated ranking represents the consensus
by computing Kendall tau distance against each user's ranking list.

Kendall tau distance = number of pairwise disagreements between two rankings.
"""

import numpy as np
from collections import defaultdict
import argparse
from scipy.stats import kendalltau
import time


def load_user_rankings(filepath="recommendations.txt"):
    """Load user rankings from file."""
    rankings = []
    
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                user_id = int(parts[0])
                movie_ids = [int(m) for m in parts[1:]]
                rankings.append((user_id, movie_ids))
    
    print(f"Loaded {len(rankings)} user rankings")
    return rankings


def load_aggregated_rankings(filepath="aggregated_ranking.txt"):
    """Load aggregated rankings from file (all methods in one file)."""
    aggregated = {}
    current_method = None
    current_ranking = []
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("# ") and not line.startswith("# Rank"):
                # Save previous method if exists
                if current_method and current_ranking:
                    aggregated[current_method] = current_ranking
                
                # Start new method
                current_method = line[2:]  # Remove "# "
                current_ranking = []
            
            elif line and not line.startswith("#"):
                # Parse ranking line: rank item_id score
                parts = line.split()
                if len(parts) >= 2:
                    item_id = int(parts[1])
                    current_ranking.append(item_id)
        
        # Save last method
        if current_method and current_ranking:
            aggregated[current_method] = current_ranking
    
    print(f"Loaded {len(aggregated)} aggregation methods")
    return aggregated


def kendall_tau_distance(ranking1, ranking2):
    """
    Compute Kendall tau distance between two rankings.
    
    This counts the number of pairwise disagreements.
    Only considers items that appear in both rankings.
    
    Returns:
        distance: Number of discordant pairs
        normalized_distance: Distance normalized by max possible (n*(n-1)/2)
        tau: Kendall tau correlation coefficient (-1 to 1)
    """
    # Find common items
    set1 = set(ranking1)
    set2 = set(ranking2)
    common = set1 & set2
    
    if len(common) < 2:
        return 0, 0.0, 1.0  # No pairs to compare
    
    # Create position maps for common items only
    pos1 = {item: i for i, item in enumerate(ranking1) if item in common}
    pos2 = {item: i for i, item in enumerate(ranking2) if item in common}
    
    # Get ordered list of common items (by ranking1's order)
    common_items = [item for item in ranking1 if item in common]
    n = len(common_items)
    
    # Count discordant pairs
    discordant = 0
    concordant = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            item_i = common_items[i]
            item_j = common_items[j]
            
            # In ranking1, item_i comes before item_j (by construction)
            # Check if same order in ranking2
            if pos2[item_i] < pos2[item_j]:
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = n * (n - 1) / 2
    normalized = discordant / total_pairs if total_pairs > 0 else 0
    tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 1.0
    
    return discordant, normalized, tau


def kendall_tau_distance_fast(ranking1, ranking2, max_items=500):
    """
    Fast approximation using top-k items only.
    Uses scipy's kendalltau for efficiency.
    """
    # Use only top items for efficiency
    top1 = ranking1[:max_items]
    top2 = ranking2[:max_items]
    
    # Find common items
    common = set(top1) & set(top2)
    
    if len(common) < 2:
        return 0, 0.0, 1.0
    
    # Create rank vectors for common items
    pos1 = {item: i for i, item in enumerate(top1) if item in common}
    pos2 = {item: i for i, item in enumerate(top2) if item in common}
    
    common_list = list(common)
    ranks1 = [pos1[item] for item in common_list]
    ranks2 = [pos2[item] for item in common_list]
    
    # Use scipy's kendalltau (returns tau and p-value)
    tau, _ = kendalltau(ranks1, ranks2)
    
    # Convert tau to distance
    n = len(common)
    max_pairs = n * (n - 1) / 2
    
    # tau = (concordant - discordant) / total_pairs
    # So discordant = (1 - tau) * total_pairs / 2
    normalized_distance = (1 - tau) / 2
    distance = int(normalized_distance * max_pairs)
    
    return distance, normalized_distance, tau


def evaluate_method(method_name, aggregated_ranking, user_rankings, max_items=500, sample_users=None):
    """
    Evaluate a single aggregation method against all user rankings.
    
    Returns statistics about Kendall tau distances.
    """
    distances = []
    normalized_distances = []
    taus = []
    
    # Optionally sample users for faster evaluation
    if sample_users and len(user_rankings) > sample_users:
        indices = np.random.choice(len(user_rankings), sample_users, replace=False)
        eval_rankings = [user_rankings[i] for i in indices]
    else:
        eval_rankings = user_rankings
    
    for user_id, user_ranking in eval_rankings:
        dist, norm_dist, tau = kendall_tau_distance_fast(
            aggregated_ranking, user_ranking, max_items
        )
        distances.append(dist)
        normalized_distances.append(norm_dist)
        taus.append(tau)
    
    return {
        'method': method_name,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'mean_normalized': np.mean(normalized_distances),
        'std_normalized': np.std(normalized_distances),
        'mean_tau': np.mean(taus),
        'std_tau': np.std(taus),
        'n_users': len(eval_rankings)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Rank Aggregation Methods')
    parser.add_argument('--rankings', '-r', default='recommendations.txt',
                        help='User rankings file')
    parser.add_argument('--aggregated', '-a', default='aggregated_ranking.txt',
                        help='Aggregated rankings file')
    parser.add_argument('--output', '-o', default='evaluation_results.txt',
                        help='Output file for results')
    parser.add_argument('--max-items', '-m', type=int, default=500,
                        help='Max items to consider per ranking (for efficiency)')
    parser.add_argument('--sample-users', '-s', type=int, default=None,
                        help='Sample N users for faster evaluation (default: all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Evaluating Rank Aggregation Methods - Kendall Tau Distance")
    print("=" * 70)
    
    # Load data
    user_rankings = load_user_rankings(args.rankings)
    aggregated = load_aggregated_rankings(args.aggregated)
    
    if args.sample_users:
        print(f"Sampling {args.sample_users} users for evaluation")
    print(f"Using top {args.max_items} items per ranking")
    
    # Evaluate each method
    results = []
    print(f"\nEvaluating {len(aggregated)} methods...")
    
    for i, (method_name, agg_ranking) in enumerate(aggregated.items(), 1):
        print(f"  [{i:2d}/{len(aggregated)}] {method_name}...", end=" ", flush=True)
        start = time.time()
        
        stats = evaluate_method(
            method_name, agg_ranking, user_rankings,
            max_items=args.max_items,
            sample_users=args.sample_users
        )
        results.append(stats)
        
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s)")
    
    # Sort results by mean Kendall tau (higher is better = more agreement)
    results.sort(key=lambda x: -x['mean_tau'])
    
    # Save results
    with open(args.output, 'w') as f:
        f.write("Rank Aggregation Evaluation - Kendall Tau Distance\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Users evaluated: {results[0]['n_users']}\n")
        f.write(f"Max items per ranking: {args.max_items}\n\n")
        
        f.write("Results (sorted by Mean Tau, higher = better agreement):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<5} {'Method':<15} {'Mean Tau':<12} {'Std Tau':<12} {'Mean Norm Dist':<15} {'Std Norm Dist':<15}\n")
        f.write("-" * 80 + "\n")
        
        for rank, stats in enumerate(results, 1):
            f.write(f"{rank:<5} {stats['method']:<15} {stats['mean_tau']:<12.4f} {stats['std_tau']:<12.4f} "
                    f"{stats['mean_normalized']:<15.4f} {stats['std_normalized']:<15.4f}\n")
        
        f.write("\n\nDetailed Statistics:\n")
        f.write("-" * 80 + "\n")
        
        for stats in results:
            f.write(f"\n{stats['method']}:\n")
            f.write(f"  Kendall Tau:        mean={stats['mean_tau']:.4f}, std={stats['std_tau']:.4f}\n")
            f.write(f"  Normalized Distance: mean={stats['mean_normalized']:.4f}, std={stats['std_normalized']:.4f}\n")
            f.write(f"  Raw Distance:        mean={stats['mean_distance']:.1f}, std={stats['std_distance']:.1f}, "
                    f"min={stats['min_distance']:.0f}, max={stats['max_distance']:.0f}\n")
    
    print(f"\nResults saved to '{args.output}'")
    
    # Display summary
    print("\n" + "=" * 70)
    print("Results Summary (sorted by Mean Kendall Tau, higher = better)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Method':<15} {'Mean Tau':<12} {'Mean Norm Dist':<15}")
    print("-" * 50)
    
    for rank, stats in enumerate(results, 1):
        print(f"{rank:<5} {stats['method']:<15} {stats['mean_tau']:<12.4f} {stats['mean_normalized']:<15.4f}")
    
    print("\n" + "-" * 50)
    print("Interpretation:")
    print("  Tau = 1.0:  Perfect agreement")
    print("  Tau = 0.0:  Random/no correlation")
    print("  Tau = -1.0: Perfect disagreement")
    print("  Norm Dist: 0 = identical, 1 = completely reversed")


if __name__ == "__main__":
    main()

