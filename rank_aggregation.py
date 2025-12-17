"""
Rank Aggregation Methods for Recommendation Lists

Implements comprehensive rank aggregation methods to combine
individual user recommendation lists into a single consensus ranking.

Methods:
- Comb* family: CombMIN, CombMAX, CombSUM, CombANZ, CombMNZ
- Markov Chain: MC1, MC2, MC3, MC4
- Position-based: BordaCount, Dowdall, Median, Mean, HPA
- Reciprocal: RRF, iRANK, ER
- Advanced: PostNDCG, CG, DIBRA
"""

import numpy as np
from collections import defaultdict
import argparse


def load_recommendations(filepath="recommendations.txt"):
    """
    Load recommendations from file.
    Format: user_id movie_id1 movie_id2 movie_id3 ...
    """
    rankings = []
    all_items = set()
    
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                movie_ids = [int(m) for m in parts[1:]]
                rankings.append(movie_ids)
                all_items.update(movie_ids)
    
    print(f"Loaded {len(rankings)} user rankings")
    print(f"Total unique items: {len(all_items)}")
    
    return rankings, all_items


def normalize_scores(scores):
    """Normalize scores to [0, 1] range."""
    if not scores:
        return scores
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s - min_s == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}


def get_item_scores_from_rankings(rankings, all_items):
    """
    Convert rankings to normalized scores for each item.
    Score = 1 - (rank-1)/(list_length) for items in list, 0 otherwise.
    """
    item_scores = defaultdict(list)  # item -> list of scores from each ranking
    
    for ranking in rankings:
        n = len(ranking)
        items_in_ranking = set(ranking)
        
        for pos, item in enumerate(ranking):
            # Score: higher is better, normalized to [0, 1]
            score = 1.0 - (pos / n) if n > 0 else 0
            item_scores[item].append(score)
        
        # Items not in this ranking get score 0
        for item in all_items:
            if item not in items_in_ranking:
                item_scores[item].append(0.0)
    
    return item_scores


# =============================================================================
# Comb* Methods (Information Retrieval Score Fusion)
# =============================================================================

def comb_min(rankings, all_items):
    """CombMIN: Minimum score across all rankings."""
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {item: min(s_list) for item, s_list in item_scores.items()}
    return sorted(scores.items(), key=lambda x: -x[1])


def comb_max(rankings, all_items):
    """CombMAX: Maximum score across all rankings."""
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {item: max(s_list) for item, s_list in item_scores.items()}
    return sorted(scores.items(), key=lambda x: -x[1])


def comb_sum(rankings, all_items):
    """CombSUM: Sum of scores across all rankings."""
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {item: sum(s_list) for item, s_list in item_scores.items()}
    return sorted(scores.items(), key=lambda x: -x[1])


def comb_anz(rankings, all_items):
    """CombANZ: Average of Non-Zero scores."""
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {}
    for item, s_list in item_scores.items():
        non_zero = [s for s in s_list if s > 0]
        scores[item] = sum(non_zero) / len(non_zero) if non_zero else 0
    return sorted(scores.items(), key=lambda x: -x[1])


def comb_mnz(rankings, all_items):
    """CombMNZ: Sum * count of non-zero scores."""
    item_scores = get_item_scores_from_rankings(rankings, all_items)
    scores = {}
    for item, s_list in item_scores.items():
        non_zero = [s for s in s_list if s > 0]
        scores[item] = sum(non_zero) * len(non_zero)
    return sorted(scores.items(), key=lambda x: -x[1])


# =============================================================================
# Markov Chain Methods (MC1-MC4, Dwork et al. 2001)
# =============================================================================

def build_pairwise_matrix(rankings, all_items, top_k=500):
    """Build pairwise preference matrix from rankings (using top-k items per ranking)."""
    items = list(all_items)
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    
    # pref[i][j] = number of rankings where i beats j
    pref = np.zeros((n, n))
    
    for ranking in rankings:
        # Only consider top-k items
        top_items = ranking[:top_k]
        for i, item_i in enumerate(top_items):
            idx_i = item_to_idx[item_i]
            for j in range(i + 1, len(top_items)):
                item_j = top_items[j]
                idx_j = item_to_idx[item_j]
                pref[idx_i][idx_j] += 1
    
    return pref, items, item_to_idx


def mc1(rankings, all_items, top_k=500, max_iter=100, tol=1e-6):
    """
    MC1: From state i, uniformly transition to any state j that beats i
    in at least one ranking.
    """
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    
    # Transition: uniform over items that beat current item
    transition = np.zeros((n, n))
    for i in range(n):
        winners = [j for j in range(n) if pref[j][i] > 0]
        if winners:
            for j in winners:
                transition[i][j] = 1.0 / len(winners)
        else:
            transition[i] = 1.0 / n  # uniform if no winners
    
    # Power iteration
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol:
            break
        scores = new_scores
    
    result = [(items[i], scores[i]) for i in range(n)]
    return sorted(result, key=lambda x: -x[1])


def mc2(rankings, all_items, top_k=500, max_iter=100, tol=1e-6):
    """
    MC2: From state i, transition to j with probability proportional to
    how many times j beats i.
    """
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    
    transition = np.zeros((n, n))
    for i in range(n):
        total = sum(pref[j][i] for j in range(n))
        if total > 0:
            for j in range(n):
                transition[i][j] = pref[j][i] / total
        else:
            transition[i] = 1.0 / n
    
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol:
            break
        scores = new_scores
    
    result = [(items[i], scores[i]) for i in range(n)]
    return sorted(result, key=lambda x: -x[1])


def mc3(rankings, all_items, top_k=500, damping=0.85, max_iter=100, tol=1e-6):
    """MC3: Ergodic version of MC1 with damping factor."""
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    
    transition = np.zeros((n, n))
    for i in range(n):
        winners = [j for j in range(n) if pref[j][i] > 0]
        if winners:
            for j in winners:
                transition[i][j] = 1.0 / len(winners)
        else:
            transition[i] = 1.0 / n
    
    # Add damping
    transition = damping * transition + (1 - damping) / n
    
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol:
            break
        scores = new_scores
    
    result = [(items[i], scores[i]) for i in range(n)]
    return sorted(result, key=lambda x: -x[1])


def mc4(rankings, all_items, top_k=500, damping=0.85, max_iter=100, tol=1e-6):
    """MC4: Ergodic version of MC2 with damping factor."""
    pref, items, item_to_idx = build_pairwise_matrix(rankings, all_items, top_k)
    n = len(items)
    
    transition = np.zeros((n, n))
    for i in range(n):
        total = sum(pref[j][i] for j in range(n))
        if total > 0:
            for j in range(n):
                transition[i][j] = pref[j][i] / total
        else:
            transition[i] = 1.0 / n
    
    transition = damping * transition + (1 - damping) / n
    
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = transition.T @ scores
        new_scores /= new_scores.sum()
        if np.abs(new_scores - scores).max() < tol:
            break
        scores = new_scores
    
    result = [(items[i], scores[i]) for i in range(n)]
    return sorted(result, key=lambda x: -x[1])


# =============================================================================
# Position-based Methods
# =============================================================================

def borda_count(rankings, all_items):
    """Borda Count: Position-based scoring (k-1 for 1st, k-2 for 2nd, ...)."""
    scores = defaultdict(float)
    
    for ranking in rankings:
        k = len(ranking)
        for pos, item in enumerate(ranking):
            scores[item] += (k - 1 - pos)
    
    return sorted(scores.items(), key=lambda x: -x[1])


def dowdall(rankings, all_items):
    """Dowdall System: Harmonic weights (1, 1/2, 1/3, ...)."""
    scores = defaultdict(float)
    
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            scores[item] += 1.0 / (pos + 1)
    
    return sorted(scores.items(), key=lambda x: -x[1])


def median_rank(rankings, all_items):
    """Median: Aggregate by median rank position."""
    item_ranks = defaultdict(list)
    n_rankings = len(rankings)
    
    for ranking in rankings:
        rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
        max_rank = len(ranking) + 1
        
        for item in all_items:
            item_ranks[item].append(rank_dict.get(item, max_rank))
    
    scores = {}
    for item, ranks in item_ranks.items():
        scores[item] = -np.median(ranks)  # Negative because lower rank is better
    
    return sorted(scores.items(), key=lambda x: -x[1])


def mean_rank(rankings, all_items):
    """Mean: Aggregate by mean rank position."""
    item_ranks = defaultdict(list)
    
    for ranking in rankings:
        rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
        max_rank = len(ranking) + 1
        
        for item in all_items:
            item_ranks[item].append(rank_dict.get(item, max_rank))
    
    scores = {}
    for item, ranks in item_ranks.items():
        scores[item] = -np.mean(ranks)  # Negative because lower rank is better
    
    return sorted(scores.items(), key=lambda x: -x[1])


def hpa(rankings, all_items):
    """HPA (Highest Position Aggregation): Best rank across all lists."""
    item_best_rank = {}
    
    for item in all_items:
        item_best_rank[item] = float('inf')
    
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            if rank < item_best_rank[item]:
                item_best_rank[item] = rank
    
    # Convert to score (lower rank = higher score)
    max_rank = max(item_best_rank.values())
    scores = {item: max_rank - rank + 1 for item, rank in item_best_rank.items()}
    
    return sorted(scores.items(), key=lambda x: -x[1])


# =============================================================================
# Reciprocal/Score-based Methods
# =============================================================================

def rrf(rankings, all_items, k=60):
    """
    RRF (Reciprocal Rank Fusion): Score = sum(1 / (k + rank)).
    Default k=60 as per original paper.
    """
    scores = defaultdict(float)
    
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            scores[item] += 1.0 / (k + rank)
    
    return sorted(scores.items(), key=lambda x: -x[1])


def irank(rankings, all_items):
    """iRANK: Inverse rank sum (1/rank for each appearance)."""
    scores = defaultdict(float)
    
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            scores[item] += 1.0 / rank
    
    return sorted(scores.items(), key=lambda x: -x[1])


def er(rankings, all_items):
    """
    ER (Expected Reciprocal): Based on cascade model.
    Score considers probability of reaching each position.
    """
    scores = defaultdict(float)
    
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            # Probability of reaching this position * reciprocal rank
            # Using simplified cascade model
            prob_reach = 1.0 / rank
            scores[item] += prob_reach / rank
    
    return sorted(scores.items(), key=lambda x: -x[1])


# =============================================================================
# Advanced Methods
# =============================================================================

def postndcg(rankings, all_items):
    """
    PostNDCG: NDCG-style weighting for position.
    Score = sum(1 / log2(rank + 1)) for each appearance.
    """
    scores = defaultdict(float)
    
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            rank = pos + 1
            scores[item] += 1.0 / np.log2(rank + 1)
    
    return sorted(scores.items(), key=lambda x: -x[1])


def cg(rankings, all_items):
    """
    CG (Cumulative Gain): Score based on relevance at position.
    Using position-based relevance: rel = max_pos - pos.
    """
    scores = defaultdict(float)
    
    for ranking in rankings:
        max_pos = len(ranking)
        for pos, item in enumerate(ranking):
            # Relevance decreases with position
            relevance = max_pos - pos
            scores[item] += relevance
    
    return sorted(scores.items(), key=lambda x: -x[1])


def dibra(rankings, all_items):
    """
    DIBRA (Distance-Based Rank Aggregation):
    Minimizes sum of squared rank differences.
    Uses iterative reweighting based on distance to consensus.
    """
    items = list(all_items)
    n_items = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    
    # Initialize with mean rank
    item_ranks = defaultdict(list)
    for ranking in rankings:
        rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
        max_rank = len(ranking) + 1
        for item in all_items:
            item_ranks[item].append(rank_dict.get(item, max_rank))
    
    # Initial scores based on mean rank
    scores = {item: -np.mean(ranks) for item, ranks in item_ranks.items()}
    
    # Iterative refinement (3 iterations)
    for _ in range(3):
        # Get current consensus ranking
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        consensus_rank = {item: pos + 1 for pos, (item, _) in enumerate(sorted_items)}
        
        # Reweight based on distance to consensus
        new_scores = defaultdict(float)
        for ranking in rankings:
            rank_dict = {item: pos + 1 for pos, item in enumerate(ranking)}
            max_rank = len(ranking) + 1
            
            # Calculate weight for this ranking (inverse distance to consensus)
            total_dist = 0
            for item in ranking:
                r1 = rank_dict[item]
                r2 = consensus_rank.get(item, n_items)
                total_dist += (r1 - r2) ** 2
            
            weight = 1.0 / (1 + np.sqrt(total_dist / len(ranking)))
            
            for item in all_items:
                rank = rank_dict.get(item, max_rank)
                new_scores[item] += weight * (max_rank - rank)
        
        scores = dict(new_scores)
    
    return sorted(scores.items(), key=lambda x: -x[1])


# =============================================================================
# Run All Methods
# =============================================================================

ALL_METHODS = {
    'CombMIN': comb_min,
    'CombMAX': comb_max,
    'CombSUM': comb_sum,
    'CombANZ': comb_anz,
    'CombMNZ': comb_mnz,
    'MC1': mc1,
    'MC2': mc2,
    'MC3': mc3,
    'MC4': mc4,
    'BordaCount': borda_count,
    'Dowdall': dowdall,
    'Median': median_rank,
    'RRF': rrf,
    'iRANK': irank,
    'Mean': mean_rank,
    'HPA': hpa,
    'PostNDCG': postndcg,
    'ER': er,
    'CG': cg,
    'DIBRA': dibra,
}


def run_all_methods(rankings, all_items):
    """Run all aggregation methods and return results."""
    results = {}
    total = len(ALL_METHODS)
    
    print("\nRunning all aggregation methods...")
    
    for i, (name, method) in enumerate(ALL_METHODS.items(), 1):
        print(f"  [{i:2d}/{total}] {name}...")
        results[name] = method(rankings, all_items)
    
    print("Done!")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rank Aggregation for Recommendations')
    parser.add_argument('--input', '-i', default='recommendations.txt',
                        help='Input recommendations file')
    parser.add_argument('--output', '-o', default='aggregated_ranking.txt',
                        help='Output file for aggregated rankings')
    parser.add_argument('--top-k', '-k', type=int, default=100,
                        help='Number of top items to output per method')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Rank Aggregation - All Methods")
    print("=" * 60)
    
    # Load data
    rankings, all_items = load_recommendations(args.input)
    
    # Run all methods
    all_results = run_all_methods(rankings, all_items)
    
    # Save all results to single file
    with open(args.output, 'w') as f:
        for method_name in ALL_METHODS.keys():
            result = all_results[method_name]
            
            f.write(f"# {method_name}\n")
            f.write(f"# Rank ItemID Score\n")
            
            for rank, (item, score) in enumerate(result[:args.top_k], 1):
                f.write(f"{rank} {item} {score:.6f}\n")
            
            f.write("\n")
    
    print(f"\nAll results saved to '{args.output}'")
    
    # Display summary
    print("\n" + "=" * 70)
    print("Summary - Top 10 Items per Method")
    print("=" * 70)
    
    for method_name in ALL_METHODS.keys():
        result = all_results[method_name]
        print(f"\n{method_name}:")
        top_items = [str(item) for item, _ in result[:10]]
        print(f"  {' '.join(top_items)}")


if __name__ == "__main__":
    main()
