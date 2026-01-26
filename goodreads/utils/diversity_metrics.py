def calculate_pop_group_item_diversity(full_committee, item_groups, k):
    """
    Calculate diversity metrics for a committee based on item popularity groups.
    
    Args:
        full_committee: List of item IDs in the committee
        item_groups: DataFrame with columns ['item', 'binned'] mapping items to groups
        k: Number of top items to consider
    
    Returns:
        coverage_results: Dict mapping group -> coverage ratio
        percentage_results: Dict mapping group -> percentage of top-k
    """
    committee_at_k = full_committee[:k]
    coverage_results, percentage_results = {}, {}
    df = item_groups[item_groups['item'].isin(full_committee)]
    
    for group in item_groups['binned'].unique():
        covered_items = df[(df['binned'] == group) & (df.item.isin(committee_at_k))]['item'].tolist()
        group_items = df[df['binned'] == group]['item'].to_list()
        coverage = len(covered_items) / len(group_items) if group_items else 0
        percentage = len(covered_items) / k
        percentage_results[group] = percentage
        coverage_results[group] = coverage
        
    return coverage_results, percentage_results

