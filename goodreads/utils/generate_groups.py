"""
Goodreads Item Group Generator
==============================
Creates popularity-based item groups for fair ranking evaluation.

This script bins books into groups based on their popularity (number of ratings).
The binning uses log-scaled counts to create more balanced group sizes.

Expected input format (ratings.csv):
    user_id,book_id,rating
    1,123,4
    ...

Output: item_groups.pkl with columns [item, count_of_users, log_smoothed, binned]
"""

import pandas as pd
import math
import pickle
import argparse
import os


def generate_item_groups(ratings_path, output_path, n_bins=5, log_base=5):
    """
    Generate item popularity groups from ratings data.
    
    Args:
        ratings_path: Path to ratings CSV file
        output_path: Path to save item_groups.pkl
        n_bins: Number of popularity bins
        log_base: Base for logarithmic scaling
    
    Returns:
        DataFrame with item groups
    """
    print(f"Loading ratings from {ratings_path}...")
    
    # Load data - handle different column naming conventions
    df = pd.read_csv(ratings_path)
    
    # Normalize column names
    col_mapping = {
        'User-ID': 'user_id', 'userId': 'user_id', 'user': 'user_id',
        'Book-ID': 'book_id', 'bookId': 'book_id', 'book': 'book_id', 
        'item_id': 'book_id', 'item': 'book_id',
        'Rating': 'rating', 'book_rating': 'rating', 'score': 'rating'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Ensure we have required columns
    if 'book_id' not in df.columns:
        # Try to find a column that could be book_id
        for col in df.columns:
            if 'book' in col.lower() or 'item' in col.lower():
                df = df.rename(columns={col: 'book_id'})
                break
    
    print(f"Loaded {len(df)} ratings for {df['book_id'].nunique()} books")
    
    # Count ratings per item
    item_grouped = df.groupby('book_id').size().reset_index(name='count_of_users')
    item_grouped = item_grouped.rename(columns={'book_id': 'item'})
    
    # Log-scale the counts for better binning
    item_grouped['log_smoothed'] = item_grouped['count_of_users'].apply(
        lambda x: math.log(max(x, 1), log_base)
    )
    
    # Bin items into groups
    item_grouped['binned'] = pd.cut(
        item_grouped['log_smoothed'], 
        n_bins, 
        retbins=False, 
        labels=list(range(n_bins))
    )
    
    # Print group statistics
    print("\nGroup distribution:")
    print(item_grouped.groupby('binned')['item'].count())
    
    # Save to pickle
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    pickle.dump(item_grouped, open(output_path, 'wb'))
    print(f"\nSaved item groups to {output_path}")
    
    return item_grouped


def main():
    parser = argparse.ArgumentParser(description="Generate Goodreads Item Groups")
    parser.add_argument('--ratings', '-r', default='data/goodreads/ratings.csv',
                        help='Path to ratings CSV file')
    parser.add_argument('--output', '-o', default='data/goodreads/item_groups.pkl',
                        help='Path to save item groups pickle')
    parser.add_argument('--n-bins', '-n', type=int, default=5,
                        help='Number of popularity bins')
    parser.add_argument('--log-base', '-b', type=float, default=5,
                        help='Base for logarithmic scaling')
    args = parser.parse_args()
    
    generate_item_groups(
        ratings_path=args.ratings,
        output_path=args.output,
        n_bins=args.n_bins,
        log_base=args.log_base
    )


if __name__ == "__main__":
    main()

