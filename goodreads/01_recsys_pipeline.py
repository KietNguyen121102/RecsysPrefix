import pandas as pd
from surprise import Dataset, Reader, SVD, dump
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
import numpy as np
import csv
import argparse 
import os

"""
Goodreads RecSys Pipeline
=========================
This script trains an SVD-based recommender system on Goodreads book ratings
and generates per-user book recommendations (full rankings).

Expected input format (ratings.csv):
    user_id,book_id,rating
    1,123,4
    1,456,5
    ...

Output: recommendations.csv with columns [User_ID, Book_ID, Estimated_Rating]
"""

def get_dataframe_from_predictions(predictions):
    """Convert Surprise predictions to DataFrame."""
    preds_dict = [{
        'User_ID': pred.uid,
        'Book_ID': pred.iid,
        'actual_rating': pred.r_ui,
        'Estimated_Rating': pred.est
    } for pred in predictions]
    return pd.DataFrame(preds_dict)


def calculate_ranking_metrics(predictions, k=10, threshold=4.0):
    """
    Calculates the Recall@k and NDCG@k for a set of Surprise predictions.
    
    Args:
        predictions: List of Prediction objects returned by algo.test()
        k (int): The 'k' in Recall@k and NDCG@k (top k items).
        threshold (float): Ratings above this value are considered 'relevant'.
    
    Returns:
        tuple: (mean_recall, mean_ndcg)
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    recalls = []
    ndcgs = []

    for uid, user_ratings in user_est_true.items():
        # Sort by estimated value (descending)
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        # RECALL@K
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)

        if n_rel == 0:
            recalls.append(0)
        else:
            recalls.append(n_rel_and_rec_k / n_rel)

        # NDCG@K
        dcg = 0
        for i, (_, true_r) in enumerate(top_k):
            if true_r >= threshold:
                dcg += 1 / np.log2(i + 2)

        user_ratings.sort(key=lambda x: x[1], reverse=True)
        ideal_top_k = user_ratings[:k]
        
        idcg = 0
        for i, (_, true_r) in enumerate(ideal_top_k):
            if true_r >= threshold:
                idcg += 1 / np.log2(i + 2)

        if idcg == 0:
            ndcgs.append(0)
        else:
            ndcgs.append(dcg / idcg)

    return sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)


def export_recommendations(model_path, output_file='recommendations.csv', top_n=None):
    """
    Generates Top-N recommendations for EVERY user and writes them to a CSV file.
    If top_n is None, exports full rankings for all items.
    """
    print(f"Loading model from {model_path}...")
    _, algo = dump.load(model_path)
    trainset = algo.trainset
    
    all_item_inner_ids = list(trainset.all_items())
    
    print(f"Starting export to {output_file}...")
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['User_ID', 'Book_ID', 'Estimated_Rating'])
        
        for i, u_inner_id in enumerate(trainset.all_users()):
            if i % 500 == 0:
                print(f"  Processed {i}/{trainset.n_users} users...")
            
            u_raw_id = trainset.to_raw_uid(u_inner_id)
            user_rated_items = {item_idx: rating for (item_idx, rating) in trainset.ur[u_inner_id]}

            user_predictions = []
            
            for i_inner_id in all_item_inner_ids:
                if i_inner_id in user_rated_items:
                    score = user_rated_items[i_inner_id]
                else:
                    score = algo.estimate(u_inner_id, i_inner_id)
                user_predictions.append((i_inner_id, score))
            
            user_predictions.sort(key=lambda x: x[1], reverse=True)
            
            if top_n:
                top_recs = user_predictions[:top_n]
            else:
                top_recs = user_predictions
            
            for i_inner_id, score in top_recs:
                i_raw_id = trainset.to_raw_iid(i_inner_id)
                writer.writerow([u_raw_id, i_raw_id, round(score, 4)])

    print("--- Export Complete ---")


def load_goodreads_data(ratings_path, min_user_ratings=20, min_book_ratings=20, 
                        max_users=None, max_items=None, seed=42):
    """
    Load Goodreads ratings data and apply filtering + subsampling.
    
    Expected CSV format: user_id,book_id,rating
    
    Args:
        ratings_path: Path to ratings CSV file
        min_user_ratings: Minimum ratings per user to keep
        min_book_ratings: Minimum ratings per book to keep
        max_users: Maximum number of users to sample (None = all)
        max_items: Maximum number of items to sample (None = all)
        seed: Random seed for reproducible sampling
    
    Returns:
        Filtered and subsampled DataFrame
    """
    print(f"Loading ratings from {ratings_path}...")
    
    # Try different common Goodreads formats
    try:
        df = pd.read_csv(ratings_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
    
    # Normalize column names (handle different naming conventions)
    col_mapping = {
        'User-ID': 'user_id', 'userId': 'user_id', 'user': 'user_id',
        'Book-ID': 'book_id', 'bookId': 'book_id', 'book': 'book_id', 'item_id': 'book_id',
        'Rating': 'rating', 'book_rating': 'rating', 'score': 'rating'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    required_cols = ['user_id', 'book_id', 'rating']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found columns: {df.columns.tolist()}")
    
    print(f"Initial data: {len(df)} ratings, {df['user_id'].nunique()} users, {df['book_id'].nunique()} books")
    
    # Filter users and books with minimum ratings
    if min_user_ratings > 0:
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['user_id'].isin(valid_users)]
        print(f"After user filter (>={min_user_ratings}): {len(df)} ratings, {df['user_id'].nunique()} users")
    
    if min_book_ratings > 0:
        book_counts = df['book_id'].value_counts()
        valid_books = book_counts[book_counts >= min_book_ratings].index
        df = df[df['book_id'].isin(valid_books)]
        print(f"After book filter (>={min_book_ratings}): {len(df)} ratings, {df['book_id'].nunique()} books")
    
    # Subsample users if max_users specified
    if max_users is not None and df['user_id'].nunique() > max_users:
        np.random.seed(seed)
        all_users = df['user_id'].unique()
        sampled_users = np.random.choice(all_users, size=max_users, replace=False)
        df = df[df['user_id'].isin(sampled_users)]
        print(f"After user sampling (max {max_users}): {len(df)} ratings, {df['user_id'].nunique()} users")
    
    # Subsample items from items that sampled users have rated (not from all items)
    if max_items is not None and df['book_id'].nunique() > max_items:
        np.random.seed(seed + 1)  # Different seed for items
        # Only consider items that exist in current (user-filtered) dataframe
        available_items = df['book_id'].unique()
        sampled_items = np.random.choice(available_items, size=min(max_items, len(available_items)), replace=False)
        df = df[df['book_id'].isin(sampled_items)]
        print(f"After item sampling (max {max_items}): {len(df)} ratings, {df['book_id'].nunique()} books")
    
    # Re-filter after subsampling with relaxed thresholds (at least 1 rating)
    if max_users is not None or max_items is not None:
        # Use relaxed thresholds after subsampling to avoid losing all data
        min_post_sample = 1
        prev_len = 0
        while len(df) != prev_len:
            prev_len = len(df)
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_post_sample].index
            df = df[df['user_id'].isin(valid_users)]
            book_counts = df['book_id'].value_counts()
            valid_books = book_counts[book_counts >= min_post_sample].index
            df = df[df['book_id'].isin(valid_books)]
        print(f"After re-filtering: {len(df)} ratings, {df['user_id'].nunique()} users, {df['book_id'].nunique()} books")
    
    return df


def surprise_pipeline(ratings_path, model_out, min_user_ratings=20, min_book_ratings=20,
                      max_users=None, max_items=None, seed=42):
    """Main training pipeline for Goodreads data."""
    print("=" * 60)
    print("GOODREADS SURPRISE PIPELINE")
    print("=" * 60)
    
    # Load and filter data
    df = load_goodreads_data(ratings_path, min_user_ratings, min_book_ratings,
                             max_users, max_items, seed)
    
    # Determine rating scale
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()
    print(f"Rating scale: {min_rating} to {max_rating}")
    
    # Create Surprise dataset
    reader = Reader(rating_scale=(min_rating, max_rating))
    data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
    
    # Train/Test Split
    trainset, testset = train_test_split(data, test_size=0.20, random_state=42)
    print(f"Data split. Train size: {trainset.n_ratings}, Test size: {len(testset)}")

    # Model Definition (SVD)
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

    # Training
    print("Training the model...")
    algo.fit(trainset)

    # Save model
    os.makedirs(os.path.dirname(model_out) if os.path.dirname(model_out) else '.', exist_ok=True)
    dump.dump(model_out, algo=algo)
    print(f"Model saved to {model_out}")

    # Evaluation
    print("Evaluating on Test Set...")
    predictions = algo.test(testset)
    
    # Ranking metrics
    k_val = 10
    rel_threshold = 4.0

    mean_recall, mean_ndcg = calculate_ranking_metrics(predictions, k=k_val, threshold=rel_threshold)

    print(f"\n--- Evaluation Results (Top-{k_val}) ---")
    print(f"Recall@{k_val}: {mean_recall:.4f}")
    print(f"NDCG@{k_val}:   {mean_ndcg:.4f}")

    mean_recall, mean_ndcg = calculate_ranking_metrics(predictions, k=20, threshold=rel_threshold)
    print(f"Recall@20: {mean_recall:.4f}")
    print(f"NDCG@20:   {mean_ndcg:.4f}")

    # RMSE/MAE
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    print(f"Final RMSE: {rmse:.4f}")
    print(f"Final MAE:  {mae:.4f}")
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Goodreads RecSys Pipeline")
    parser.add_argument("--data-path", type=str, default="data/goodreads/ratings.csv",
                        help="Path to Goodreads ratings CSV")
    parser.add_argument("--model-out", type=str, default="./model",
                        help="Path to save trained model")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save recommendations CSV")
    parser.add_argument("--min-user-ratings", type=int, default=20,
                        help="Minimum ratings per user")
    parser.add_argument("--min-book-ratings", type=int, default=20,
                        help="Minimum ratings per book")
    parser.add_argument("--max-users", type=int, default=None,
                        help="Maximum number of users to sample (None=all)")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Maximum number of items/books to sample (None=all)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Number of recommendations per user (None=all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()
    
    print(args)
    
    surprise_pipeline(
        ratings_path=args.data_path,
        model_out=args.model_out,
        min_user_ratings=args.min_user_ratings,
        min_book_ratings=args.min_book_ratings,
        max_users=args.max_users,
        max_items=args.max_items,
        seed=args.seed
    )
    
    export_recommendations(args.model_out, output_file=args.output_file, top_n=args.top_n)


if __name__ == "__main__":
    main()

