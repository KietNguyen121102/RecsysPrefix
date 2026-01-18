import pandas as pd
from surprise import Dataset, Reader, SVD, dump
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
import numpy as np
import csv
import ipdb 

def get_dataframe_from_predictions(predictions):
    # A list of dictionaries to store prediction details
    preds_dict = [{
        'User_ID': pred.uid,
        'Book_ID': pred.iid,
        'actual_rating': pred.r_ui,
        'Estimated_Rating': pred.est
    } for pred in predictions]

    # Convert the list of dictionaries to a pandas DataFrame
    df_predictions = pd.DataFrame(preds_dict)
    return df_predictions

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
    
    # 1. Map predictions to each user
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    recalls = []
    ndcgs = []

    for uid, user_ratings in user_est_true.items():
        # --- PREPARATION ---
        # Sort user ratings by estimated value (descending) to get the "Ranking"
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Get the top k recommendations
        top_k = user_ratings[:k]

        # --- RECALL@K CALCULATION ---
        # Relevant items = items strictly better than the threshold in the user's history
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        
        # Relevant items IN THE TOP K
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in top_k)

        
        if n_rel == 0:
            recalls.append(0)
        else:
            recalls.append(n_rel_and_rec_k / n_rel)

        # --- NDCG@K CALCULATION ---
        # 1. Calculate DCG (Discounted Cumulative Gain)
        dcg = 0
        for i, (_, true_r) in enumerate(top_k):
            if true_r >= threshold:
                # We use binary relevance: 1 if relevant, 0 otherwise
                dcg += 1 / np.log2(i + 2) # i+2 because rank i starts at 0

        # 2. Calculate IDCG (Ideal DCG)
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

    # Return the average across all users
    return sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)

def export_recommendations(model_path, output_file='recommendations_goodreads.csv', top_n=10):
    """
    Generates Top-N recommendations for EVERY user and writes them to a CSV file line-by-line.
    """
    print(f"Loading model from {model_path}...")
    _, algo = dump.load(model_path)
    trainset = algo.trainset
    
    # 1. Prepare internal lists to speed up the loop
    all_item_inner_ids = list(trainset.all_items())
    
    print(f"Starting export to {output_file}...")
    
    # 2. Open the file in write mode
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write Header - Using Movie_ID for compatibility with downstream scripts
        writer.writerow(['User_ID', 'Movie_ID', 'Estimated_Rating'])
        
        # 3. Iterate over every user in the training set
        for i, u_inner_id in enumerate(trainset.all_users()):
            
            # Progress Logger
            if i % 500 == 0:
                print(f"  Processed {i}/{trainset.n_users} users...")
            
            # Get the Raw User ID (for the file)
            u_raw_id = trainset.to_raw_uid(u_inner_id)
            
            # Identify items user has already rated (to exclude them)
            user_rated_inner_ids = set(item_idx for (item_idx, _) in trainset.ur[u_inner_id])

            
            user_predictions = []
            
            # 4. Score all items for this user
            for i_inner_id in all_item_inner_ids:
                if i_inner_id not in user_rated_inner_ids:
                    # Estimate rating (fast calculation)
                    est_score = algo.estimate(u_inner_id, i_inner_id)
                    user_predictions.append((i_inner_id, est_score))
            
            # 5. Pick Top-N
            user_predictions.sort(key=lambda x: x[1], reverse=True)
            top_recs = user_predictions
            
            # 6. Write to file immediately
            for i_inner_id, score in top_recs:
                i_raw_id = trainset.to_raw_iid(i_inner_id)
                writer.writerow([u_raw_id, i_raw_id, round(score, 4)])

    print("--- Export Complete ---")

def surprise_pipeline():
    print("=" * 60)
    print("STARTING SURPRISE PIPELINE (GOODREADS)")
    print("=" * 60)
    
    # 1. Load Data from Goodreads CSV
    file_path = 'goodreads_interactions.csv'
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset size: {len(df)} interactions")
    
    # Filter to only rated items (rating > 0) and read books
    df = df[(df['rating'] > 0) & (df['is_read'] == 1)]
    
    # Keep only necessary columns
    df = df[['user_id', 'book_id', 'rating']]
    
    print(f"After filtering (rating > 0, is_read = 1): {len(df)} interactions")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique books: {df['book_id'].nunique()}")
    
    # ========== SAMPLE 1000 RANDOM USERS ==========
    N_USERS = 1000
    RANDOM_SEED = 42
    
    all_users = df['user_id'].unique()
    np.random.seed(RANDOM_SEED)
    sampled_users = np.random.choice(all_users, size=min(N_USERS, len(all_users)), replace=False)
    
    df = df[df['user_id'].isin(sampled_users)]
    
    print(f"\n--- Sampled {N_USERS} random users ---")
    print(f"Sampled interactions: {len(df)}")
    print(f"Sampled users: {df['user_id'].nunique()}")
    print(f"Unique books in sample: {df['book_id'].nunique()}")
    
    # Create Surprise dataset from DataFrame
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)

    # 2. Train/Test Split
    trainset, testset = train_test_split(data, test_size=0.20, random_state=42)
    print(f"Data split. Train size: {trainset.n_ratings}, Test size: {len(testset)}")

    # 3. Model Definition (SVD)
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)

    # 4. Training
    print("Training the model...")
    algo.fit(trainset)

    dump.dump('./model_goodreads', algo=algo)
    print("Model saved to ./model_goodreads")

    # 5. Prediction & Evaluation
    print("Evaluating on Test Set...")
    predictions = algo.test(testset)
    
    # Compute recall and ndcg
    k_val = 10
    rel_threshold = 4.0  # Items rated 4 or 5 are considered "Relevant"

    mean_recall, mean_ndcg = calculate_ranking_metrics(predictions, k=k_val, threshold=rel_threshold)

    print(f"\n--- Evaluation Results (Top-{k_val}) ---")
    print(f"Recall@{k_val}: {mean_recall:.4f}")
    print(f"NDCG@{k_val}:   {mean_ndcg:.4f}")

    mean_recall, mean_ndcg = calculate_ranking_metrics(predictions, k=20, threshold=rel_threshold)
    print(f"Recall@20: {mean_recall:.4f}")
    print(f"NDCG@20:   {mean_ndcg:.4f}")

    # Compute RMSE
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    print(f"Final RMSE: {rmse:.4f}")
    print("=" * 60)
    print("SURPRISE PIPELINE COMPLETE")
    print("=" * 60)


surprise_pipeline()
export_recommendations('./model_goodreads', output_file='recommendations_goodreads.csv')

