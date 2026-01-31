import pandas as pd 
import ipdb 
import math 
import pickle 
import numpy as np
import yaml 

def generate_item_groups(dataset): 
    # 1. Load Data
    if dataset == 'ml1m': 
        file_path = '/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/ratings.dat'
        
        # Define the format: UserID :: MovieID :: Rating :: Timestamp
        df = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
        item_grouped = df.groupby('item').count().reset_index().rename(columns={'user': 'count_of_users'})
        item_grouped['log_smoothed'] = item_grouped['count_of_users'].apply(lambda x: math.log(x, 5))
        item_grouped['binned'] = pd.cut(item_grouped['log_smoothed'], 5, retbins=False, labels=list(range(5)))
        print(item_grouped.groupby('binned').count())
        pickle.dump(item_grouped, open('/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/item_groups.pkl', 'wb'))
    
    if dataset == 'goodreads': 
        file_path = '/u/rsalgani/2024-2025/RecsysPrefix/data/goodreads/goodreads_sample.csv'
        ipdb.set_trace() 
        # Define the format: UserID :: MovieID :: Rating :: Timestamp
        df = pd.read_csv(file_path, sep=',', engine='python')
        item_grouped = df.groupby('book_id').count().reset_index().rename(columns={'user_id': 'count_of_users'})
        item_grouped['log_smoothed'] = item_grouped['count_of_users'].apply(lambda x: math.log(x, 5))
        item_grouped['binned'] = pd.cut(item_grouped['log_smoothed'], 5, retbins=False, labels=list(range(5)))
        print(item_grouped.groupby('binned').count())
        print(item_grouped)
        pickle.dump(item_grouped, open('/u/rsalgani/2024-2025/RecsysPrefix/data/goodreads/item_groups.pkl', 'wb'))


# generate_item_groups(dataset='goodreads') 

def user_bin_distribution(user_grouped: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Turns item_grps list-of-bins into:
      - counts_0..counts_4
      - p_0..p_4 (normalized)
    """
    def counts_from_list(lst):
        c = np.bincount(np.asarray(lst, dtype=int), minlength=n_bins)
        return c

    counts = np.vstack(user_grouped["item_grps"].apply(counts_from_list).to_numpy())
    counts_df = pd.DataFrame(counts, columns=[f"count_{i}" for i in range(n_bins)])

    # proportions
    row_sums = counts_df.sum(axis=1).replace(0, np.nan)
    p_df = counts_df.div(row_sums, axis=0).fillna(0.0)
    p_df.columns = [f"p_{i}" for i in range(n_bins)]

    return pd.concat([user_grouped[["user"]].reset_index(drop=True), counts_df, p_df], axis=1)


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy. Higher => more heterogeneous."""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def generate_user_groups(): 
    file_path = '/u/rsalgani/2024-2025/RecsysPrefix/data/ml-1m/ratings.dat'
    ipdb.set_trace() 
    
    # define the format: UserID :: MovieID :: Rating :: Timestamp
    df = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
    df = df[df.rating >=4] #Isolate items they actually liked 
    
    # popularity bins for items
    item_grouped = df.groupby('item').count().reset_index().rename(columns={'user': 'count_of_users'})
    item_grouped['log_smoothed'] = item_grouped['count_of_users'].apply(lambda x: math.log(x, 5))
    item_grouped['binned'] = pd.cut(item_grouped['log_smoothed'], 5, retbins=False, labels=list(range(5)))
    df_w_bins = df.merge(item_grouped, on='item')
    
    # collect user taste profiles 
    user_grouped = df_w_bins.groupby('user')['binned'].apply(lambda x: list(x)).reset_index().rename(columns={'binned':'item_grps'})
    ipdb.set_trace() 
    n_item_bins = len(item_grouped['binned'].unique().tolist()) 
    user_feats = user_bin_distribution(user_grouped, n_bins=n_item_bins)
    p_cols = [f"p_{i}" for i in range(n_item_bins)]
    P = user_feats[p_cols].to_numpy()

    # diversity metrics
    user_feats["entropy"] = [entropy_from_probs(p) for p in P]   
    user_feats["entropy_bin"] = pd.cut(
    user_feats["entropy"], bins = 3,
    labels=[0, 1, 2]
    )
    pickle.dump(item_grouped, open('data/ml-1m/user_groups.pkl', 'wb'))
    # ipdb.set_trace() 
    # print(user_feats)
# generate_user_groups() 