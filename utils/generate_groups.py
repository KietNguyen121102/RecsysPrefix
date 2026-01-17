import pandas as pd 
import ipdb 
import math 
import pickle 

print("--- STARTING SURPRISE PIPELINE ---")
    
# 1. Load Data
file_path = 'data/ml-1m/ratings.dat'

ipdb.set_trace() 
# Define the format: UserID :: MovieID :: Rating :: Timestamp
df = pd.read_csv(file_path, sep='::', engine='python', names=['user', 'item', 'rating', 'timestamp'])
item_grouped = df.groupby('item').count().reset_index().rename(columns={'user': 'count_of_users'})
item_grouped['log_smoothed'] = item_grouped['count_of_users'].apply(lambda x: math.log(x, 5))
item_grouped['binned'] = pd.cut(item_grouped['log_smoothed'], 5, retbins=False, labels=list(range(5)))
print(item_grouped.groupby('binned').count())
pickle.dump(item_grouped, open('data/ml-1m/item_groups.pkl', 'wb'))
