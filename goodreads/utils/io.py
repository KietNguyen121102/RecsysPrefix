import pickle 

# =============================================================================
# Data Loading
# =============================================================================

def load_consensus_ranking(file_path):
    """
    Loads a consensus ranking file (rank item score).
    Returns a list of item IDs in ranking order.
    """
    rank_map = {}
    item_id_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Format: Rank ItemID Score
                    # We only care about the ItemID and its Rank (order)
                    rank = int(parts[0])
                    item_id = parts[1]
                    item_id_list.append(int(item_id)) 
                    rank_map[item_id] = rank
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
        
    return item_id_list

def load_sampled_preferences(file_path):
    """
    Loads sampled user preferences from a pickle file.
    Expects DataFrame with columns: User_ID, Ranked_Items (list)
    """
    preferences = pickle.load(open(file_path, 'rb'))
    return preferences

