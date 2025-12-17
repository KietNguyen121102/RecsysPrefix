"""
Matrix Factorization for MovieLens 1M Dataset
Using Stochastic Gradient Descent (SGD) with biases.
"""

import numpy as np
from scipy.sparse import csr_matrix
import time


def load_movielens_1m(data_path="ml-1m"):
    """Load MovieLens 1M dataset."""
    print("Loading MovieLens 1M dataset...")
    
    ratings = []
    with open(f"{data_path}/ratings.dat", "r", encoding="latin-1") as f:
        for line in f:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            ratings.append((int(user_id), int(movie_id), float(rating)))
    
    movies = {}
    with open(f"{data_path}/movies.dat", "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0])
            title = parts[1]
            genres = parts[2]
            movies[movie_id] = {"title": title, "genres": genres}
    
    print(f"Loaded {len(ratings):,} ratings for {len(movies):,} movies")
    return ratings, movies


def train_test_split(ratings, test_ratio=0.2, random_seed=42):
    """Split ratings into train and test sets."""
    np.random.seed(random_seed)
    ratings = np.array(ratings)
    n_ratings = len(ratings)
    
    indices = np.random.permutation(n_ratings)
    test_size = int(n_ratings * test_ratio)
    
    train_ratings = [tuple(ratings[i]) for i in indices[test_size:]]
    test_ratings = [tuple(ratings[i]) for i in indices[:test_size]]
    
    print(f"Train/Test Split: {len(train_ratings):,} train, {len(test_ratings):,} test")
    return train_ratings, test_ratings


def create_mappings(all_ratings):
    """Create user/movie ID mappings."""
    user_ids = sorted(set(int(r[0]) for r in all_ratings))
    movie_ids = sorted(set(int(r[1]) for r in all_ratings))
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}
    
    return user_to_idx, movie_to_idx, idx_to_user, idx_to_movie


class MatrixFactorizationSGD:
    """
    Matrix Factorization using SGD with biases.
    
    Predicts: r̂_ui = μ + b_u + b_i + p_u · q_i
    
    This is the standard approach used in production systems.
    """
    
    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, verbose=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr  # Learning rate
        self.reg = reg  # Regularization
        self.verbose = verbose
        
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
        self.P = None  # User factors
        self.Q = None  # Item factors
        
    def fit(self, train_data, test_data=None, n_users=None, n_items=None):
        """
        Train the model using SGD.
        
        Args:
            train_data: List of (user_idx, item_idx, rating) tuples
            test_data: Optional list for evaluation
            n_users: Total number of users
            n_items: Total number of items
        """
        # Determine dimensions
        if n_users is None:
            n_users = max(r[0] for r in train_data) + 1
        if n_items is None:
            n_items = max(r[1] for r in train_data) + 1
        
        # Compute global mean
        self.global_mean = np.mean([r[2] for r in train_data])
        
        # Initialize parameters
        np.random.seed(42)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        if self.verbose:
            print(f"\nTraining Matrix Factorization (SGD)")
            print(f"Global mean: {self.global_mean:.3f}")
            print(f"Factors: {self.n_factors}, LR: {self.lr}, Reg: {self.reg}")
            print("-" * 55)
            header = f"{'Epoch':>6} | {'Train RMSE':>11}"
            if test_data:
                header += f" | {'Test RMSE':>10}"
            header += f" | {'Time':>7}"
            print(header)
            print("-" * 55)
        
        train_data = np.array(train_data)
        
        for epoch in range(self.n_epochs):
            start_time = time.time()
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            # SGD updates
            for user_idx, item_idx, rating in train_data:
                user_idx, item_idx = int(user_idx), int(item_idx)
                
                # Predict
                pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
                pred += self.P[user_idx] @ self.Q[item_idx]
                
                # Error
                error = rating - pred
                
                # Update biases
                self.user_bias[user_idx] += self.lr * (error - self.reg * self.user_bias[user_idx])
                self.item_bias[item_idx] += self.lr * (error - self.reg * self.item_bias[item_idx])
                
                # Update latent factors
                p_u = self.P[user_idx].copy()
                self.P[user_idx] += self.lr * (error * self.Q[item_idx] - self.reg * self.P[user_idx])
                self.Q[item_idx] += self.lr * (error * p_u - self.reg * self.Q[item_idx])
            
            # Calculate RMSE
            train_rmse = self._calculate_rmse(train_data)
            elapsed = time.time() - start_time
            
            log_msg = f"{epoch + 1:>6}/{self.n_epochs} | {train_rmse:>11.4f}"
            
            if test_data:
                test_rmse = self._calculate_rmse(test_data)
                log_msg += f" | {test_rmse:>10.4f}"
            
            log_msg += f" | {elapsed:>6.2f}s"
            
            if self.verbose:
                print(log_msg)
        
        return self
    
    def _calculate_rmse(self, data):
        """Calculate RMSE on data."""
        squared_errors = []
        for user_idx, item_idx, rating in data:
            user_idx, item_idx = int(user_idx), int(item_idx)
            pred = self.predict(user_idx, item_idx)
            squared_errors.append((rating - pred) ** 2)
        return np.sqrt(np.mean(squared_errors))
    
    def predict(self, user_idx, item_idx):
        """Predict rating for a user-item pair."""
        pred = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx]
        pred += self.P[user_idx] @ self.Q[item_idx]
        # Clip to valid rating range
        return np.clip(pred, 1, 5)
    
    def predict_all_for_user(self, user_idx):
        """Predict all item ratings for a user."""
        pred = self.global_mean + self.user_bias[user_idx] + self.item_bias
        pred += self.P[user_idx] @ self.Q.T
        return np.clip(pred, 1, 5)
    
    def get_full_ranking(self, user_idx, exclude_items=None):
        """Get full ranking of all items for a user (sorted by predicted score)."""
        scores = self.predict_all_for_user(user_idx)
        
        if exclude_items is not None:
            scores[list(exclude_items)] = -np.inf
        
        # Return all items sorted by score (descending)
        ranked_items = np.argsort(scores)[::-1]
        
        # Filter out excluded items from the ranking
        if exclude_items is not None:
            ranked_items = [idx for idx in ranked_items if idx not in exclude_items]
        
        return ranked_items


def generate_recommendations(model, train_data, idx_to_user, idx_to_movie,
                             output_file="recommendations.txt"):
    """Generate full ranking over all items for all users."""
    # Get rated items per user
    user_rated = {}
    for user_idx, item_idx, _ in train_data:
        user_idx, item_idx = int(user_idx), int(item_idx)
        if user_idx not in user_rated:
            user_rated[user_idx] = set()
        user_rated[user_idx].add(item_idx)
    
    n_users = len(idx_to_user)
    n_items = len(idx_to_movie)
    print(f"\nGenerating full ranking over {n_items:,} items for {n_users:,} users...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for user_idx in range(n_users):
            user_id = idx_to_user[user_idx]
            exclude = user_rated.get(user_idx, set())
            
            # Get full ranking (all items sorted by predicted score)
            ranked_items = model.get_full_ranking(user_idx, exclude_items=exclude)
            
            # Output: user_id movie_id1 movie_id2 movie_id3 ... (all items ranked)
            movie_ids = [str(idx_to_movie[item_idx]) for item_idx in ranked_items]
            f.write(f"{user_id} {' '.join(movie_ids)}\n")
            
            if (user_idx + 1) % 1000 == 0:
                print(f"  Processed {user_idx + 1:,}/{n_users:,} users...")
    
    print(f"Full rankings saved to '{output_file}'")


def main():
    print("=" * 60)
    print("Matrix Factorization Recommender System (SGD)")
    print("=" * 60)
    
    # Load data
    ratings, movies = load_movielens_1m()
    
    # Split data
    train_ratings, test_ratings = train_test_split(ratings, test_ratio=0.2)
    
    # Create mappings
    user_to_idx, movie_to_idx, idx_to_user, idx_to_movie = create_mappings(ratings)
    
    # Convert to indexed format
    train_data = [(user_to_idx[int(u)], movie_to_idx[int(m)], float(r)) 
                  for u, m, r in train_ratings]
    test_data = [(user_to_idx[int(u)], movie_to_idx[int(m)], float(r)) 
                 for u, m, r in test_ratings]
    
    print(f"Train: {len(train_data):,} ratings, Test: {len(test_data):,} ratings")
    
    # Train model
    model = MatrixFactorizationSGD(
        n_factors=20,      # Latent factors (smaller = less overfitting)
        n_epochs=30,       # Training epochs
        lr=0.005,          # Learning rate
        reg=0.02,          # Regularization strength
        verbose=True
    )
    model.fit(train_data, test_data=test_data, 
              n_users=len(user_to_idx), n_items=len(movie_to_idx))
    
    # Generate full rankings for all users
    generate_recommendations(
        model, train_data, idx_to_user, idx_to_movie,
        output_file="recommendations.txt"
    )
    
    # Show sample (top 10 from full ranking)
    print("\n" + "=" * 60)
    print("Sample Ranking (User 1, top 10):")
    print("=" * 60)
    
    user_idx = user_to_idx[1]
    exclude = set(d[1] for d in train_data if d[0] == user_idx)
    ranked_items = model.get_full_ranking(user_idx, exclude_items=exclude)
    
    movie_ids = [str(idx_to_movie[item_idx]) for item_idx in ranked_items[:10]]
    print(f"User 1: {' '.join(movie_ids)}")
    print(f"(Full ranking contains {len(ranked_items)} items)")


if __name__ == "__main__":
    main()
