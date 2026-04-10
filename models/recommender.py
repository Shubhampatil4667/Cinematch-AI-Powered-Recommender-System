import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommenderSystem:
    def __init__(self, data_path):
        self.data_path = data_path
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.item_similarity = None
        self.user_ratings_mean = None

    def load_data(self):
        movies_path = os.path.join(self.data_path, 'movies.csv')
        ratings_path = os.path.join(self.data_path, 'ratings.csv')
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Clean duplicates
        self.ratings_df.drop_duplicates(inplace=True)
        self.movies_df.drop_duplicates(subset=['movieId'], inplace=True)
        
    def build_user_item_matrix(self):
        # Create user-item interaction matrix
        self.user_item_matrix = self.ratings_df.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
    def train_models(self):
        logger.info("Training Matrix Factorization Model (SVD)...")
        matrix = self.user_item_matrix.values
        
        # Demean the data based on user average
        self.user_ratings_mean = np.mean(matrix, axis=1)
        matrix_demeaned = matrix - self.user_ratings_mean.reshape(-1, 1)
        
        # Compute Truncated SVD for latent user and item features
        self.user_factors = self.svd.fit_transform(matrix_demeaned)
        self.item_factors = self.svd.components_.T
        
        # Item-item collaborative filtering for hybrid system usage
        logger.info("Training Item-Item Similarity Model...")
        self.item_similarity = cosine_similarity(self.item_factors)
        
    def evaluate(self):
        logger.info("Evaluating Model...")
        # Predict ratings for all items for all users
        predictions = np.dot(self.user_factors, self.item_factors.T) + self.user_ratings_mean.reshape(-1, 1)
        matrix = self.user_item_matrix.values
        
        # Calculate RMSE only on actual observed ratings ignoring unrated items (0)
        mask = matrix > 0
        mse = np.mean((predictions[mask] - matrix[mask])**2)
        rmse = np.sqrt(mse)
        logger.info(f"RMSE (Training Overlap): {rmse:.4f}")
        return rmse

    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'movies_df': self.movies_df,
            'user_item_matrix': self.user_item_matrix,
            'svd': self.svd,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'item_similarity': self.item_similarity,
            'user_ratings_mean': self.user_ratings_mean
        }, os.path.join(path, 'recommender_core.pkl'))
        logger.info("Models saved successfully.")
