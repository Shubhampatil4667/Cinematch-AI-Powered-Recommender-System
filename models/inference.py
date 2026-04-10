import joblib
import os
import numpy as np

class InferenceEngine:
    def __init__(self, model_path):
        model_file = os.path.join(model_path, 'recommender_core.pkl')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found at {model_file}. Run training first.")
            
        model_data = joblib.load(model_file)
        self.movies_df = model_data['movies_df']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.item_similarity = model_data['item_similarity']
        self.user_ratings_mean = model_data['user_ratings_mean']
        
        # Mapping helpers
        self.movie_idx_to_id = self.user_item_matrix.columns.tolist()
        self.movie_id_to_idx = {m_id: idx for idx, m_id in enumerate(self.movie_idx_to_id)}
        self.user_idx_to_id = self.user_item_matrix.index.tolist()
        self.user_id_to_idx = {u_id: idx for idx, u_id in enumerate(self.user_idx_to_id)}

    def get_user_recommendations_svd(self, user_id, num_recs=10):
        if user_id not in self.user_id_to_idx:
            return {"error": "User not found. Try an ID that exists in the dataset."}
        
        user_idx = self.user_id_to_idx[user_id]
        # Generate predictions for all movies for this user based on latent factors
        user_pred_ratings = np.dot(self.user_factors[user_idx, :], self.item_factors.T) + self.user_ratings_mean[user_idx]
        
        user_ratings_original = self.user_item_matrix.iloc[user_idx, :]
        already_rated = user_ratings_original[user_ratings_original > 0].index.tolist()
        
        sorted_indices = np.argsort(user_pred_ratings)[::-1]
        
        recommendations = []
        for idx in sorted_indices:
            movie_id = self.movie_idx_to_id[idx]
            # Don't recommend what they've already seen
            if movie_id not in already_rated:
                movie_data = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                recommendations.append({
                    "movieId": int(movie_id),
                    "title": str(movie_data['title']),
                    "genres": str(movie_data['genres']),
                    "predicted_rating": round(float(user_pred_ratings[idx]), 2),
                    "reason": "Based on your overall taste profile."
                })
                if len(recommendations) >= num_recs:
                    break
        return recommendations

    def get_similar_movies(self, movie_id, num_recs=5):
        if movie_id not in self.movie_id_to_idx:
            return {"error": "Movie not found"}
            
        movie_idx = self.movie_id_to_idx[movie_id]
        sim_scores = self.item_similarity[movie_idx]
        
        # Skip the 0th element itself, sort descending
        sorted_indices = np.argsort(sim_scores)[::-1][1:num_recs+1]
        
        recommendations = []
        for idx in sorted_indices:
            m_id = self.movie_idx_to_id[idx]
            movie_data = self.movies_df[self.movies_df['movieId'] == m_id].iloc[0]
            recommendations.append({
                "movieId": int(m_id),
                "title": str(movie_data['title']),
                "genres": str(movie_data['genres']),
                "similarity_score": round(float(sim_scores[idx]), 3)
            })
        return recommendations

    def hybrid_recommendations(self, user_id, num_recs=10):
        if user_id not in self.user_id_to_idx:
            return {"error": "User not found. Try an ID that exists in the dataset."}
            
        user_idx = self.user_id_to_idx[user_id]
        user_ratings_original = self.user_item_matrix.iloc[user_idx, :]
        
        # Identify movies the user highly praised
        top_movies_indices = np.argsort(user_ratings_original.values)[::-1][:3]
        top_movie_ids = [self.movie_idx_to_id[i] for i in top_movies_indices if user_ratings_original.values[i] > 3]
        
        # Fallback to pure SVD if they haven't liked anything strongly yet
        if not top_movie_ids:
            return self.get_user_recommendations_svd(user_id, num_recs)
            
        # 50% from globally optimal Matrix Factorization
        svd_recs = self.get_user_recommendations_svd(user_id, num_recs)
        final_recs = []
        
        for i, rec in enumerate(svd_recs):
            if i < max(1, num_recs // 2):
                final_recs.append(rec)
            else:
                break
                
        # 50% from Item-Item similarities based on their favorite movie
        top_movie_id = top_movie_ids[0]
        top_movie_title = self.movies_df[self.movies_df['movieId'] == top_movie_id]['title'].values[0]
        
        items_needed = num_recs - len(final_recs)
        similar_movies = self.get_similar_movies(top_movie_id, max(10, items_needed * 2))
        
        already_rated = user_ratings_original[user_ratings_original > 0].index.tolist()
        existing_rec_ids = [r['movieId'] for r in final_recs]
        
        for rec in similar_movies:
            if len(final_recs) >= num_recs:
                break
            # Ignore dupes and already watched
            if rec['movieId'] not in existing_rec_ids and rec['movieId'] not in already_rated:
                rec['reason'] = f"Because you loved '{top_movie_title}'."
                rec['predicted_rating'] = None # Derived from item similarity, predicting raw user score isn't pure
                # Remove extra key layout consistency
                rec.pop('similarity_score', None)
                final_recs.append(rec)
                
        return final_recs
