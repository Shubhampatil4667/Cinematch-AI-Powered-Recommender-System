import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import download_and_extract
from models.recommender import RecommenderSystem

def main():
    print("========================================")
    print("AI Recommender System - Training Pipeline")
    print("========================================\n")

    print("[Step 1] Downloading and preparing dataset...")
    download_and_extract()
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ml-latest-small")
    
    print("\n[Step 2] Initializing Recommender System...")
    recommender = RecommenderSystem(data_path)
    
    print("\n[Step 3] Loading & Cleaning Data...")
    recommender.load_data()
    print(f"Loaded {len(recommender.movies_df)} movies and {len(recommender.ratings_df)} ratings.")
    
    print("\n[Step 4] Building User-Item Matrix...")
    recommender.build_user_item_matrix()
    
    print("\n[Step 5] Training Models (SVD & Item-Item Similarity)...")
    recommender.train_models()
    
    print("\n[Step 6] Evaluating Model Performance...")
    recommender.evaluate()
    
    print("\n[Step 7] Saving Trained Models for API consumption...")
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "saved_models")
    recommender.save_models(save_path)
    
    print("\n[SUCCESS] Setup Complete! You can now launch the Backend API and Frontend App.")

if __name__ == "__main__":
    main()
