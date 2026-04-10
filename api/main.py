from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sys
import os

# Append project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.inference import InferenceEngine

app = FastAPI(
    title="Netflix/Amazon Style Recommender API",
    description="High-performance recommender engine backend built with FastAPI",
    version="1.0.0"
)

# Avoid CORs issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "saved_models")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

engine = None

@app.on_event("startup")
def load_model():
    global engine
    try:
        engine = InferenceEngine(MODEL_DIR)
        print("[OK] Production Model loaded successfully into memory.")
    except Exception as e:
        print(f"[WARN] Model loading failed ({str(e)}). Ensure models are trained.")

# Mount static frontend directory
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def home():
    # Serve the main index.html built with vanilla JS
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Online", "description": "Welcome to the Recommender API. Frontend not found."}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, num_recs: int = 10, method: str = 'hybrid'):
    """
    Get generic personalized recommendations for a unique user.
    Methods: 'svd' (Matrix Factorization) or 'hybrid' (SVD + Item-Item)
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is still loading or hasn't been trained.")
        
    if method == 'svd':
        recs = engine.get_user_recommendations_svd(user_id, num_recs)
    elif method == 'hybrid':
        recs = engine.hybrid_recommendations(user_id, num_recs)
    else:
        raise HTTPException(status_code=400, detail="Method param must be 'svd' or 'hybrid'.")
        
    if isinstance(recs, dict) and "error" in recs:
        raise HTTPException(status_code=404, detail=recs["error"])
        
    return {"user_id": user_id, "method": method, "recommendations": recs}

@app.get("/similar/{movie_id}")
def get_similar_movies(movie_id: int, num_recs: int = 5):
    """
    Find top-N items similar to a given item utilizing cosine similarity.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")
        
    recs = engine.get_similar_movies(movie_id, num_recs)
    if isinstance(recs, dict) and "error" in recs:
        raise HTTPException(status_code=404, detail=recs["error"])
        
    return {"movie_id": movie_id, "similar_movies": recs}
