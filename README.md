# AI-Powered Recommender System 🎬

A complete, end-to-end Machine Learning Recommender System built with production-grade architecture. Simulates the same core mechanisms found in industry-leading personalized platforms like Netflix and Amazon.

## 🌟 Core Features
- **Collaborative Filtering**: Recommends items by finding similar user tastes using Truncated SVD (Matrix Factorization).
- **Hybrid AI Approach**: Intelligently combines robust Item-Item cosine similarity and SVD algorithms.
- **Explainable AI (XAI)**: Provides context logic to the user (e.g. *"Because you liked the Matrix"*).
- **Production-Ready Backend**: High throughput, low latency FastAPI REST service serving memory-loaded `.pkl` models.
- **Modern Streaming UI**: Streamlit interface customized with CSS mimicking premium dark-mode content libraries.
- **Automated ML Pipeline**: 1-script execution to fetch `ml-latest-small`, preprocess, demean matrices, train and serialize the artifacts.

## 📁 Repository Structure
```
AI_Recommender_System/
│
├── api/
│   └── main.py              # FastAPI server & endpoints routing
├── app/
│   └── main_app.py          # Interactive Streamlit Web UI
├── data/
│   └── dataset.py           # Automated dataset fetching
├── models/
│   ├── recommender.py       # Custom Class for dataset shaping & training routines
│   ├── inference.py         # Latent factor dot-products & similarity matching
│   └── saved_models/        # Output directory for `.pkl` models
├── scripts/
│   └── train_pipeline.py    # Master runner script
├── requirements.txt         # Required Python packages
├── Dockerfile               # Container spec for one-click deployment
└── README.md
```

## 🚀 Setup & Execution Guide (Local)

### 1. Environment Configuration
Ensure Python 3.9+ is installed globally or in your base environment.
```bash
python -m venv venv

# Windows Prompt:
venv\Scripts\activate

# Mac/Linux Prompt:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Complete the Data Training Pipeline
You MUST run this once prior to executing the backend! This trains the model.
```bash
python scripts/train_pipeline.py
```
> *This script dynamically downloads MovieLens 100K data and builds your SVD latent features.*

### 3. Initialize the Core API (Terminal 1)
Your frontend requests predictions directly from this API locally on `localhost:8000`.
```bash
uvicorn api.main:app --reload --port 8000
```
> Interactive API Spec: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Deploy the UI Dashboard (Terminal 2)
In a secondary terminal (ensure virtual environment is active):
```bash
streamlit run app/main_app.py
```

---

## 🚢 Cloud Deployment (Render / Railway)
The project ships with a containerization config enabling immediate Cloud deployments.

### Using Docker
```bash
docker build -t cinema-ai:latest .
docker run -p 8000:8000 -p 8501:8501 cinema-ai:latest
```

### Platform as a Service (PaaS) Guide
1. Link your Github directly to your chosen provider (e.g. Render).
2. **Setup Background worker (API)**:
   - Config: Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. **Setup Foreground App (Streamlit)**:
   - Config: Start Command: `streamlit run app/main_app.py --server.port $PORT --server.address 0.0.0.0`
   - *Ensure you update the `API_URL` const inside `app/main_app.py` to point to the created background worker's internet URL.*

## 📈 Future Scalability Enhancements
- **Neural Collaborative Filtering (NCF)**: Integration of PyTorch autoencoders to capture deep, non-linear relationships.
- **Cold Start Combative Design**: Parse IMDb summary NLP descriptions against newly registered accounts that have zero watch history.
- **Microservice Event Bus**: Shift to RabbitMQ streaming data architecture for live retrains.
