FROM python:3.10-slim

WORKDIR /app

# Upgrade base and install runtime
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy our robust system over
COPY . .

# Let data fetch inside the docker container process once to bootstrap
RUN python scripts/train_pipeline.py

# Expected default ports to bind to
EXPOSE 8000
EXPOSE 8501

# Start script running concurrently for ease (in real prod use docker-compose)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/main_app.py --server.port 8501 --server.address 0.0.0.0"]
