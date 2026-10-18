FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY ml/ /app/ml/
COPY dashboard/ /app/dashboard/
COPY data/ /app/data/

# Ensure output directories exist
RUN mkdir -p /app/data /app/ml/models /app/ml/dataset

# Support dynamic port assignment across cloud providers (Render, Hugging Face, etc.)
ENV PORT=8000
EXPOSE 8000 7860 8501

# Start the unified FastAPI inference engine & interactive dashboard
CMD ["sh", "-c", "uvicorn ml.realtime_inference:app --host 0.0.0.0 --port ${PORT:-8000}"]
