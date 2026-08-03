FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY ml/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ml/ /app/ml/
COPY dashboard/ /app/dashboard/
COPY data/ /app/data/

# Create data output directories
RUN mkdir -p /app/data /app/ml/models /app/ml/dataset

# Default API port
EXPOSE 8000
# Streamlit port
EXPOSE 8501

CMD ["uvicorn", "ml.realtime_inference:app", "--host", "0.0.0.0", "--port", "8000"]
