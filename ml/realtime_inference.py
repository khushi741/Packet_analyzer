import os
import sys
import joblib
import pandas as pd
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Base directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# Global variables for models
model = None
feature_cols = None
anomaly_detector = None


def resolve_model_file(filename: str) -> Optional[str]:
    """Find model file from multiple candidate locations."""
    candidates = [
        os.path.join(ML_DIR, "models", filename),
        os.path.join(BASE_DIR, "ml", "models", filename),
        os.path.join("ml", "models", filename),
        os.path.join("models", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_models():
    global model, feature_cols, anomaly_detector
    model_path = resolve_model_file("traffic_classifier.joblib")
    cols_path = resolve_model_file("feature_columns.joblib")
    anomaly_path = resolve_model_file("anomaly_detector.joblib")

    if model_path and cols_path:
        model = joblib.load(model_path)
        feature_cols = joblib.load(cols_path)
        if anomaly_path:
            try:
                anomaly_detector = joblib.load(anomaly_path)
            except Exception as e:
                print(f"Warning: could not load anomaly detector: {e}")
        print("Models loaded successfully.")
    else:
        print("Models not found. Please train the model first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="🛡️ Hybrid DPI + ML Traffic Classifier API",
    description="Real-time network traffic classification combining Deep Packet Inspection with Machine Learning.",
    version="1.0.0",
    lifespan=lifespan,
)

# Load models on initialization
load_models()

# Enable CORS for browser dashboard and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlowFeaturesInput(BaseModel):
    flow_duration_ms: float = 0.0
    total_packets: int = 0
    total_bytes: int = 0
    forward_packets: int = 0
    backward_packets: int = 0
    forward_bytes: int = 0
    backward_bytes: int = 0
    mean_packet_size: float = 0.0
    std_packet_size: float = 0.0
    min_packet_size: int = 0
    max_packet_size: int = 0
    mean_inter_arrival_time_ms: float = 0.0
    std_inter_arrival_time_ms: float = 0.0
    packets_per_second: float = 0.0
    bytes_per_second: float = 0.0
    tcp_udp_flag: int = 6  # default TCP
    detected_sni: str = ""
    dpi_app_type: str = "UNKNOWN"


@app.post("/predict-flow")
def predict_flow(features: FlowFeaturesInput):
    global model, feature_cols, anomaly_detector
    if model is None or feature_cols is None:
        load_models()
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not trained or loaded yet.")

    # Map input request to dictionary (Pydantic v1 and v2 compatible)
    if hasattr(features, "model_dump"):
        data_dict = features.model_dump()
    else:
        data_dict = features.dict()

    # Intelligent feature auto-derivation if user/frontend omitted derived statistics
    duration_ms = float(data_dict.get("flow_duration_ms", 0.0))
    duration_sec = duration_ms / 1000.0 if duration_ms > 0 else 1.0
    tot_packets = int(data_dict.get("total_packets", 0))
    tot_bytes = int(data_dict.get("total_bytes", 0))
    mean_size = float(data_dict.get("mean_packet_size", 0.0))
    if mean_size <= 0 and tot_packets > 0:
        mean_size = tot_bytes / tot_packets
        data_dict["mean_packet_size"] = mean_size

    if data_dict.get("packets_per_second", 0.0) <= 0.0 and tot_packets > 0:
        data_dict["packets_per_second"] = tot_packets / duration_sec
    if data_dict.get("bytes_per_second", 0.0) <= 0.0 and tot_bytes > 0:
        data_dict["bytes_per_second"] = tot_bytes / duration_sec

    if data_dict.get("min_packet_size", 0) <= 0 and mean_size > 0:
        data_dict["min_packet_size"] = max(40, int(mean_size * 0.4))
    if data_dict.get("max_packet_size", 0) <= 0 and mean_size > 0:
        data_dict["max_packet_size"] = min(1500, int(mean_size * 1.5))
    if data_dict.get("std_packet_size", 0.0) <= 0.0 and mean_size > 0:
        data_dict["std_packet_size"] = round(mean_size * 0.25, 2)

    iat = float(data_dict.get("mean_inter_arrival_time_ms", 0.0))
    if data_dict.get("std_inter_arrival_time_ms", 0.0) <= 0.0 and iat > 0:
        data_dict["std_inter_arrival_time_ms"] = round(iat * 0.3, 2)

    # Prepare row matching feature columns order
    row_data = {}
    for col in feature_cols:
        row_data[col] = [data_dict.get(col, 0.0)]

    df_input = pd.DataFrame(row_data)

    # Predict
    prediction = model.predict(df_input)[0]
    probs = model.predict_proba(df_input)[0]
    confidence = float(max(probs))

    # Check for anomaly
    is_anomaly = False
    if anomaly_detector is not None:
        try:
            anomaly_score = anomaly_detector.predict(df_input)[0]
            if anomaly_score == -1:
                is_anomaly = True
        except Exception:
            pass

    return {
        "prediction": prediction,
        "confidence": confidence,
        "is_anomaly": is_anomaly,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "anomaly_detector_loaded": anomaly_detector is not None,
    }


# Serve static web dashboard directly from FastAPI if available
if os.path.exists(DASHBOARD_DIR):
    # Route for explicit dashboard home
    @app.get("/dashboard")
    def dashboard_redirect():
        return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

    # Mount static assets (style.css, app.js, index.html) at root
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard_static")


if __name__ == "__main__":
    import uvicorn
    # Support Hugging Face Spaces (default 7860) or standard Render/Docker ($PORT / 8000)
    default_port = 7860 if os.environ.get("SPACE_ID") else 8000
    port = int(os.environ.get("PORT", default_port))
    print(f"Starting server on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
