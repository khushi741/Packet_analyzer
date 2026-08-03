import os
import joblib
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ML-Enhanced Traffic Classifier API", version="1.0")

# Global variables for models
model = None
feature_cols = None
anomaly_detector = None

def load_models():
    global model, feature_cols, anomaly_detector
    model_path = "ml/models/traffic_classifier.joblib"
    cols_path = "ml/models/feature_columns.joblib"
    anomaly_path = "ml/models/anomaly_detector.joblib"
    
    if os.path.exists(model_path) and os.path.exists(cols_path):
        model = joblib.load(model_path)
        feature_cols = joblib.load(cols_path)
        if os.path.exists(anomaly_path):
            anomaly_detector = joblib.load(anomaly_path)
        print("Models loaded successfully.")
    else:
        print("Models not found. Please train the model first.")

@app.on_event("startup")
def startup_event():
    load_models()

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
            
    # Map input request to dictionary
    data_dict = features.dict()
    
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
        anomaly_score = anomaly_detector.predict(df_input)[0]
        if anomaly_score == -1:
            is_anomaly = True
            
    return {
        "prediction": prediction,
        "confidence": confidence,
        "is_anomaly": is_anomaly
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
