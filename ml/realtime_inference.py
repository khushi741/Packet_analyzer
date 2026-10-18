import os
import joblib
import pandas as pd
from typing import Optional, List, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# Base directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# Global variables for models
model: Any = None
feature_cols: Optional[List[str]] = None
anomaly_detector: Any = None


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
        if model is None or feature_cols is None:
            raise HTTPException(status_code=503, detail="Model is not trained or loaded yet.")

    # Local reference ensures type checkers narrow Optional[List[str]] to List[str]
    active_cols = feature_cols
    if active_cols is None:
        raise HTTPException(status_code=503, detail="Feature columns are not available.")

    # Map input request to dictionary
    data_dict = features.model_dump()

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
    for col in active_cols:
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
@app.get("/api/health")
def health(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse("<!DOCTYPE html><html><head><title>Health OK</title></head><body><h1>Service Healthy</h1><p>Status: Online</p></body></html>")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "anomaly_detector_loaded": anomaly_detector is not None,
    }


@app.get("/api/analytics")
def get_analytics():
    """Aggregates flow features from CSV and model importances for data showcase dashboard."""
    csv_candidates = [
        os.path.join(BASE_DIR, "data", "flow_features.csv"),
        os.path.join("data", "flow_features.csv"),
        "flow_features.csv",
    ]
    csv_file = None
    for p in csv_candidates:
        if os.path.exists(p):
            csv_file = p
            break

    if not csv_file:
        return {
            "available": False,
            "message": "Dataset not found. Run PCAP capture or synthetic generator."
        }

    try:
        df = pd.read_csv(csv_file)
        total_flows = len(df)
        total_packets = int(df["total_packets"].sum()) if "total_packets" in df.columns else 0
        total_bytes = int(df["total_bytes"].sum()) if "total_bytes" in df.columns else 0
        total_mb = round(total_bytes / (1024 * 1024), 2)
        avg_dur = round(float(df["flow_duration_ms"].mean() / 1000.0), 2) if "flow_duration_ms" in df.columns else 0.0

        # DPI vs ML Identification split
        dpi_mask = df["dpi_app_type"].apply(
            lambda x: str(x).upper() not in ["UNKNOWN", "HTTPS", "HTTP", "NAN", ""]
        ) if "dpi_app_type" in df.columns else pd.Series([False] * total_flows)
        dpi_count = int(dpi_mask.sum())
        ml_count = total_flows - dpi_count

        # Class distribution
        class_dist = {}
        if "label" in df.columns:
            for k, v in df["label"].value_counts().items():
                name = str(k).replace("_", " ").title()
                class_dist[name] = int(v)

        # Feature importances from model
        feature_importance_list = []
        if model is not None and hasattr(model, "feature_importances_"):
            cols = feature_cols
            if cols is None:
                c_path = resolve_model_file("feature_columns.joblib")
                if c_path and os.path.exists(c_path):
                    cols = joblib.load(c_path)
            if cols:
                for feat, imp in sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])[:8]:
                    feature_importance_list.append({
                        "name": feat.replace("_", " ").title(),
                        "raw": feat,
                        "importance": round(float(imp), 4),
                        "percent": round(float(imp) * 100, 1),
                    })

        # Throughput buckets
        bps = df["bytes_per_second"] if "bytes_per_second" in df.columns else pd.Series([])
        throughput_buckets = {
            "< 1 KB/s": int((bps < 1000).sum()),
            "1 - 10 KB/s": int(((bps >= 1000) & (bps < 10000)).sum()),
            "10 - 100 KB/s": int(((bps >= 10000) & (bps < 100000)).sum()),
            "100 KB/s - 1 MB/s": int(((bps >= 100000) & (bps < 1000000)).sum()),
            "> 1 MB/s": int((bps >= 1000000).sum()),
        }

        # Sample flows for table
        sample_flows = []
        if "total_bytes" in df.columns:
            sorted_df = df.sort_values(by="total_bytes", ascending=False)
        else:
            sorted_df = df
        for _, row in sorted_df.head(25).iterrows():
            is_dpi = str(row.get("dpi_app_type", "")).upper() not in ["UNKNOWN", "HTTPS", "HTTP", "NAN", ""]
            app_name = str(row.get("label", row.get("dpi_app_type", "Unknown"))).replace("_", " ").title()
            sample_flows.append({
                "duration_s": round(float(row.get("flow_duration_ms", 0)) / 1000.0, 1),
                "packets": int(row.get("total_packets", 0)),
                "bytes": int(row.get("total_bytes", 0)),
                "bytes_kb": round(float(row.get("total_bytes", 0)) / 1024.0, 1),
                "sni": str(row.get("detected_sni", "")) if pd.notna(row.get("detected_sni")) and str(row.get("detected_sni")).strip() != "" else "-",
                "dpi_type": str(row.get("dpi_app_type", "UNKNOWN")),
                "app": app_name,
                "method": "DPI" if is_dpi else "ML",
            })

        return {
            "available": True,
            "total_flows": total_flows,
            "total_packets": total_packets,
            "total_bytes": total_bytes,
            "total_mb": total_mb,
            "avg_duration_sec": avg_dur,
            "dpi_count": dpi_count,
            "ml_count": ml_count,
            "class_distribution": class_dist,
            "throughput_buckets": throughput_buckets,
            "feature_importances": feature_importance_list,
            "sample_flows": sample_flows,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


# Serve static web dashboard directly from FastAPI if available
if os.path.exists(DASHBOARD_DIR):
    # Route for explicit dashboard home and data view
    @app.get("/dashboard")
    @app.get("/data")
    @app.get("/data.html")
    @app.get("/analytics")
    def dashboard_redirect():
        return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

    # Dedicated routes for calculator
    @app.get("/calculate")
    @app.get("/calculator")
    @app.get("/calculator.html")
    @app.get("/predict-flow")
    def calculator_redirect():
        calc_path = os.path.join(DASHBOARD_DIR, "calculator.html")
        if os.path.exists(calc_path):
            return FileResponse(calc_path)
        return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

    # Mount static assets (style.css, app.js, index.html) at root
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard_static")


if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) == 0

    default_port = 7860 if os.environ.get("SPACE_ID") else 8000
    env_port = os.environ.get("PORT")

    if env_port:
        port = int(env_port)
    else:
        port = default_port
        while is_port_in_use(port) and port < default_port + 20:
            print(f"Port {port} is currently in use, trying port {port + 1}...")
            port += 1

    print(f"Starting server on http://localhost:{port} (or http://127.0.0.1:{port})...")
    uvicorn.run(app, host="0.0.0.0", port=port)
