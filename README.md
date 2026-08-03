# 🛡️ Packet Analyzer — Hybrid DPI + ML Traffic Classifier

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue?logo=cplusplus)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.80%2B-009688?logo=fastapi)

A hybrid network traffic classification engine that combines **rule-based Deep Packet Inspection (DPI)** with **Machine Learning** to identify encrypted application traffic (YouTube, Netflix, Zoom, etc.) even when payload inspection is impossible.

---

## Problem Statement

Traditional Deep Packet Inspection identifies network applications by reading plaintext identifiers inside packets — specifically the **TLS Server Name Indication (SNI)** field and **HTTP Host** header. However, modern internet traffic increasingly defeats this approach: **TLS 1.3 with Encrypted Client Hello (ECH)**, **QUIC/HTTP3**, and VPN tunneling hide or encrypt these identifiers entirely. As a result, a DPI-only engine classifies 30–70% of modern traffic as "Unknown."

This project addresses the gap by adding a **Machine Learning fallback**: when DPI cannot read encrypted payloads, the system extracts **behavioral flow metadata** (packet sizes, timing patterns, directionality, throughput) and uses a trained Random Forest classifier to predict the application type with high accuracy — no payload decryption required.

---

## Architecture

```text
               ┌──────────────────────────────────────┐
               │         PCAP File / Network          │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │      C++ Packet Parser (Ethernet     │
               │       → IP → TCP/UDP dissection)     │
               └──────────────────┬───────────────────┘
                                  │
                        ┌─────────┴─────────┐
                        │                   │
                   SNI Visible?         SNI Hidden?
                        │                   │
                        ▼                   ▼
               ┌────────────────┐  ┌────────────────────┐
               │  Rule-Based    │  │  Extract 16 Flow   │
               │  DPI Match     │  │  Features (stats,  │
               │  (SNI → App)   │  │  timing, sizes)    │
               └────────┬───────┘  └────────┬───────────┘
                        │                   │
                        │           HTTP POST to FastAPI
                        │           (port 8080)
                        │                   │
                        │                   ▼
                        │          ┌────────────────────┐
                        │          │  Python ML Service  │
                        │          │  (RandomForest /    │
                        │          │   GradientBoosting) │
                        │          └────────┬───────────┘
                        │                   │
                        ▼                   ▼
               ┌──────────────────────────────────────┐
               │    Unified Classification Report     │
               │    + Real-Time Dashboard             │
               └──────────────────────────────────────┘
```

---

## Project Structure

```
Packet_analyzer/
├── include/                        # C++ headers
│   ├── ml_feature_exporter.h       #   Flow feature extraction structs
│   ├── ml_classifier.h             #   HTTP client for ML API calls
│   ├── packet_parser.h             #   Ethernet/IP/TCP packet dissection
│   ├── flow_tracker.h              #   5-tuple flow aggregation
│   └── sni_extractor.h             #   TLS SNI extraction
├── src/                            # C++ source
│   ├── main_working.cpp            #   Single-threaded DPI + ML pipeline
│   ├── dpi_mt.cpp                  #   Multi-threaded pipeline
│   ├── ml_feature_exporter.cpp     #   Feature computation implementation
│   └── ml_classifier.cpp           #   Socket client implementation
├── ml/                             # Python ML pipeline
│   ├── capture_traffic.py          #   Live traffic capture (pyshark/tshark)
│   ├── load_iscx_dataset.py        #   ISCX VPN-nonVPN dataset loader
│   ├── generate_synthetic_flows.py #   Synthetic training data generator
│   ├── build_dataset.py            #   Multi-source dataset merger
│   ├── train_model.py              #   Model training + rigorous evaluation
│   ├── predict.py                  #   CLI prediction tool
│   ├── realtime_inference.py       #   FastAPI REST API server
│   ├── test_pipeline.py            #   Unit tests
│   ├── requirements.txt            #   Python dependencies
│   ├── dataset/                    #   Training data (generated/captured)
│   └── models/                     #   Trained models + evaluation artifacts
│       └── evaluation/             #     Metrics, confusion matrix, reports
├── dashboard/                      # Frontend interfaces
│   ├── index.html                  #   Interactive HTML/JS sandbox dashboard
│   ├── style.css                   #   Dark-mode glassmorphism styling
│   ├── app.js                      #   Chart.js visualization + API client
│   └── app.py                      #   Streamlit analytics dashboard
├── CMakeLists.txt                  # C++ build configuration
├── Dockerfile                      # Container build
├── docker-compose.yml              # Multi-service orchestration
├── DPI_ENGINE_DOCS.md              # Detailed C++ DPI engine documentation
└── HYBRID_ML_EXTENSION.md          # ML extension technical guide
```

---

## Setup & Run

### Prerequisites

- **Python 3.9+** with pip
- **tshark** (Wireshark CLI) — only needed for live traffic capture
- **CMake + C++17 compiler** — only needed to compile C++ engine

### 1. Install Python Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r ml/requirements.txt
```

### 2. Prepare Training Data

You have three options (can be combined):

```bash
# Option A: Generate synthetic training data (quickest, always works)
python ml/generate_synthetic_flows.py

# Option B: Capture real traffic (requires tshark installed)
python ml/capture_traffic.py --interface "Wi-Fi" --label youtube --duration 600
python ml/capture_traffic.py --interface "Wi-Fi" --label zoom --duration 600
python ml/capture_traffic.py --interface "Wi-Fi" --label web_browsing --duration 600

# Option C: Load ISCX public dataset (place CSVs in ml/dataset/iscx/)
python ml/load_iscx_dataset.py
```

### 3. Build Dataset & Train Model

```bash
python ml/build_dataset.py     # Merge all data sources
python ml/train_model.py       # Train + evaluate (outputs to ml/models/evaluation/)
```

### 4. Start the ML Classification API

```bash
uvicorn ml.realtime_inference:app --reload --port 8080
```

### 5. Open the Dashboard

```bash
# Option A: HTML/JS sandbox (open in browser)
# Windows:
ii dashboard/index.html
# Or simply double-click dashboard/index.html in File Explorer

# Option B: Streamlit analytics dashboard
streamlit run dashboard/app.py
```

### 6. Compile & Run C++ Engine (optional, requires CMake)

```bash
cmake -B build
cmake --build build
./build/dpi_working test_dpi.pcap output.pcap
```

---

## Results

This project evaluates the classifier using two distinct testing protocols: a synthetic train/test split and a direct generalization test against live captured traffic.

### 1. Synthetic Benchmark
*This represents model evaluation on a clean, generated test set mirroring the training distributions.*

| Metric | Score |
|---|---|
| **Model** | RandomForest (100 trees) |
| **Test Accuracy** | **98.33%** |
| **Weighted Precision** | 98.35% |
| **Weighted Recall** | 98.33% |
| **Weighted F1-Score** | 98.33% |
| **Overfitting Gap** | 1.67% (✓ No overfitting) |

### 2. Real-World Generalization Test
*This represents the model trained **only on synthetic data** and evaluated on **100% real traffic** captured from your active network connection (215 flows).*

| Metric | Score |
|---|---|
| **Real Test Accuracy** | **13.02%** |
| **Weighted Precision** | 11.64% |
| **Weighted Recall** | 13.02% |
| **Weighted F1-Score** | 10.66% |
| **Mean Model Confidence** | 37.10% |
| **Generalization Gap** | **85.31%** (Generalization Failure) |

#### Per-Class Real-World Accuracy:
* **Zoom**: **73.08%** (19/26) ✓
* **YouTube**: **21.43%** (9/42) ⚠
* **Web Browsing**: **0.00%** (0/61) ⚠
* **Social Media**: **0.00%** (0/42) ⚠
* **File Download**: **0.00%** (0/44) ⚠

---

## Error Analysis & Hypotheses

The large generalization gap (**85.31%**) exposes the limitation of training models purely on synthetic distributions. Here is a breakdown of why the model struggled:

1. **Zoom Generalization Success (73.08%)**: 
   Zoom was the only class that transferred successfully. This is because video calls generate highly distinctive, symmetric, continuous bidirectional traffic (audio/video channels run in parallel). The model successfully learned this pattern and could distinguish it from asymmetric web/file traffic.
2. **Metadata-Based Feature Coarseness**: 
   Since packet-level packet drivers (Npcap/tshark) were unavailable, real traffic features were estimated from active connection metadata (`psutil`). Polling connections every 2 seconds produces coarser, time-averaged metrics compared to the high-resolution packet timestamps generated in synthetic data.
3. **Class Overlap & Traffic Noise**:
   In the synthetic generator, YouTube always has a high mean packet size (800–1300 bytes), while web browsing is always smaller (200–500 bytes). In reality, modern web browsing (e.g., loading media-heavy news sites or GitHub) pulls down large assets, causing its packet size profile to look identical to video streaming to the model. This is visible in the confusion matrix where **web_browsing and social_media were consistently misclassified as netflix**.
4. **CDN Intermediaries**:
   Real streaming traffic (YouTube/Netflix) is served via distributed Content Delivery Networks (CDNs) and uses optimized adaptive bitrate algorithms (e.g., DASH). This introduces high variance and burst patterns that simple uniform random variables cannot replicate.

---

## DPI vs ML Comparison

* **DPI-only Evaluation**: On the real captured test set, rule-based DPI (SNI-matching) classified **23.3% of flows** correctly. The remaining **76.7%** of flows had no visible SNI and were classified as `UNKNOWN`.
* **ML Integration Value**: While the synthetic-only model has poor generalization to raw real traffic metrics (13% accuracy), it successfully classified Zoom traffic with **73.1% accuracy** without needing any plaintext SNI. To close the remaining gap, the model must be retrained using a portion of the real captured flows.

> Detailed reports and confusion matrices are located in:
> - `ml/models/evaluation/classification_report.txt` — Class statistics
> - `ml/models/evaluation/confusion_matrix_real.png` — Heatmap of real misclassifications
> - `ml/models/evaluation/real_world_report.txt` — Generalization error analysis log

---

## Dashboards

### Interactive Sandbox (`dashboard/index.html`)
A dark-mode glassmorphism web interface for testing individual flow classifications in real-time. Adjust packet size, duration, and byte count parameters, then submit to the FastAPI backend for instant predictions with confidence scores and interactive Chart.js visualizations.

### Streamlit Analytics (`dashboard/app.py`)
A data analytics dashboard that reads from the C++ engine's CSV output (`data/flow_features.csv`) to display global traffic distributions, model confidence histograms, and feature importance charts.

---

## Limitations

1. **Dataset size**: The synthetic generator produces 1,200 flows. For production use, supplement with real captured traffic or the ISCX dataset (thousands of flows per class).
2. **Traffic classes**: Currently supports 6 application types (YouTube, Netflix, Zoom, Web Browsing, File Download, Social Media). Adding more classes requires extending the training data and label maps.
3. **Network conditions**: Models trained on one network environment (e.g., home Wi-Fi) may not generalize perfectly to different networks (e.g., corporate, mobile). Retraining on local captures improves accuracy.
4. **Single-machine inference**: The FastAPI service runs locally. For production deployment, consider load balancing and model serving frameworks.
5. **Temporal drift**: Network application behavior changes over time (codec updates, protocol changes). Periodic retraining is recommended.
6. **Encrypted metadata**: The system cannot classify traffic that uses fully encrypted transport with no distinguishable behavioral patterns (e.g., constant-rate padding).

---

## Documentation

- **[DPI Engine Deep Dive](DPI_ENGINE_DOCS.md)** — comprehensive documentation of the C++ packet parsing, flow tracking, SNI extraction, and blocking engine
- **[ML Extension Guide](HYBRID_ML_EXTENSION.md)** — technical details of the ML pipeline, feature engineering, and API integration

---

## License

This project is for educational and research purposes.
