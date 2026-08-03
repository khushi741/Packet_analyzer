# Hybrid DPI + Machine Learning Traffic Classification Extension

This document provides a comprehensive guide to the **ML-Enhanced Traffic Analyzer** extension added to the Packet_analyzer repository.

---

## Architecture Overview

The extended traffic analyzer operates as a hybrid engine:

1. **Traditional DPI Path**: Reads raw packets, parses network and transport headers, tracks active 5-tuple flows, and performs pattern-matching on unencrypted handshakes (specifically TLS SNI and HTTP Host headers) or protocol-specific rules.
2. **Behavioral ML Path**: When traditional DPI classification returns `UNKNOWN`, the system triggers the ML classification stage. It compiles flow-level statistics (byte ratios, packet counts, inter-arrival times, flow duration) and performs REST API query calls to the Python ML Inference service.

```text
PCAP / Live Traffic
        │
        ▼
C++ Packet Parser
        │
        ▼
Flow Tracker
        │
 ┌──────┴────────┐
 │               │
 ▼               ▼
Traditional DPI  ML Feature Extraction
(SNI / Rules)           │
 │                      ▼
 │             Python ML Service (FastAPI)
 │                      │
 └──────────┬───────────┘
            ▼
    Unified Traffic Classification
```

---

## Why Machine Learning Traffic Classification Is Needed

* **Encrypted Client Hello (ECH)**: New versions of TLS (such as draft revisions of TLS 1.3) encrypt the SNI field, leaving no visible plaintext domains for traditional DPI.
* **HTTP/3 and QUIC**: The shift to UDP-based QUIC wraps transport-layer parameters in encryption, hiding handshake elements from raw parsers.
* **Unknown Protocols**: Ad-hoc VPN traffic, obfuscated games, and custom communication channels do not present standard application signatures.
* **Behavioral Fingerprinting**: Allows classification based purely on packet size distributions, data rates, and directionality metrics.

---

## Hybrid Strategy

| Traffic Scenario | Technique Used | Description |
|---|---|---|
| SNI Plaintext Visible | Traditional DPI | Matches exact domains (e.g. `youtube.com`) to specific app profiles. |
| Known Port/Protocol | Rule-based classification | Fallback matching (e.g. DNS on port 53, HTTPS on port 443). |
| Encrypted / ECH / QUIC | Behavioral ML Classifier | Computes flow feature vectors and uses Random Forest/GBM models. |
| Low Prediction Confidence | Unknown Class | Fallback type when both DPI and ML classifiers cannot determine class. |

---

## Flow Feature Vector Specification

The C++ flow engine records packet metadata and computes the following variables for ML classification:

1. **`flow_duration_ms`**: Duration between the first and last packet of the flow.
2. **`total_packets`**: Total bidirectional packet count.
3. **`total_bytes`**: Total bidirectional byte volume.
4. **`forward_packets`**: Packets sent in the connection-initiation direction.
5. **`backward_packets`**: Packets received in the reverse direction.
6. **`forward_bytes` / `backward_bytes`**: Directional byte volumes.
7. **`mean_packet_size` / `std_packet_size`**: Statistical size metrics.
8. **`min_packet_size` / `max_packet_size`**: Size boundaries.
9. **`mean_inter_arrival_time_ms`**: Average time gap between packet arrivals.
10. **`std_inter_arrival_time_ms`**: Deviation of packet arrival gaps.
11. **`packets_per_second` / `bytes_per_second`**: Average throughput rates.
12. **`tcp_udp_flag`**: Network protocol indicator (6 = TCP, 17 = UDP).

---

## Explainability (SHAP and Feature Importance)

* **Feature Importances**: During model training, scikit-learn computes the impurity-based importance score for each feature. The results are plotted as `ml/models/feature_importance.png` and displayed interactively on the Streamlit dashboard.
* **SHAP Explainability**: SHAP (SHapley Additive exPlanations) values can be calculated using the Python API to detail the positive or negative contribution of each feature towards a predicted class.

---

## Getting Started

### 1. Build C++ Engine
```bash
cmake -B build
cmake --build build
```
This produces `dpi_working` (single-threaded) and `dpi_mt` (multi-threaded).

### 2. Python Environment Setup
```bash
python -m venv .venv
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r ml/requirements.txt
```

### 3. Generate Data and Train Model
```bash
python ml/generate_synthetic_flows.py
python ml/build_dataset.py
python ml/train_model.py
```

### 4. Start ML REST API
```bash
uvicorn ml.realtime_inference:app --reload --port 8000
```

### 5. Run Network Analytics Dashboard
```bash
streamlit run dashboard/app.py
```
