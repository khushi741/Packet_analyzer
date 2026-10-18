"""
Dataset Builder — merges multiple data sources into a unified training dataset.

Data source priority:
  1. Captured flows  (ml/dataset/captured_flows.csv)  — PRIMARY (live traffic)
  2. ISCX flows      (ml/dataset/iscx_flows.csv)      — SECONDARY (public dataset)
  3. Synthetic flows  (ml/dataset/synthetic_flows.csv) — SUPPLEMENT (generated)
  4. C++ exported     (data/flow_features.csv)         — RUNTIME (from DPI engine)

Each row includes a `source` column so the trainer can report data provenance.

Usage:
    python ml/build_dataset.py
"""

import os
import sys

# Configure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np


NUMERIC_COLS = [
    "flow_duration_ms", "total_packets", "total_bytes",
    "forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
    "mean_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
    "mean_inter_arrival_time_ms", "std_inter_arrival_time_ms",
    "packets_per_second", "bytes_per_second",
]

DATA_SOURCES = [
    ("ml/dataset/captured_flows.csv",  "captured",  "Live-captured traffic (PRIMARY)"),
    ("ml/dataset/iscx_flows.csv",      "iscx",      "ISCX VPN-nonVPN dataset (SECONDARY)"),
    ("ml/dataset/synthetic_flows.csv", "synthetic",  "Synthetic generated flows (SUPPLEMENT)"),
    ("data/flow_features.csv",         "cpp_export", "C++ DPI engine export (RUNTIME)"),
]

OUTPUT_PATH = "ml/dataset/traffic_dataset.csv"


def build_dataset():
    os.makedirs("ml/dataset", exist_ok=True)

    dfs = []
    source_counts = {}

    print("=" * 60)
    print("  Dataset Builder — Data Source Report")
    print("=" * 60)

    for path, source_tag, description in DATA_SOURCES:
        if os.path.exists(path):
            df = pd.read_csv(path)

            # Add source column if not present
            if "source" not in df.columns:
                df["source"] = source_tag

            # For C++ exports: infer label from dpi_app_type if missing
            if "label" not in df.columns:
                if "dpi_app_type" in df.columns:
                    df["label"] = df["dpi_app_type"].str.lower().fillna("unknown")
                    df["label"] = df["label"].replace({
                        "https": "web_browsing",
                        "http": "web_browsing",
                        "dns": "unknown",
                    })
                else:
                    df["label"] = "unknown"

            dfs.append(df)
            source_counts[source_tag] = len(df)
            print(f"  [+] {description}")
            print(f"    Path: {path}  |  Rows: {len(df)}")
        else:
            print(f"  [-] {description}")
            print(f"    Path: {path}  |  NOT FOUND (skipped)")

    print()

    if not dfs:
        print("ERROR: No data sources found. Run one of:")
        print("  python ml/generate_synthetic_flows.py      (synthetic data)")
        print("  python ml/capture_traffic.py               (live capture)")
        print("  python ml/load_iscx_dataset.py             (ISCX dataset)")
        return

    # Merge all sources
    df_combined = pd.concat(dfs, ignore_index=True)

    # Clean missing values
    df_combined = df_combined.fillna({
        "detected_sni": "",
        "dpi_app_type": "UNKNOWN",
        "source": "unknown",
    })

    for col in NUMERIC_COLS:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")
            df_combined[col] = df_combined[col].fillna(0.0)
            df_combined[col] = df_combined[col].replace([np.inf, -np.inf], 0.0)

    # Fill tcp_udp_flag if missing
    if "tcp_udp_flag" in df_combined.columns:
        df_combined["tcp_udp_flag"] = pd.to_numeric(
            df_combined["tcp_udp_flag"], errors="coerce"
        ).fillna(6).astype(int)
    else:
        df_combined["tcp_udp_flag"] = 6

    # Save
    df_combined.to_csv(OUTPUT_PATH, index=False)

    # Print summary
    total = len(df_combined)
    print("-" * 60)
    print(f"  Combined dataset: {total} total flows")
    print()
    print("  Source breakdown:")
    for src, count in source_counts.items():
        pct = count / total * 100
        bar = "=" * int(pct / 2)
        print(f"    {src:12s}: {count:6d} flows ({pct:5.1f}%) {bar}")

    print()
    print("  Label distribution:")
    label_counts = df_combined["label"].value_counts()
    for label, count in label_counts.items():
        pct = count / total * 100
        print(f"    {label:16s}: {count:6d} ({pct:5.1f}%)")

    print()
    print(f"  Saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset()
