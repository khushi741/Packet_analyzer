"""
ISCX VPN-nonVPN 2016 Dataset Loader.

Reads public ISCX/CICFlowMeter-format CSV files and maps their columns and
labels to the internal schema used by the ML traffic classifier.

Usage:
    1. Download the ISCX VPN-nonVPN 2016 dataset CSVs from:
       https://www.unb.ca/cic/datasets/vpn.html
    2. Place CSV files in:  ml/dataset/iscx/
    3. Run:  python ml/load_iscx_dataset.py

The script outputs:  ml/dataset/iscx_flows.csv
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

# ── Column mapping: ISCX/CICFlowMeter → internal schema ─────────────────────
# CICFlowMeter CSVs use these column names (with leading spaces in some versions)
COLUMN_MAP = {
    # Duration
    "Flow Duration": "flow_duration_ms",
    " Flow Duration": "flow_duration_ms",
    # Packet counts
    "Total Fwd Packets": "forward_packets",
    " Total Fwd Packets": "forward_packets",
    "Total Backward Packets": "backward_packets",
    " Total Backward Packets": "backward_packets",
    # Byte counts
    "Total Length of Fwd Packets": "forward_bytes",
    " Total Length of Fwd Packets": "forward_bytes",
    "Total Length of Bwd Packets": "backward_bytes",
    " Total Length of Bwd Packets": "backward_bytes",
    # Packet size stats
    "Fwd Packet Length Mean": "mean_fwd_pkt_size",
    " Fwd Packet Length Mean": "mean_fwd_pkt_size",
    "Bwd Packet Length Mean": "mean_bwd_pkt_size",
    " Bwd Packet Length Mean": "mean_bwd_pkt_size",
    "Fwd Packet Length Std": "std_fwd_pkt_size",
    " Fwd Packet Length Std": "std_fwd_pkt_size",
    "Packet Length Mean": "mean_packet_size",
    " Packet Length Mean": "mean_packet_size",
    "Packet Length Std": "std_packet_size",
    " Packet Length Std": "std_packet_size",
    "Min Packet Length": "min_packet_size",
    " Min Packet Length": "min_packet_size",
    "Max Packet Length": "max_packet_size",
    " Max Packet Length": "max_packet_size",
    # Inter-arrival time
    "Flow IAT Mean": "mean_inter_arrival_time_ms",
    " Flow IAT Mean": "mean_inter_arrival_time_ms",
    "Flow IAT Std": "std_inter_arrival_time_ms",
    " Flow IAT Std": "std_inter_arrival_time_ms",
    # Rates
    "Flow Packets/s": "packets_per_second",
    " Flow Packets/s": "packets_per_second",
    "Flow Bytes/s": "bytes_per_second",
    " Flow Bytes/s": "bytes_per_second",
    # Protocol
    "Protocol": "protocol_raw",
    " Protocol": "protocol_raw",
    # Label
    "Label": "iscx_label",
    " Label": "iscx_label",
}

# ── Label mapping: ISCX application names → our 6 traffic classes ────────────
# The ISCX VPN-nonVPN dataset uses labels like "youtube", "vimeo", "facebook",
# "hangouts", "skype", "spotify", "ftps", etc.
LABEL_MAP = {
    # Video streaming → youtube / netflix
    "youtube": "youtube",
    "vimeo": "youtube",         # Similar streaming behavior
    "netflix": "netflix",
    # Video conferencing → zoom
    "hangouts": "zoom",         # Similar bidirectional A/V pattern
    "skype": "zoom",
    "voipbuster": "zoom",
    # Social media
    "facebook": "social_media",
    "aim": "social_media",
    "icq": "social_media",
    # File transfer → file_download
    "ftps": "file_download",
    "sftp": "file_download",
    "scp": "file_download",
    # Web / email → web_browsing
    "email": "web_browsing",
    "gmail": "web_browsing",
    "https": "web_browsing",
    "http": "web_browsing",
    # Streaming audio (mapped to social_media as closest match)
    "spotify": "social_media",
    # Torrent → file_download
    "torrent": "file_download",
    "bittorrent": "file_download",
}

OUTPUT_PATH = "ml/dataset/iscx_flows.csv"
ISCX_DIR = "ml/dataset/iscx"


def load_iscx_dataset():
    """Load and convert ISCX/CICFlowMeter CSVs to internal schema."""

    if not os.path.exists(ISCX_DIR):
        os.makedirs(ISCX_DIR, exist_ok=True)
        print(f"Created directory: {ISCX_DIR}")
        print(f"\nTo use this loader:")
        print(f"  1. Download ISCX VPN-nonVPN 2016 dataset from:")
        print(f"     https://www.unb.ca/cic/datasets/vpn.html")
        print(f"  2. Place the CSV files in: {ISCX_DIR}/")
        print(f"  3. Re-run: python ml/load_iscx_dataset.py")
        return

    csv_files = glob.glob(os.path.join(ISCX_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {ISCX_DIR}/")
        print(f"\nDownload the ISCX VPN-nonVPN 2016 dataset from:")
        print(f"  https://www.unb.ca/cic/datasets/vpn.html")
        print(f"Place CSV files in {ISCX_DIR}/ and re-run this script.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {ISCX_DIR}/")

    all_dfs = []
    for csv_path in csv_files:
        print(f"  Loading: {os.path.basename(csv_path)}...", end=" ")
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            all_dfs.append(df)
            print(f"{len(df)} rows")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_dfs:
        print("No data loaded successfully.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(df)}")

    # ── Rename columns ───────────────────────────────────────────────────────
    # Build rename map from whatever columns exist in the data
    rename_map = {}
    for orig_col, new_col in COLUMN_MAP.items():
        stripped = orig_col.strip()
        if stripped in df.columns:
            rename_map[stripped] = new_col
        elif orig_col in df.columns:
            rename_map[orig_col] = new_col

    df = df.rename(columns=rename_map)

    # ── Map labels ───────────────────────────────────────────────────────────
    if "iscx_label" in df.columns:
        # Normalize: lowercase, strip whitespace, remove "vpn-" prefix
        df["iscx_label"] = df["iscx_label"].astype(str).str.strip().str.lower()
        df["iscx_label"] = df["iscx_label"].str.replace("vpn-", "", regex=False)
        df["label"] = df["iscx_label"].map(LABEL_MAP).fillna("unknown")
    elif "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip().str.lower()
        df["Label"] = df["Label"].str.replace("vpn-", "", regex=False)
        df["label"] = df["Label"].map(LABEL_MAP).fillna("unknown")
    else:
        print("WARNING: No 'Label' column found. All flows labeled 'unknown'.")
        df["label"] = "unknown"

    # ── Compute derived columns ──────────────────────────────────────────────
    # total_packets and total_bytes
    if "forward_packets" in df.columns and "backward_packets" in df.columns:
        df["total_packets"] = (
            pd.to_numeric(df["forward_packets"], errors="coerce").fillna(0)
            + pd.to_numeric(df["backward_packets"], errors="coerce").fillna(0)
        ).astype(int)

    if "forward_bytes" in df.columns and "backward_bytes" in df.columns:
        df["total_bytes"] = (
            pd.to_numeric(df["forward_bytes"], errors="coerce").fillna(0)
            + pd.to_numeric(df["backward_bytes"], errors="coerce").fillna(0)
        ).astype(int)

    # ISCX flow duration is in microseconds — convert to milliseconds
    if "flow_duration_ms" in df.columns:
        df["flow_duration_ms"] = (
            pd.to_numeric(df["flow_duration_ms"], errors="coerce").fillna(0) / 1000.0
        )

    # ISCX IAT is in microseconds — convert to milliseconds
    for iat_col in ["mean_inter_arrival_time_ms", "std_inter_arrival_time_ms"]:
        if iat_col in df.columns:
            df[iat_col] = (
                pd.to_numeric(df[iat_col], errors="coerce").fillna(0) / 1000.0
            )

    # TCP/UDP flag from protocol number
    if "protocol_raw" in df.columns:
        df["tcp_udp_flag"] = pd.to_numeric(df["protocol_raw"], errors="coerce").fillna(6).astype(int)
    else:
        df["tcp_udp_flag"] = 6

    # Add source column and metadata
    df["detected_sni"] = ""
    df["dpi_app_type"] = "UNKNOWN"
    df["source"] = "iscx"

    # ── Select and clean final columns ───────────────────────────────────────
    final_cols = [
        "flow_duration_ms", "total_packets", "total_bytes",
        "forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
        "mean_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
        "mean_inter_arrival_time_ms", "std_inter_arrival_time_ms",
        "packets_per_second", "bytes_per_second", "tcp_udp_flag",
        "detected_sni", "dpi_app_type", "label", "source",
    ]

    # Fill missing columns with 0
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0 if col not in ["detected_sni", "dpi_app_type", "label", "source"] else ""

    df_out = df[final_cols].copy()

    # Clean numeric columns
    numeric_cols = final_cols[:16]
    for col in numeric_cols:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0)
        df_out[col] = df_out[col].replace([np.inf, -np.inf], 0)

    # Remove rows with all-zero features (empty flows)
    feature_sum = df_out[numeric_cols].sum(axis=1)
    df_out = df_out[feature_sum > 0]

    # Remove the "unknown" label rows (unmapped ISCX labels)
    known = df_out[df_out["label"] != "unknown"]
    unknown = df_out[df_out["label"] == "unknown"]
    print(f"\nLabel distribution:")
    print(known["label"].value_counts().to_string())
    if len(unknown) > 0:
        print(f"  (dropped {len(unknown)} unmapped 'unknown' rows)")
    df_out = known

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df_out)} flows to {OUTPUT_PATH}")


if __name__ == "__main__":
    load_iscx_dataset()
