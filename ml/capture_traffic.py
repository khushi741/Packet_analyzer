"""
Live Traffic Capture Script using PyShark (tshark wrapper).

Captures real network traffic on a specified interface, groups packets into
5-tuple flows, computes the same 16 statistical features used by the ML model,
and appends labeled results to ml/dataset/captured_flows.csv.

Usage:
    python ml/capture_traffic.py --interface "Wi-Fi" --label youtube --duration 600
    python ml/capture_traffic.py --interface "Ethernet" --label zoom --duration 900
    python ml/capture_traffic.py --list-interfaces

Requirements:
    - tshark (Wireshark CLI) must be installed and on PATH
    - pip install pyshark
"""

import os
import sys
import csv
import time
import argparse
import statistics
from collections import defaultdict

try:
    import pyshark
except ImportError:
    print("ERROR: pyshark is not installed. Run: pip install pyshark")
    print("NOTE:  pyshark also requires tshark (Wireshark CLI) to be installed.")
    print("       Download from: https://www.wireshark.org/download.html")
    sys.exit(1)


# ── Feature schema (must match train_model.py) ──────────────────────────────
FEATURE_COLUMNS = [
    "flow_duration_ms", "total_packets", "total_bytes",
    "forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
    "mean_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
    "mean_inter_arrival_time_ms", "std_inter_arrival_time_ms",
    "packets_per_second", "bytes_per_second", "tcp_udp_flag",
    "detected_sni", "dpi_app_type", "label", "source",
]

OUTPUT_PATH = "ml/dataset/captured_flows.csv"


def list_interfaces():
    """Print available network interfaces via tshark."""
    print("\nAvailable network interfaces:")
    print("-" * 50)
    try:
        import subprocess
        result = subprocess.run(
            ["tshark", "-D"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("tshark returned an error. Is Wireshark/tshark installed?")
            print(result.stderr)
    except FileNotFoundError:
        print("ERROR: tshark not found on PATH.")
        print("Install Wireshark from https://www.wireshark.org/download.html")
        print("and ensure 'Add tshark to PATH' is checked during installation.")


def extract_five_tuple(packet):
    """Extract 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol) from a packet."""
    try:
        src_ip = str(packet.ip.src)
        dst_ip = str(packet.ip.dst)
    except AttributeError:
        return None  # Non-IP packet

    protocol = 0
    src_port = 0
    dst_port = 0

    try:
        if hasattr(packet, "tcp"):
            protocol = 6
            src_port = int(packet.tcp.srcport)
            dst_port = int(packet.tcp.dstport)
        elif hasattr(packet, "udp"):
            protocol = 17
            src_port = int(packet.udp.srcport)
            dst_port = int(packet.udp.dstport)
        else:
            return None  # Skip non-TCP/UDP
    except (AttributeError, ValueError):
        return None

    return (src_ip, dst_ip, src_port, dst_port, protocol)


def extract_sni(packet):
    """Try to extract TLS SNI from a packet."""
    try:
        if hasattr(packet, "tls"):
            # PyShark exposes SNI under tls.handshake.extensions_server_name
            sni = packet.tls.handshake_extensions_server_name
            return str(sni)
    except AttributeError:
        pass
    return ""


def compute_flow_features(flow_data, label):
    """
    Compute the 16 ML features from raw packet records of a single flow.

    flow_data: dict with keys:
        'timestamps': list of float epoch times
        'sizes': list of int packet sizes
        'directions': list of 'fwd' or 'bwd'
        'protocol': int (6=TCP, 17=UDP)
        'sni': str (first SNI seen, or '')
    """
    timestamps = flow_data["timestamps"]
    sizes = flow_data["sizes"]
    directions = flow_data["directions"]

    if len(timestamps) < 2:
        return None  # Need at least 2 packets for meaningful features

    # Duration
    flow_duration_ms = (timestamps[-1] - timestamps[0]) * 1000.0
    if flow_duration_ms <= 0:
        flow_duration_ms = 1.0  # Avoid division by zero

    # Packet / byte counts
    total_packets = len(sizes)
    total_bytes = sum(sizes)
    fwd_sizes = [s for s, d in zip(sizes, directions) if d == "fwd"]
    bwd_sizes = [s for s, d in zip(sizes, directions) if d == "bwd"]
    forward_packets = len(fwd_sizes)
    backward_packets = len(bwd_sizes)
    forward_bytes = sum(fwd_sizes) if fwd_sizes else 0
    backward_bytes = sum(bwd_sizes) if bwd_sizes else 0

    # Packet size statistics
    mean_packet_size = statistics.mean(sizes) if sizes else 0
    std_packet_size = statistics.stdev(sizes) if len(sizes) >= 2 else 0
    min_packet_size = min(sizes) if sizes else 0
    max_packet_size = max(sizes) if sizes else 0

    # Inter-arrival times
    iats = [
        (timestamps[i] - timestamps[i - 1]) * 1000.0
        for i in range(1, len(timestamps))
    ]
    mean_iat = statistics.mean(iats) if iats else 0
    std_iat = statistics.stdev(iats) if len(iats) >= 2 else 0

    # Rates
    duration_sec = flow_duration_ms / 1000.0
    packets_per_second = total_packets / duration_sec if duration_sec > 0 else 0
    bytes_per_second = total_bytes / duration_sec if duration_sec > 0 else 0

    sni = flow_data.get("sni", "")
    dpi_app_type = "UNKNOWN"
    # Simple SNI-to-DPI mapping for the subset with visible SNI
    if sni:
        sni_lower = sni.lower()
        if "youtube" in sni_lower or "googlevideo" in sni_lower:
            dpi_app_type = "YOUTUBE"
        elif "netflix" in sni_lower or "nflxvideo" in sni_lower:
            dpi_app_type = "NETFLIX"
        elif "zoom" in sni_lower:
            dpi_app_type = "ZOOM"
        elif "facebook" in sni_lower or "fbcdn" in sni_lower or "instagram" in sni_lower:
            dpi_app_type = "FACEBOOK"
        else:
            dpi_app_type = "HTTPS"

    return [
        round(flow_duration_ms, 2),
        total_packets,
        total_bytes,
        forward_packets,
        backward_packets,
        forward_bytes,
        backward_bytes,
        round(mean_packet_size, 2),
        round(std_packet_size, 2),
        min_packet_size,
        max_packet_size,
        round(mean_iat, 2),
        round(std_iat, 2),
        round(packets_per_second, 2),
        round(bytes_per_second, 2),
        flow_data["protocol"],
        sni,
        dpi_app_type,
        label,
        "captured",
    ]


def capture_traffic(interface, label, duration_seconds):
    """
    Capture live traffic, aggregate into flows, compute features, save to CSV.
    """
    print(f"\n{'='*60}")
    print(f"  Live Traffic Capture")
    print(f"  Interface : {interface}")
    print(f"  Label     : {label}")
    print(f"  Duration  : {duration_seconds}s")
    print(f"{'='*60}\n")

    # Raw packet storage keyed by 5-tuple
    flows = defaultdict(lambda: {
        "timestamps": [],
        "sizes": [],
        "directions": [],
        "protocol": 6,
        "sni": "",
        "initiator": None,  # First src_ip seen = forward direction
    })

    print(f"[*] Starting capture on '{interface}' for {duration_seconds} seconds...")
    print("[*] Generate traffic NOW (browse YouTube, start a Zoom call, etc.)\n")

    try:
        capture = pyshark.LiveCapture(
            interface=interface,
            # Only capture IP traffic
            bpf_filter="ip",
        )

        packet_count = 0
        start_time = time.time()

        for packet in capture.sniff_continuously():
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                break

            five_tuple = extract_five_tuple(packet)
            if five_tuple is None:
                continue

            src_ip, dst_ip, src_port, dst_port, protocol = five_tuple

            # Normalize tuple so (A->B) and (B->A) map to same flow
            forward_tuple = five_tuple
            reverse_tuple = (dst_ip, src_ip, dst_port, src_port, protocol)
            if reverse_tuple in flows:
                flow_key = reverse_tuple
                direction = "bwd"
            elif forward_tuple in flows:
                flow_key = forward_tuple
                direction = "fwd"
            else:
                # New flow — first packet is forward
                flow_key = forward_tuple
                direction = "fwd"
                flows[flow_key]["initiator"] = src_ip

            # Determine direction based on initiator
            if flows[flow_key]["initiator"] == src_ip:
                direction = "fwd"
            else:
                direction = "bwd"

            try:
                pkt_size = int(packet.length)
            except (AttributeError, ValueError):
                pkt_size = 0

            try:
                pkt_time = float(packet.sniff_timestamp)
            except (AttributeError, ValueError):
                pkt_time = time.time()

            flows[flow_key]["timestamps"].append(pkt_time)
            flows[flow_key]["sizes"].append(pkt_size)
            flows[flow_key]["directions"].append(direction)
            flows[flow_key]["protocol"] = protocol

            # Try to extract SNI
            sni = extract_sni(packet)
            if sni and not flows[flow_key]["sni"]:
                flows[flow_key]["sni"] = sni

            packet_count += 1
            if packet_count % 100 == 0:
                remaining = duration_seconds - elapsed
                print(f"  [{packet_count} packets captured, {remaining:.0f}s remaining]")

    except KeyboardInterrupt:
        print("\n[!] Capture interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Capture failed: {e}")
        print("Make sure tshark is installed and the interface name is correct.")
        print("Run: python ml/capture_traffic.py --list-interfaces")
        return

    print(f"\n[*] Capture complete. {packet_count} packets in {len(flows)} flows.")

    # Compute features for each flow
    os.makedirs("ml/dataset", exist_ok=True)
    file_exists = os.path.exists(OUTPUT_PATH)

    rows_written = 0
    with open(OUTPUT_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(FEATURE_COLUMNS)

        for flow_key, flow_data in flows.items():
            features = compute_flow_features(flow_data, label)
            if features is not None:
                writer.writerow(features)
                rows_written += 1

    print(f"[*] Exported {rows_written} flows to {OUTPUT_PATH}")
    print(f"[*] Total flows in file: {rows_written + (sum(1 for _ in open(OUTPUT_PATH)) - 1 if file_exists else 0)}")
    print(f"\n[TIP] Run this script again with a different --label to capture more classes.")
    print(f"      Example: python ml/capture_traffic.py --interface \"{interface}\" --label zoom --duration 600")


def main():
    parser = argparse.ArgumentParser(
        description="Capture live network traffic and extract ML flow features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available interfaces
  python ml/capture_traffic.py --list-interfaces

  # Capture YouTube traffic for 10 minutes
  python ml/capture_traffic.py --interface "Wi-Fi" --label youtube --duration 600

  # Capture Zoom traffic for 15 minutes
  python ml/capture_traffic.py --interface "Wi-Fi" --label zoom --duration 900

  # Capture web browsing for 10 minutes
  python ml/capture_traffic.py --interface "Ethernet" --label web_browsing --duration 600

Supported labels:
  youtube, netflix, zoom, web_browsing, file_download, social_media, gaming
        """,
    )
    parser.add_argument(
        "--interface", type=str, default="Wi-Fi",
        help="Network interface name (default: Wi-Fi). Use --list-interfaces to see options.",
    )
    parser.add_argument(
        "--label", type=str, default="web_browsing",
        choices=["youtube", "netflix", "zoom", "web_browsing", "file_download", "social_media", "gaming"],
        help="Traffic class label for this capture session.",
    )
    parser.add_argument(
        "--duration", type=int, default=600,
        help="Capture duration in seconds (default: 600 = 10 minutes).",
    )
    parser.add_argument(
        "--list-interfaces", action="store_true",
        help="List available network interfaces and exit.",
    )

    args = parser.parse_args()

    if args.list_interfaces:
        list_interfaces()
        return

    capture_traffic(args.interface, args.label, args.duration)


if __name__ == "__main__":
    main()
