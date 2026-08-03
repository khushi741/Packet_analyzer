import os
import csv
import random
import numpy as np

def generate_flows(num_flows=1200):
    classes = ["youtube", "netflix", "zoom", "web_browsing", "file_download", "social_media", "unknown"]
    
    # Feature columns matching step 1
    features = [
        "flow_duration_ms", "total_packets", "total_bytes", 
        "forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
        "mean_packet_size", "std_packet_size", "min_packet_size", "max_packet_size",
        "mean_inter_arrival_time_ms", "std_inter_arrival_time_ms",
        "packets_per_second", "bytes_per_second", "tcp_udp_flag",
        "detected_sni", "dpi_app_type", "label", "source"
    ]
    
    os.makedirs("ml/dataset", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    filepath = "ml/dataset/synthetic_flows.csv"
    
    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(features)
        
        for _ in range(num_flows):
            label = random.choice(classes)
            
            # Default values
            tcp_udp_flag = 6  # Default TCP
            detected_sni = ""
            dpi_app_type = "UNKNOWN"
            
            if label == "youtube":
                flow_duration_ms = random.uniform(5000, 30000)
                total_packets = random.randint(100, 1000)
                forward_packets = int(total_packets * random.uniform(0.1, 0.2))
                backward_packets = total_packets - forward_packets
                
                min_packet_size = 64
                max_packet_size = 1500
                mean_packet_size = random.uniform(800, 1300)
                std_packet_size = random.uniform(200, 400)
                
                forward_bytes = forward_packets * random.uniform(64, 128)
                backward_bytes = backward_packets * random.uniform(1000, 1460)
                total_bytes = forward_bytes + backward_bytes
                
                mean_inter_arrival_time_ms = random.uniform(5, 20)
                std_inter_arrival_time_ms = random.uniform(2, 10)
                
                if random.random() < 0.3:
                    detected_sni = "youtube.com"
                    dpi_app_type = "YOUTUBE"
                    
            elif label == "netflix":
                flow_duration_ms = random.uniform(10000, 60000)
                total_packets = random.randint(200, 2000)
                forward_packets = int(total_packets * random.uniform(0.05, 0.15))
                backward_packets = total_packets - forward_packets
                
                min_packet_size = 64
                max_packet_size = 1500
                mean_packet_size = random.uniform(1000, 1400)
                std_packet_size = random.uniform(100, 300)
                
                forward_bytes = forward_packets * random.uniform(64, 100)
                backward_bytes = backward_packets * random.uniform(1200, 1460)
                total_bytes = forward_bytes + backward_bytes
                
                mean_inter_arrival_time_ms = random.uniform(2, 15)
                std_inter_arrival_time_ms = random.uniform(1, 8)
                
                if random.random() < 0.3:
                    detected_sni = "netflix.com"
                    dpi_app_type = "NETFLIX"
                    
            elif label == "zoom":
                flow_duration_ms = random.uniform(15000, 60000)
                total_packets = random.randint(300, 1500)
                # Zoom is symmetric udp/tcp
                tcp_udp_flag = 17 if random.random() < 0.7 else 6
                forward_packets = int(total_packets * random.uniform(0.45, 0.55))
                backward_packets = total_packets - forward_packets
                
                min_packet_size = 100
                max_packet_size = 1200
                mean_packet_size = random.uniform(300, 600)
                std_packet_size = random.uniform(50, 150)
                
                forward_bytes = forward_packets * mean_packet_size * random.uniform(0.9, 1.1)
                backward_bytes = backward_packets * mean_packet_size * random.uniform(0.9, 1.1)
                total_bytes = forward_bytes + backward_bytes
                
                mean_inter_arrival_time_ms = random.uniform(10, 30)
                std_inter_arrival_time_ms = random.uniform(2, 8)
                
                if random.random() < 0.3:
                    detected_sni = "zoom.us"
                    dpi_app_type = "ZOOM"
                    
            elif label == "web_browsing":
                flow_duration_ms = random.uniform(500, 5000)
                total_packets = random.randint(10, 80)
                forward_packets = int(total_packets * random.uniform(0.3, 0.45))
                backward_packets = total_packets - forward_packets
                
                min_packet_size = 64
                max_packet_size = 1500
                mean_packet_size = random.uniform(200, 500)
                std_packet_size = random.uniform(150, 350)
                
                forward_bytes = forward_packets * random.uniform(100, 300)
                backward_bytes = backward_packets * random.uniform(300, 800)
                total_bytes = forward_bytes + backward_bytes
                
                mean_inter_arrival_time_ms = random.uniform(20, 150)
                std_inter_arrival_time_ms = random.uniform(10, 100)
                
                if random.random() < 0.3:
                    detected_sni = "wikipedia.org"
                    dpi_app_type = "HTTPS"
                    
            elif label == "file_download":
                flow_duration_ms = random.uniform(8000, 45000)
                total_packets = random.randint(500, 3000)
                forward_packets = int(total_packets * random.uniform(0.02, 0.08))
                backward_packets = total_packets - forward_packets
                
                min_packet_size = 64
                max_packet_size = 1500
                mean_packet_size = random.uniform(1100, 1450)
                std_packet_size = random.uniform(50, 200)
                
                forward_bytes = forward_packets * random.uniform(64, 80)
                backward_bytes = backward_packets * random.uniform(1300, 1460)
                total_bytes = forward_bytes + backward_bytes
                
                mean_inter_arrival_time_ms = random.uniform(1, 10)
                std_inter_arrival_time_ms = random.uniform(0.5, 5)
                
            elif label == "social_media":
                flow_duration_ms = random.uniform(2000, 15000)
                total_packets = random.randint(30, 300)
                forward_packets = int(total_packets * random.uniform(0.2, 0.35))
                backward_packets = total_packets - forward_packets
                
                min_packet_size = 64
                max_packet_size = 1500
                mean_packet_size = random.uniform(300, 700)
                std_packet_size = random.uniform(150, 400)
                
                forward_bytes = forward_packets * random.uniform(150, 400)
                backward_bytes = backward_packets * random.uniform(400, 1000)
                total_bytes = forward_bytes + backward_bytes
                
                mean_inter_arrival_time_ms = random.uniform(10, 80)
                std_inter_arrival_time_ms = random.uniform(5, 50)
                
                if random.random() < 0.3:
                    detected_sni = "facebook.com"
                    dpi_app_type = "FACEBOOK"
                    
            else:  # unknown
                flow_duration_ms = random.uniform(100, 3000)
                total_packets = random.randint(3, 20)
                forward_packets = int(total_packets * random.uniform(0.4, 0.6))
                backward_packets = total_packets - forward_packets
                
                min_packet_size = 60
                max_packet_size = 1000
                mean_packet_size = random.uniform(70, 300)
                std_packet_size = random.uniform(10, 150)
                
                forward_bytes = forward_packets * mean_packet_size
                backward_bytes = backward_packets * mean_packet_size
                total_bytes = forward_bytes + backward_bytes
                
                mean_inter_arrival_time_ms = random.uniform(50, 500)
                std_inter_arrival_time_ms = random.uniform(20, 300)
                
            packets_per_second = (total_packets / (flow_duration_ms / 1000.0)) if flow_duration_ms > 0 else 0
            bytes_per_second = (total_bytes / (flow_duration_ms / 1000.0)) if flow_duration_ms > 0 else 0
            
            writer.writerow([
                flow_duration_ms, total_packets, total_bytes,
                forward_packets, backward_packets, forward_bytes, backward_bytes,
                mean_packet_size, std_packet_size, min_packet_size, max_packet_size,
                mean_inter_arrival_time_ms, std_inter_arrival_time_ms,
                packets_per_second, bytes_per_second, tcp_udp_flag,
                detected_sni, dpi_app_type, label, "synthetic"
            ])
            
    print(f"Generated {num_flows} synthetic flows in {filepath}")

if __name__ == "__main__":
    generate_flows()
