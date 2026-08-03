#include "ml_feature_exporter.h"
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <iostream>

namespace DPI {

static std::mutex csv_mutex;

void FlowTracker::addPacket(uint32_t src_ip, size_t packet_size, double ts_ms) {
    if (!has_forward_ip) {
        forward_ip = src_ip;
        has_forward_ip = true;
    }

    packet_sizes.push_back(packet_size);
    timestamps_ms.push_back(ts_ms);

    if (src_ip == forward_ip) {
        forward_packets++;
        forward_bytes += packet_size;
    } else {
        backward_packets++;
        backward_bytes += packet_size;
    }
}

FlowFeatures FlowTracker::getFeatures() const {
    FlowFeatures f;
    f.total_packets = packet_sizes.size();
    f.forward_packets = forward_packets;
    f.backward_packets = backward_packets;
    f.forward_bytes = forward_bytes;
    f.backward_bytes = backward_bytes;
    f.tcp_udp_flag = protocol;
    f.detected_sni = sni;
    f.dpi_app_type = appTypeToString(dpi_app_type);

    if (f.total_packets == 0) return f;

    // Calculate total bytes
    f.total_bytes = std::accumulate(packet_sizes.begin(), packet_sizes.end(), 0ULL);

    // Min and Max packet sizes
    f.min_packet_size = *std::min_element(packet_sizes.begin(), packet_sizes.end());
    f.max_packet_size = *std::max_element(packet_sizes.begin(), packet_sizes.end());

    // Mean packet size
    f.mean_packet_size = static_cast<double>(f.total_bytes) / f.total_packets;

    // Std of packet sizes
    double size_sq_sum = 0.0;
    for (uint32_t sz : packet_sizes) {
        size_sq_sum += (sz - f.mean_packet_size) * (sz - f.mean_packet_size);
    }
    f.std_packet_size = (f.total_packets > 1) ? std::sqrt(size_sq_sum / f.total_packets) : 0.0;

    // Duration and rate features
    if (timestamps_ms.size() >= 2) {
        f.flow_duration_ms = timestamps_ms.back() - timestamps_ms.front();
        if (f.flow_duration_ms > 0) {
            f.packets_per_second = (f.total_packets / (f.flow_duration_ms / 1000.0));
            f.bytes_per_second = (f.total_bytes / (f.flow_duration_ms / 1000.0));
        }
    }

    // Inter-arrival times
    if (timestamps_ms.size() >= 2) {
        std::vector<double> iats;
        iats.reserve(timestamps_ms.size() - 1);
        for (size_t i = 1; i < timestamps_ms.size(); i++) {
            iats.push_back(timestamps_ms[i] - timestamps_ms[i-1]);
        }

        double iat_sum = std::accumulate(iats.begin(), iats.end(), 0.0);
        f.mean_inter_arrival_time_ms = iat_sum / iats.size();

        double iat_sq_sum = 0.0;
        for (double iat : iats) {
            iat_sq_sum += (iat - f.mean_inter_arrival_time_ms) * (iat - f.mean_inter_arrival_time_ms);
        }
        f.std_inter_arrival_time_ms = std::sqrt(iat_sq_sum / iats.size());
    }

    return f;
}

void MLFeatureExporter::initializeCSV(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(csv_mutex);
    std::ifstream check_file(filepath);
    if (check_file.good()) return; // File already exists

    std::ofstream out(filepath);
    if (!out.is_open()) return;

    out << "flow_duration_ms,total_packets,total_bytes,"
        << "forward_packets,backward_packets,forward_bytes,backward_bytes,"
        << "mean_packet_size,std_packet_size,min_packet_size,max_packet_size,"
        << "mean_inter_arrival_time_ms,std_inter_arrival_time_ms,"
        << "packets_per_second,bytes_per_second,tcp_udp_flag,"
        << "detected_sni,dpi_app_type\n";
    out.close();
}

void MLFeatureExporter::exportToCSV(const std::string& filepath, const FlowFeatures& features) {
    std::lock_guard<std::mutex> lock(csv_mutex);
    std::ofstream out(filepath, std::ios::app);
    if (!out.is_open()) return;

    out << features.flow_duration_ms << ","
        << features.total_packets << ","
        << features.total_bytes << ","
        << features.forward_packets << ","
        << features.backward_packets << ","
        << features.forward_bytes << ","
        << features.backward_bytes << ","
        << features.mean_packet_size << ","
        << features.std_packet_size << ","
        << features.min_packet_size << ","
        << features.max_packet_size << ","
        << features.mean_inter_arrival_time_ms << ","
        << features.std_inter_arrival_time_ms << ","
        << features.packets_per_second << ","
        << features.bytes_per_second << ","
        << static_cast<int>(features.tcp_udp_flag) << ","
        << features.detected_sni << ","
        << features.dpi_app_type << "\n";
    out.close();
}

} // namespace DPI
