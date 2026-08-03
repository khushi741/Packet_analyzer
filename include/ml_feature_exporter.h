#ifndef ML_FEATURE_EXPORTER_H
#define ML_FEATURE_EXPORTER_H

#include <string>
#include <vector>
#include <chrono>
#include "types.h"

namespace DPI {

struct FlowFeatures {
    double flow_duration_ms = 0.0;
    uint32_t total_packets = 0;
    uint64_t total_bytes = 0;

    uint32_t forward_packets = 0;
    uint32_t backward_packets = 0;

    uint64_t forward_bytes = 0;
    uint64_t backward_bytes = 0;

    double mean_packet_size = 0.0;
    double std_packet_size = 0.0;
    uint32_t min_packet_size = 0;
    uint32_t max_packet_size = 0;

    double mean_inter_arrival_time_ms = 0.0;
    double std_inter_arrival_time_ms = 0.0;

    double packets_per_second = 0.0;
    double bytes_per_second = 0.0;

    uint8_t tcp_udp_flag = 0;

    std::string detected_sni;
    std::string dpi_app_type;
};

// Tracks packet-level stats inside a flow to compute features at the end
struct FlowTracker {
    FiveTuple tuple;
    uint32_t forward_ip = 0;
    bool has_forward_ip = false;
    
    std::vector<uint32_t> packet_sizes;
    std::vector<double> timestamps_ms; // offset since first packet or epoch

    uint32_t forward_packets = 0;
    uint32_t backward_packets = 0;
    uint64_t forward_bytes = 0;
    uint64_t backward_bytes = 0;
    
    uint8_t protocol = 0;
    std::string sni;
    AppType dpi_app_type = AppType::UNKNOWN;

    void addPacket(uint32_t src_ip, size_t packet_size, double ts_ms);
    FlowFeatures getFeatures() const;
};

class MLFeatureExporter {
public:
    static void exportToCSV(const std::string& filepath, const FlowFeatures& features);
    static void initializeCSV(const std::string& filepath);
};

} // namespace DPI

#endif // ML_FEATURE_EXPORTER_H
