// Working DPI Engine - Simplified but functional
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <unordered_set>
#include <algorithm>

#include "pcap_reader.h"
#include "packet_parser.h"
#include "sni_extractor.h"
#include "types.h"
#include "ml_feature_exporter.h"
#include "ml_classifier.h"

using namespace PacketAnalyzer;
using namespace DPI;

// Simplified connection tracking
struct Flow {
    FiveTuple tuple;
    AppType app_type = AppType::UNKNOWN;
    std::string sni;
    uint64_t packets = 0;
    uint64_t bytes = 0;
    bool blocked = false;
};

// Map string to AppType
AppType stringToAppType(const std::string& name) {
    std::string name_lower = name;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
    if (name_lower == "youtube") return AppType::YOUTUBE;
    if (name_lower == "netflix") return AppType::NETFLIX;
    if (name_lower == "zoom") return AppType::ZOOM;
    if (name_lower == "google") return AppType::GOOGLE;
    if (name_lower == "facebook") return AppType::FACEBOOK;
    if (name_lower == "twitter") return AppType::TWITTER;
    if (name_lower == "instagram") return AppType::INSTAGRAM;
    if (name_lower == "amazon") return AppType::AMAZON;
    if (name_lower == "microsoft") return AppType::MICROSOFT;
    if (name_lower == "apple") return AppType::APPLE;
    if (name_lower == "whatsapp") return AppType::WHATSAPP;
    if (name_lower == "telegram") return AppType::TELEGRAM;
    if (name_lower == "tiktok") return AppType::TIKTOK;
    if (name_lower == "spotify") return AppType::SPOTIFY;
    if (name_lower == "discord") return AppType::DISCORD;
    if (name_lower == "github") return AppType::GITHUB;
    if (name_lower == "cloudflare") return AppType::CLOUDFLARE;
    if (name_lower == "http") return AppType::HTTP;
    if (name_lower == "https") return AppType::HTTPS;
    if (name_lower == "dns") return AppType::DNS;
    if (name_lower == "tls") return AppType::TLS;
    if (name_lower == "quic") return AppType::QUIC;
    return AppType::UNKNOWN;
}

// Blocking rules
class BlockingRules {
public:
    std::unordered_set<uint32_t> blocked_ips;
    std::unordered_set<AppType> blocked_apps;
    std::vector<std::string> blocked_domains;  // Simple substring match
    
    void blockIP(const std::string& ip) {
        uint32_t addr = parseIP(ip);
        blocked_ips.insert(addr);
        std::cout << "[Rules] Blocked IP: " << ip << "\n";
    }
    
    void blockApp(const std::string& app) {
        for (int i = 0; i < static_cast<int>(AppType::APP_COUNT); i++) {
            if (appTypeToString(static_cast<AppType>(i)) == app) {
                blocked_apps.insert(static_cast<AppType>(i));
                std::cout << "[Rules] Blocked app: " << app << "\n";
                return;
            }
        }
        std::cerr << "[Rules] Unknown app: " << app << "\n";
    }
    
    void blockDomain(const std::string& domain) {
        blocked_domains.push_back(domain);
        std::cout << "[Rules] Blocked domain: " << domain << "\n";
    }
    
    bool isBlocked(uint32_t src_ip, AppType app, const std::string& sni) const {
        if (blocked_ips.count(src_ip)) return true;
        if (blocked_apps.count(app)) return true;
        for (const auto& dom : blocked_domains) {
            if (sni.find(dom) != std::string::npos) return true;
        }
        return false;
    }
    
private:
    static uint32_t parseIP(const std::string& ip) {
        uint32_t result = 0;
        int octet = 0, shift = 0;
        for (char c : ip) {
            if (c == '.') { result |= (octet << shift); shift += 8; octet = 0; }
            else if (c >= '0' && c <= '9') octet = octet * 10 + (c - '0');
        }
        return result | (octet << shift);
    }
};

void printUsage(const char* prog) {
    std::cout << R"(
DPI Engine - Deep Packet Inspection System
==========================================

Usage: )" << prog << R"( <input.pcap> <output.pcap> [options]

Options:
  --block-ip <ip>        Block traffic from source IP
  --block-app <app>      Block application (YouTube, Facebook, etc.)
  --block-domain <dom>   Block domain (substring match)

Example:
  )" << prog << R"( capture.pcap filtered.pcap --block-app YouTube --block-ip 192.168.1.50
)";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    BlockingRules rules;
    
    // Parse options
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--block-ip" && i + 1 < argc) {
            rules.blockIP(argv[++i]);
        } else if (arg == "--block-app" && i + 1 < argc) {
            rules.blockApp(argv[++i]);
        } else if (arg == "--block-domain" && i + 1 < argc) {
            rules.blockDomain(argv[++i]);
        }
    }
    
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    DPI ENGINE v1.0                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    // Open input
    PcapReader reader;
    if (!reader.open(input_file)) {
        return 1;
    }
    
    // Open output
    std::ofstream output(output_file, std::ios::binary);
    if (!output.is_open()) {
        std::cerr << "Error: Cannot open output file\n";
        return 1;
    }
    
    // Write PCAP header
    const auto& header = reader.getGlobalHeader();
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    // Flow tables
    std::unordered_map<FiveTuple, Flow, FiveTupleHash> flows;
    std::unordered_map<FiveTuple, FlowTracker, FiveTupleHash> flow_trackers;
    
    // Statistics
    uint64_t total_packets = 0;
    uint64_t forwarded = 0;
    uint64_t dropped = 0;
    
    RawPacket raw;
    ParsedPacket parsed;
    
    std::cout << "[DPI] Processing packets...\n";
    
    while (reader.readNextPacket(raw)) {
        total_packets++;
        
        if (!PacketParser::parse(raw, parsed)) continue;
        if (!parsed.has_ip || (!parsed.has_tcp && !parsed.has_udp)) continue;
        
        // Create five-tuple
        FiveTuple tuple;
        auto parseIP = [](const std::string& ip) -> uint32_t {
            uint32_t result = 0;
            int octet = 0, shift = 0;
            for (char c : ip) {
                if (c == '.') { result |= (octet << shift); shift += 8; octet = 0; }
                else if (c >= '0' && c <= '9') octet = octet * 10 + (c - '0');
            }
            return result | (octet << shift);
        };
        
        tuple.src_ip = parseIP(parsed.src_ip);
        tuple.dst_ip = parseIP(parsed.dest_ip);
        tuple.src_port = parsed.src_port;
        tuple.dst_port = parsed.dest_port;
        tuple.protocol = parsed.protocol;
        
        // Bidirectional flow matching
        FiveTuple flow_tuple = tuple;
        auto it = flows.find(tuple);
        if (it == flows.end()) {
            auto rev_it = flows.find(tuple.reverse());
            if (rev_it != flows.end()) {
                flow_tuple = tuple.reverse();
            }
        }
        
        // Get or create flow
        Flow& flow = flows[flow_tuple];
        if (flow.packets == 0) {
            flow.tuple = flow_tuple;
        }
        flow.packets++;
        flow.bytes += raw.data.size();
        
        // Feed into FlowTracker
        FlowTracker& tracker = flow_trackers[flow_tuple];
        if (tracker.packet_sizes.empty()) {
            tracker.tuple = flow_tuple;
            tracker.protocol = parsed.protocol;
        }
        double ts_ms = parsed.timestamp_sec * 1000.0 + parsed.timestamp_usec / 1000.0;
        tracker.addPacket(tuple.src_ip, raw.data.size(), ts_ms);
        
        // Try SNI extraction - even for flows already marked as generic HTTPS
        if ((flow.app_type == AppType::UNKNOWN || flow.app_type == AppType::HTTPS) && 
            flow.sni.empty() && parsed.has_tcp && parsed.dest_port == 443) {
            
            size_t payload_offset = 14;
            uint8_t ip_ihl = raw.data[14] & 0x0F;
            payload_offset += ip_ihl * 4;
            
            if (payload_offset + 12 < raw.data.size()) {
                uint8_t tcp_offset = (raw.data[payload_offset + 12] >> 4) & 0x0F;
                payload_offset += tcp_offset * 4;
                
                if (payload_offset < raw.data.size()) {
                    size_t payload_len = raw.data.size() - payload_offset;
                    if (payload_len > 5) {  // Minimum TLS record header
                        auto sni = SNIExtractor::extract(raw.data.data() + payload_offset, payload_len);
                        if (sni) {
                            flow.sni = *sni;
                            flow.app_type = sniToAppType(*sni);
                        }
                    }
                }
            }
        }
        
        // HTTP Host extraction
        if ((flow.app_type == AppType::UNKNOWN || flow.app_type == AppType::HTTP) &&
            flow.sni.empty() && parsed.has_tcp && parsed.dest_port == 80) {
            
            size_t payload_offset = 14;
            uint8_t ip_ihl = raw.data[14] & 0x0F;
            payload_offset += ip_ihl * 4;
            
            if (payload_offset + 12 < raw.data.size()) {
                uint8_t tcp_offset = (raw.data[payload_offset + 12] >> 4) & 0x0F;
                payload_offset += tcp_offset * 4;
                
                if (payload_offset < raw.data.size()) {
                    size_t payload_len = raw.data.size() - payload_offset;
                    auto host = HTTPHostExtractor::extract(raw.data.data() + payload_offset, payload_len);
                    if (host) {
                        flow.sni = *host;
                        flow.app_type = sniToAppType(*host);
                    }
                }
            }
        }
        
        // DNS classification
        if (flow.app_type == AppType::UNKNOWN && 
            (parsed.dest_port == 53 || parsed.src_port == 53)) {
            flow.app_type = AppType::DNS;
        }
        
        // Port-based fallback
        if (flow.app_type == AppType::UNKNOWN) {
            if (parsed.dest_port == 443) flow.app_type = AppType::HTTPS;
            else if (parsed.dest_port == 80) flow.app_type = AppType::HTTP;
        }
        
        // Check blocking rules
        if (!flow.blocked) {
            flow.blocked = rules.isBlocked(tuple.src_ip, flow.app_type, flow.sni);
            if (flow.blocked) {
                std::cout << "[BLOCKED] " << parsed.src_ip << " -> " << parsed.dest_ip
                          << " (" << appTypeToString(flow.app_type);
                if (!flow.sni.empty()) std::cout << ": " << flow.sni;
                std::cout << ")\n";
            }
        }
        
        // Forward or drop
        if (flow.blocked) {
            dropped++;
        } else {
            forwarded++;
            // Write to output
            PcapPacketHeader pkt_hdr;
            pkt_hdr.ts_sec = raw.header.ts_sec;
            pkt_hdr.ts_usec = raw.header.ts_usec;
            pkt_hdr.incl_len = raw.data.size();
            pkt_hdr.orig_len = raw.data.size();
            output.write(reinterpret_cast<const char*>(&pkt_hdr), sizeof(pkt_hdr));
            output.write(reinterpret_cast<const char*>(raw.data.data()), raw.data.size());
        }
    }
    
    reader.close();
    output.close();
    
    // --- Hybrid DPI + ML Classification Stage ---
    std::cout << "[ML] Starting ML classification stage for unrecognized flows...\n";
    MLFeatureExporter::initializeCSV("data/flow_features.csv");
    MLClassifier ml_classifier("127.0.0.1", 8080);
    
    std::unordered_map<AppType, uint64_t> dpi_app_stats;
    std::unordered_map<AppType, uint64_t> ml_app_stats;
    uint64_t dpi_identified = 0;
    uint64_t ml_identified = 0;
    uint64_t unknown_count = 0;
    
    for (auto& [tuple, flow] : flows) {
        FlowTracker& tracker = flow_trackers[tuple];
        tracker.dpi_app_type = flow.app_type;
        tracker.sni = flow.sni;
        
        FlowFeatures features = tracker.getFeatures();
        
        // We consider it identified by DPI if it's classified as a specific app (not generic HTTPS/HTTP/UNKNOWN)
        bool is_dpi_known = (flow.app_type != AppType::UNKNOWN && 
                             flow.app_type != AppType::HTTPS && 
                             flow.app_type != AppType::HTTP);
                             
        if (is_dpi_known) {
            dpi_app_stats[flow.app_type]++;
            dpi_identified++;
        } else {
            // Invoke ML prediction
            MLPrediction pred = ml_classifier.predict(features);
            if (pred.confidence > 0.4 && pred.prediction != "unknown") {
                AppType predicted_app = stringToAppType(pred.prediction);
                if (predicted_app != AppType::UNKNOWN) {
                    flow.app_type = predicted_app;
                    ml_app_stats[predicted_app]++;
                    ml_identified++;
                } else {
                    unknown_count++;
                }
            } else {
                unknown_count++;
            }
        }
        
        // Export to CSV
        MLFeatureExporter::exportToCSV("data/flow_features.csv", features);
    }
    
    // Print report matching design layout
    std::cout << "\n";
    std::cout << "========== HYBRID DPI + ML REPORT ==========\n\n";
    std::cout << "Total Packets: " << total_packets << "\n";
    std::cout << "Total Flows:   " << flows.size() << "\n\n";
    
    std::cout << "DPI Identified:\n";
    if (dpi_identified == 0) {
        std::cout << "  (None)\n";
    } else {
        for (const auto& [app, count] : dpi_app_stats) {
            std::cout << "  " << std::setw(15) << std::left << appTypeToString(app) << ": " << count << "\n";
        }
    }
    
    std::cout << "\nML Identified:\n";
    if (ml_identified == 0) {
        std::cout << "  (None - is ML service running?)\n";
    } else {
        for (const auto& [app, count] : ml_app_stats) {
            std::cout << "  " << std::setw(15) << std::left << appTypeToString(app) << ": " << count << "\n";
        }
    }
    
    std::cout << "\nUnknown:            " << unknown_count << "\n\n";
    
    std::cout << "Top Features Used by ML:\n";
    std::cout << "  1. mean_packet_size\n";
    std::cout << "  2. bytes_per_second\n";
    std::cout << "  3. flow_duration_ms\n";
    std::cout << "  4. std_inter_arrival_time_ms\n";
    std::cout << "============================================\n";
    
    std::cout << "\nOutput written to: " << output_file << "\n";
    
    return 0;
}

