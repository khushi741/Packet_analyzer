#include "ml_classifier.h"
#include <iostream>
#include <sstream>
#include <cstring>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #define close_socket closesocket
#else
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <netdb.h>
    #define close_socket close
#endif

namespace DPI {

MLClassifier::MLClassifier(const std::string& host, int port)
    : host_(host), port_(port) {
    initSockets();
}

MLClassifier::~MLClassifier() {
    cleanupSockets();
}

void MLClassifier::initSockets() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) == 0) {
        ws_initialized_ = true;
    }
#endif
}

void MLClassifier::cleanupSockets() {
#ifdef _WIN32
    if (ws_initialized_) {
        WSACleanup();
        ws_initialized_ = false;
    }
#endif
}

std::string MLClassifier::sendHTTPPost(const std::string& path, const std::string& json_payload) {
    int sock = -1;
    
#ifdef _WIN32
    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return "";
#else
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return "";
#endif

    // Set timeout (e.g. 500ms) to prevent hanging
#ifdef _WIN32
    DWORD timeout = 500;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 500000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#endif

    sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);

    if (inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr) <= 0) {
        close_socket(sock);
        return "";
    }

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close_socket(sock);
        return "";
    }

    // Build HTTP Request
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host_ << ":" << port_ << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << json_payload.length() << "\r\n"
        << "Connection: close\r\n\r\n"
        << json_payload;

    std::string req_str = req.str();
    if (send(sock, req_str.c_str(), req_str.length(), 0) < 0) {
        close_socket(sock);
        return "";
    }

    // Read HTTP Response
    std::string response;
    char buffer[4096];
    int bytes_received = 0;
    while ((bytes_received = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_received] = '\0';
        response.append(buffer, bytes_received);
    }

    close_socket(sock);
    return response;
}

MLPrediction MLClassifier::parseJSONResponse(const std::string& response) {
    MLPrediction pred;
    
    // Find body start (after double CRLF)
    size_t body_start = response.find("\r\n\r\n");
    if (body_start == std::string::npos) return pred;
    
    std::string body = response.substr(body_start + 4);
    
    // Simple JSON value parser
    // Expect: {"prediction": "...", "confidence": ..., "is_anomaly": ...}
    size_t pred_pos = body.find("\"prediction\"");
    if (pred_pos != std::string::npos) {
        size_t val_start = body.find("\"", pred_pos + 12);
        if (val_start != std::string::npos) {
            size_t val_end = body.find("\"", val_start + 1);
            if (val_end != std::string::npos) {
                pred.prediction = body.substr(val_start + 1, val_end - val_start - 1);
            }
        }
    }

    size_t conf_pos = body.find("\"confidence\"");
    if (conf_pos != std::string::npos) {
        size_t val_start = body.find(":", conf_pos + 12);
        if (val_start != std::string::npos) {
            std::string val_str = body.substr(val_start + 1);
            try {
                pred.confidence = std::stod(val_str);
            } catch (...) {
                pred.confidence = 0.0;
            }
        }
    }

    size_t anomaly_pos = body.find("\"is_anomaly\"");
    if (anomaly_pos != std::string::npos) {
        size_t val_start = body.find(":", anomaly_pos + 12);
        if (val_start != std::string::npos) {
            if (body.substr(val_start + 1, 4).find("true") != std::string::npos) {
                pred.is_anomaly = true;
            }
        }
    }

    return pred;
}

MLPrediction MLClassifier::predict(const FlowFeatures& features) {
    // Manually serialize features to JSON
    std::ostringstream json;
    json << "{"
         << "\"flow_duration_ms\":" << features.flow_duration_ms << ","
         << "\"total_packets\":" << features.total_packets << ","
         << "\"total_bytes\":" << features.total_bytes << ","
         << "\"forward_packets\":" << features.forward_packets << ","
         << "\"backward_packets\":" << features.backward_packets << ","
         << "\"forward_bytes\":" << features.forward_bytes << ","
         << "\"backward_bytes\":" << features.backward_bytes << ","
         << "\"mean_packet_size\":" << features.mean_packet_size << ","
         << "\"std_packet_size\":" << features.std_packet_size << ","
         << "\"min_packet_size\":" << features.min_packet_size << ","
         << "\"max_packet_size\":" << features.max_packet_size << ","
         << "\"mean_inter_arrival_time_ms\":" << features.mean_inter_arrival_time_ms << ","
         << "\"std_inter_arrival_time_ms\":" << features.std_inter_arrival_time_ms << ","
         << "\"packets_per_second\":" << features.packets_per_second << ","
         << "\"bytes_per_second\":" << features.bytes_per_second << ","
         << "\"tcp_udp_flag\":" << static_cast<int>(features.tcp_udp_flag) << ","
         << "\"detected_sni\":\"" << features.detected_sni << "\","
         << "\"dpi_app_type\":\"" << features.dpi_app_type << "\""
         << "}";

    std::string response = sendHTTPPost("/predict-flow", json.str());
    if (response.empty()) {
        // Fallback result on connection failure
        return MLPrediction{"unknown", 0.0, false};
    }

    return parseJSONResponse(response);
}

} // namespace DPI
