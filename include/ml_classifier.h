#ifndef ML_CLASSIFIER_H
#define ML_CLASSIFIER_H

#include <string>
#include "ml_feature_exporter.h"

namespace DPI {

struct MLPrediction {
    std::string prediction = "unknown";
    double confidence = 0.0;
    bool is_anomaly = false;
};

class MLClassifier {
public:
    MLClassifier(const std::string& host = "127.0.0.1", int port = 8080);
    ~MLClassifier();

    MLPrediction predict(const FlowFeatures& features);

private:
    std::string host_;
    int port_;
    bool ws_initialized_ = false;

    void initSockets();
    void cleanupSockets();
    std::string sendHTTPPost(const std::string& path, const std::string& json_payload);
    MLPrediction parseJSONResponse(const std::string& response);
};

} // namespace DPI

#endif // ML_CLASSIFIER_H
