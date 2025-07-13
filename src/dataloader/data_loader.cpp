#include "data_loader.hpp"

shared_ptr<DataLoader> createDataLoader(const json& config_j)
{
    const json& data_loader = config_j["data_loader"];
    const json& detector = config_j["detector"];

    const string& data_type = data_loader.value("data_type",  "");
    const string& data_path = data_loader.value("data_path",  "");
    const string& label_path = data_loader.value("label_path", "");
    const int data_size = data_loader.value("data_size", -1);
    const double train_ratio = data_loader.value("train_ratio", 0.8);
    
    const double epsilon = detector.value("epsilon", 0.1);
    const double min_points = detector.value("min_points", 0.1);
    
    if (data_type == "CICIDS") {
        return make_shared<CICIDSLoader>(data_path, data_size);
    }
    else if (data_type == "HyperVision") {
        if (label_path.empty()) {
            std::cerr << "[DataLoader] Error: label_path is required for HyperVison dataset.\n";
            return nullptr;
        }
        return std::make_shared<HyperVisonLoader>(data_path, label_path, train_ratio, data_size);
    }
    
    else {
        throw std::invalid_argument("Unknown data type: " + data_type);
    }
    
    return nullptr;
}
