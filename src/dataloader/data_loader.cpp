#include "data_loader.hpp"

shared_ptr<DataLoader> createDataLoader(
    const string& data_type, 
    const string& data_path, 
    const string& label_path,
    const double train_ratio
)
{
    if (data_type == "CICIDS") {
        return make_shared<CICIDSLoader>(data_path);
    }
    else if (data_type == "HyperVision") {
        if (label_path.empty()) {
            std::cerr << "[DataLoader] Error: label_path is required for HyperVison dataset.\n";
            return nullptr;
        }
        return std::make_shared<HyperVisonLoader>(data_path, label_path, train_ratio);
    }
    
    else {
        throw std::invalid_argument("Unknown data type: " + data_type);
    }
    
    return nullptr;
}
