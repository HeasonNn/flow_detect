#include "data_loader.hpp"

shared_ptr<DataLoader> createDataLoader(const json& config)
{
    const string& data_type = config.value("data_type",  "");
    const string& data_path = config.value("data_path",  "");
    const string& label_path = config.value("label_path", "");
    const int data_size = config.value("data_size", -1);
    const double train_ratio = config.value("train_ratio", 0.8);
    
    if (data_type == "CICIDS") {
        return make_shared<CICIDSLoader>(data_path, label_path, train_ratio, data_size);
    }
    else if (data_type == "HyperVision") {
        return std::make_shared<HyperVisonLoader>(data_path, label_path, train_ratio, data_size);
    }
    else if (data_type == "Pcap") {
        return std::make_shared<PcapLoader>(data_path, label_path, train_ratio, data_size);
    }
    else {
        throw std::invalid_argument("Unknown data type: " + data_type);
    }
    
    return nullptr;
}


const auto DataLoader::getDataFileBaseName() const -> decltype(data_path_) {
    const auto& path = getDataPath();
    
    size_t last_slash = path.find_last_of('/');
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);

    size_t last_dot = filename.find_last_of('.');
    return (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);
};