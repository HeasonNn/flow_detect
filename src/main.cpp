#include <gflags/gflags.h>

#include "feature/flow_feature.hpp"
#include "feature/graph_features.hpp"
#include "dataloader/data_loader.hpp"
#include "detector/detector.hpp"


DEFINE_string(config, "../config/config.json",  "Configuration file location.");

int main(int argc, char *argv[])
{
    __START_FTIMMER__

    google::ParseCommandLineFlags(&argc, &argv, true);

    json config_j;
    try {
        std::ifstream fin(FLAGS_config);
        if (!fin.is_open()) {
            FATAL_ERROR("Cannot open config file: " + FLAGS_config);
        }
        fin >> config_j;
    } catch (const std::exception& e) {
        FATAL_ERROR(std::string("Failed to load or parse config: ") + e.what());
    }
    
    const std::string algorithm  = config_j.value("algorithm",  "");
    const std::string data_type  = config_j.value("data_type",  "");
    const std::string data_path  = config_j.value("data_path",  "");
    const std::string label_path = config_j.value("label_path", "");

    auto flowExtractor = make_shared<FlowFeatureExtractor>();
    auto graphExtractor = make_shared<GraphFeatureExtractor>();
    
    auto loader = createDataLoader(data_type, data_path, label_path);
    auto detector = createDetector(algorithm, flowExtractor, graphExtractor, loader);

    detector->run();

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    return 0;
}
