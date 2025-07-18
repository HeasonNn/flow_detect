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
        ifstream fin(FLAGS_config);
        if (!fin.is_open()) {
            FATAL_ERROR("Cannot open config file: " + FLAGS_config);
        }
        fin >> config_j;
    } catch (const exception& e) {
        FATAL_ERROR(string("Failed to load or parse config: ") + e.what());
    }
    
    const string algorithm = config_j.value("algorithm",  "");
    const string data_type = config_j.value("data_type",  "");
    const string data_path = config_j.value("data_path",  "");
    const string label_path = config_j.value("label_path", "");
    const double train_ratio = config_j.value("train_ratio", 0.8);
    const int data_size = config_j.value("data_size", -1);

    auto flowExtractor = make_shared<FlowFeatureExtractor>();
    auto graphExtractor = make_shared<GraphFeatureExtractor>();

    auto loader = createDataLoader(config_j);
    auto detector = createDetector(flowExtractor, graphExtractor, loader, config_j);

    detector->run();

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    return 0;
}
