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

    auto detector = createDetector(config_j);
    detector->run();

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    return 0;
}
