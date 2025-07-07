#include "dataloader/data_loader.hpp"
#include "feature/flow_feature.hpp"
#include "feature/graph_features.hpp"
#include "detector/detector.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <data_path>\n";
        return 1;
    }

    __START_FTIMMER__

    std::string algorithm = argv[1];
    std::string data_path = argv[2];
    std::string label_path = argv[3];

    auto flowExtractor = make_shared<FlowFeatureExtractor>();
    auto graphExtractor = make_shared<GraphFeatureExtractor>();
    
    // auto loader = make_shared<CICIDSLoader>(data_path);
    auto loader = make_shared<HyperVisonLoader>(data_path, label_path);
    auto detector = createDetector(algorithm, flowExtractor, graphExtractor, loader);

    detector->run();

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    return 0;
}
