#include "dataloader/cicids_loader.hpp"
#include "dataloader/hypervison_loader.hpp"
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

    std::string data_path = argv[1];
    std::string label_path = argv[2];

    FlowFeatureExtractor flowExtractor;
    GraphFeatureExtractor graphExtractor;
    Detector detector(AlgorithmType::RandomForest);

    // auto loader = make_shared<CICIDSLoader>(data_path);
    auto loader = make_shared<HyperVisonLoader>(data_path, label_path);
    loader->load();

    const auto train_flows = *loader->getTrainData();
    const auto test_flows = *loader->getTestData();

    // ✅ 训练阶段
    size_t total = train_flows.size();
    size_t count = 0;
    size_t print_interval = 1000; // 每1000个样本打印一次

    for (const auto &[flow, label] : train_flows)
    {
        graphExtractor.updateGraph(flow.src_ip, flow.dst_ip);
        arma::vec flowVec = flowExtractor.extract(flow);
        arma::vec graphVec = graphExtractor.extract(flow.src_ip, flow.dst_ip);

        if (flowVec.is_empty() || graphVec.is_empty())
            continue;

        rf.addSample(arma::join_vert(flowVec, graphVec), label);

        if (++count % print_interval == 0 || count == total)
        {
            std::cout << "\rProcessed " << count << " / " << total << " samples." << std::flush;
        }
    }
    std::cout << std::endl; 

    cout << " 🔄 Start train: " << "\n";
    rf.train();

    // ✅ 测试阶段：调用封装函数
    run_detection(data_path, test_flows, flowExtractor, graphExtractor, rf);

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    // rf.printFeatures();

    return 0;
}
