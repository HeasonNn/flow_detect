#include "detector.hpp"

shared_ptr<Detector> createDetector(const std::string& algorithm, 
                                    shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader) 
{
    if (algorithm == "RF") {
        return make_shared<RFDetector>(flowExtractor, graphExtractor, loader);
    }
    else if (algorithm == "DBSCAN") {
        return make_shared<DBscanDetector>(flowExtractor, graphExtractor, loader);
    }
    else if (algorithm == "GNN") {
        // TODO:
        //  GNN 子类的创建
    }
    else {
        throw std::invalid_argument("Unknown algorithm type: " + algorithm);
    }
    
    return nullptr;
}

string get_current_time_str(void) {
    auto now = chrono::system_clock::now();
    auto now_time_t = chrono::system_clock::to_time_t(now);
    
    char buf[100];
    strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", localtime(&now_time_t));
    
    return string(buf);
}