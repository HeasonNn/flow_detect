#include "detector.hpp"

shared_ptr<Detector> createDetector(const std::string& algorithm, 
                                    shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader) 
{
    if (algorithm == "RF") {
        return make_shared<RfDetector>(flowExtractor, graphExtractor, loader);
    }
    else if (algorithm == "SVM") {
        // TODO:
        //  SVM 子类的创建
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