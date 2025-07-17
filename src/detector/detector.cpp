#include "detector.hpp"


string get_current_time_str(void) {
    auto now = chrono::system_clock::now();
    auto now_time_t = chrono::system_clock::to_time_t(now);
    
    char buf[100];
    strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", localtime(&now_time_t));
    
    return string(buf);
}


shared_ptr<Detector> createDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader,
                                    const json& config_j)
{
    const string& algorithm = config_j["algorithm"];
    const json& detector = config_j["detector"];

    if (algorithm == "RF") {
        return make_shared<RFDetector>(flowExtractor, graphExtractor, loader);
    }
    else if (algorithm == "DBSCAN") {
        return make_shared<DBscanDetector>(flowExtractor, graphExtractor, loader, detector);
    }
    else if (algorithm == "Mini_Batch_KMeans") {
        return make_shared<MiniBatchKMeansDetector>(flowExtractor, graphExtractor, loader);
    }
    else {
        throw std::invalid_argument("Unknown algorithm type: " + algorithm);
    }
    
    return nullptr;
}


void Detector::Start() {
    // using FlowVec = vector<shared_ptr<pair<FlowRecord, size_t>>>;

    // if(loader_ == nullptr) {
    //     cerr << "Loader is not initialized!" <<"\n";
    // }

    // loader_->Load();
    // const auto& all_flow = loader_->getAllData();
    // auto cur_data = make_shared<FlowVec>();


    // for(size_t idx = 0; idx < all_flow->size(); idx++){
    //     if(cur_data->size() == window_size_){
    //         run();
    //         (*cur_data).clear();
    //     }
    //     cur_data->emplace_back(make_shared<decltype((*all_flow)[idx])>((*all_flow)[idx]));
    //     idx++;
    // }
}