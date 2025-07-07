//detector.hpp
#pragma once

#include "../feature/flow_feature.hpp"
#include "../feature/graph_features.hpp"
#include "../dataloader/data_loader.hpp"

#include <armadillo>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <vector>
#include <string>

using namespace std;

class Detector
{
protected:
    shared_ptr<FlowFeatureExtractor> flowExtractor_;
    shared_ptr<GraphFeatureExtractor> graphExtractor_;
    shared_ptr<DataLoader> loader_; 

public:

    explicit Detector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                      shared_ptr<GraphFeatureExtractor> graphExtractor,
                      shared_ptr<DataLoader> loader)
    : flowExtractor_(flowExtractor), 
      graphExtractor_(graphExtractor),
      loader_(loader) {
        loader_->load();
      };

    virtual void run() = 0;
    virtual ~Detector() = default;
};


class RfDetector : public Detector {
private:
    vector<arma::vec> sample_vecs_;
    vector<size_t> labels_;
    mlpack::RandomForest<> rf_;

    bool trained_ = false;

    void addSample(const arma::vec &x, size_t label) ;
    void train(void);
    size_t predict(const arma::vec &x);

    void run_detection(void);
    void printFeatures(void) const;

public:
    explicit RfDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                        shared_ptr<GraphFeatureExtractor> graphExtractor,
                        shared_ptr<DataLoader> loader)
    : Detector(flowExtractor, graphExtractor, loader) {};

    void run(void) override;
};


shared_ptr<Detector> createDetector(const std::string& algorithm, 
                                    shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader);