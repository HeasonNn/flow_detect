#pragma once

#include "../feature/flow_feature.hpp"
#include "../feature/graph_features.hpp"
#include "algorithm.hpp"

#include <vector>
#include <string>

using namespace std;

class Detector
{
private:
    // std::unique_ptr<DBSCANAlgorithm> dbscan;
    // std::unique_ptr<IsolationForestAlgorithm> iforest;
    
    shared_ptr<FlowFeatureExtractor> flowExtractor;
    shared_ptr<GraphFeatureExtractor> graphExtractor;
    
public:
    AlgorithmType algorithmType;

    virtual void addSample(const arma::vec &x, size_t label);
    virtual void train(void);
    size_t predict(const arma::vec &x) override ;

    Detector(AlgorithmType algoType);

    void run_detection(const std::string& data_path,
                       const std::vector<std::pair<FlowRecord, size_t>>& test_flows);

    void setAlgorithm(AlgorithmType newAlgoType);
};


class RfDetector : public Detector {
private:
    std::vector<arma::vec> sample_vecs;
    std::vector<size_t> labels;
    mlpack::RandomForest<> rf;
    bool trained = false;

public:
    void addSample(const arma::vec &x, size_t label);
    void train() override;
    size_t predict(const arma::vec &x) override ;
    void printFeatures() const;
};
