// algorithm.hpp
#pragma once

#include <armadillo>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <vector>

enum AlgorithmType { 
    RandomForest, 
    DBSCAN, 
    IsolationForest 
};

class DetectionAlgorithm {
public:
    virtual ~DetectionAlgorithm() = default;
    virtual void train();
    virtual size_t predict(const arma::vec& feat) = 0;
};


class RandomForestAlgorithm : public DetectionAlgorithm {
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
