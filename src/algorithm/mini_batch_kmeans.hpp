#pragma once

#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>
#include <random>

using namespace mlpack;
using namespace arma;

class MiniBatchKMeans {
private:
    size_t k;
    size_t batchSize;
    size_t maxIterations;
    double decayRate;
    
    mlpack::data::MinMaxScaler scaler;
    mat centroids;

    bool verbose;
public:
    MiniBatchKMeans(size_t clusters, size_t batchSize = 100, size_t maxIters = 1000, double decay = 0.9, bool verbose = false)
        : k(clusters), batchSize(batchSize), maxIterations(maxIters), decayRate(decay), verbose(verbose) {};
    
    void Train(const mat& data);
    const mat& Centroids() const { return centroids; };
    size_t Predict(const arma::vec& X);
};