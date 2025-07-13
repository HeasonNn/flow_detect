#pragma once

#include <mlpack/core.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <armadillo>
#include <random>

struct MiniBatchKMeansConfig {
    size_t k_ = 10;
    size_t batch_size_ = 64;
    size_t max_iters_ = 200;
    double decay_rate_ = 0.8;
    bool verbose_ = true;
};


class MiniBatchKMeans {
private:
    size_t k_;
    size_t batch_size_;
    size_t max_iters_;
    double decay_rate_;

    mlpack::data::MinMaxScaler scaler_;
    arma::mat centroids_;

    bool verbose_;
public:
    MiniBatchKMeans(size_t k, size_t batch_size, size_t max_iters, double decay_rate, bool verbose)
        : k_(k), batch_size_(batch_size), max_iters_(max_iters), decay_rate_(decay_rate), verbose_(verbose) {};

    MiniBatchKMeans(const MiniBatchKMeansConfig& config)
        : MiniBatchKMeans(config.k_, config.batch_size_, config.max_iters_, config.decay_rate_, config.verbose_) {};
        
    size_t K() const { return k_; }
    const arma::vec Centroid(size_t c) const { return centroids_.col(c); }
    mlpack::data::MinMaxScaler& Scaler() { return scaler_; }

    void Train(const arma::mat& data);
    std::pair<size_t, double> Predict(const arma::vec& X);
};