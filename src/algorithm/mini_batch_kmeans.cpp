#include "mini_batch_kmeans.hpp"

#include <chrono>
#include <iomanip>

void MiniBatchKMeans::Train(const arma::mat& data) {
    const size_t n = data.n_cols;
    centroids_.set_size(data.n_rows, k_);

    arma::mat norm_data;
    scaler_.Fit(data);
    scaler_.Transform(data, norm_data);

    mlpack::KMeans<> kmeansInit;
    arma::Row<size_t> assignments;
    kmeansInit.Cluster(norm_data, k_, assignments, centroids_);

    using clock = std::chrono::steady_clock;
    auto start_time = clock::now();

    size_t processed_samples = 0;

    for (size_t iter = 0; iter < max_iters_; ++iter) {
        auto iter_start = clock::now();

        arma::uvec indices = arma::randi<arma::uvec>(batch_size_, arma::distr_param(0, n - 1));
        arma::mat batch = norm_data.cols(indices);
        processed_samples += batch_size_;

        arma::Row<size_t> batchAssignments(batch_size_);
        for (size_t i = 0; i < batch_size_; ++i) {
            arma::vec point = batch.col(i);
            double minDist = arma::datum::inf;
            size_t bestCluster = 0;
            for (size_t j = 0; j < k_; ++j) {
                double dist = arma::norm(point - centroids_.col(j), 2);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }
            batchAssignments[i] = bestCluster;
        }

        arma::mat oldCentroids = centroids_;

        for (size_t i = 0; i < batch_size_; ++i) {
            size_t cluster = batchAssignments[i];
            arma::vec oldCenter = centroids_.col(cluster);
            centroids_.col(cluster) = decay_rate_ * oldCenter + (1 - decay_rate_) * batch.col(i);
        }
    }
}


std::pair<size_t, double> MiniBatchKMeans::Predict(const arma::vec& X) {
    arma::vec norm_X;
    scaler_.Transform(X, norm_X);
    double minDist = arma::datum::inf;
    size_t bestCluster = 0;

    for (size_t j = 0; j < k_; ++j) {
        double dist = arma::norm(norm_X - centroids_.col(j), 2);
        if (dist < minDist) {
            minDist = dist;
            bestCluster = j;
        }
    }

    return {bestCluster, minDist};
}