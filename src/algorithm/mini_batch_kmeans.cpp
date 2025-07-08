#include "mini_batch_kmeans.hpp"

#include <chrono>
#include <iomanip>

void MiniBatchKMeans::Train(const mat& data) {
    const size_t n = data.n_cols;
    centroids.set_size(data.n_rows, k);

    arma::mat norm_data;
    scaler.Fit(data);  // Fit the scaler to the data
    scaler.Transform(data, norm_data);  // Apply the transformation

    // ===== 使用 mlpack 自带的 kmeans++ 初始化中心 =====
    KMeans<> kmeansInit;
    Row<size_t> assignments;
    kmeansInit.Cluster(norm_data, k, assignments, centroids);

    using clock = std::chrono::steady_clock;
    auto start_time = clock::now();

    // 记录开始时间
    size_t processed_samples = 0;

    // ===== Mini-Batch 迭代 =====
    for (size_t iter = 0; iter < maxIterations; ++iter) {
        auto iter_start = clock::now();

        // —— 随机采样一个 mini-batch ——
        uvec indices = randi<uvec>(batchSize, distr_param(0, n - 1));
        mat batch = norm_data.cols(indices);
        processed_samples += batchSize;

        Row<size_t> batchAssignments(batchSize);
        for (size_t i = 0; i < batchSize; ++i) {
            vec point = batch.col(i);
            double minDist = datum::inf;
            size_t bestCluster = 0;
            for (size_t j = 0; j < k; ++j) {
                double dist = norm(point - centroids.col(j), 2);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = j;
                }
            }
            batchAssignments[i] = bestCluster;
        }

        mat oldCentroids = centroids;

        for (size_t i = 0; i < batchSize; ++i) {
            size_t cluster = batchAssignments[i];
            vec oldCenter = centroids.col(cluster);
            centroids.col(cluster) = decayRate * oldCenter + (1 - decayRate) * batch.col(i);
        }

        // === 打印进度信息 ===
        if (verbose && iter % 10 == 0) {
            double drift = accu(sqrt(sum(square(centroids - oldCentroids), 0)));

            double avg_intra_dist = 0.0;
            for (size_t i = 0; i < batchSize; ++i) {
                size_t cluster = batchAssignments[i];
                avg_intra_dist += norm(batch.col(i) - centroids.col(cluster), 2);
            }
            avg_intra_dist /= batchSize;

            // 计算估算的 ETA
            auto now = clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            double iter_avg = static_cast<double>(elapsed) / (processed_samples + 1);
            double eta = iter_avg * (n - processed_samples);

            // 输出每次迭代的训练进度
            std::cout << "🔁 Iteration " << std::setw(4) << iter
                      << " | Drift: " << std::fixed << std::setprecision(5) << drift
                      << " | AvgDist: " << std::fixed << std::setprecision(5) << avg_intra_dist
                      << " | ETA: " << std::setw(4) << static_cast<int>(eta) << "s"
                      << std::flush << std::endl;
        }

        // 防止过多输出导致耗时
        if (iter % 50 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 短暂暂停
        }
    }
}

size_t MiniBatchKMeans::Predict(const arma::vec& X) {

    arma::vec norm_X;
    scaler.Transform(X, norm_X);

    double minDist = arma::datum::inf;
    size_t bestCluster = 0;

    for (size_t j = 0; j < k; ++j) {
        double dist = arma::norm(X - centroids.col(j), 2);
        if (dist < minDist) {
            minDist = dist;
            bestCluster = j;
        }
    }

    return bestCluster;
}