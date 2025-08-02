#pragma once
#include <armadillo>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <iostream>
#include <cmath> // 用于 std::sqrt

#include <mlpack/core.hpp>
#include <mlpack/methods/range_search/range_search.hpp>

namespace stream {

struct MicroCluster {
    arma::vec center;
    double    weight;
    double    lastUpdate;

    MicroCluster(arma::vec pt, double ts, double w = 1.0)
        : center(std::move(pt)), weight(w), lastUpdate(ts) {}
};

class DBSTREAM {
public:
    DBSTREAM(double epsilon,
             double lambda,
             double mu,
             double beta_merge,
             double beta_noise,
             size_t max_clusters = 1000)
        : epsilon_(epsilon),
          epsilon_sq_(epsilon * epsilon),
          lambda_(lambda),
          mu_(mu),
          beta_merge_(beta_merge),
          beta_noise_(beta_noise),
          max_clusters_(max_clusters),
          last_seen_ts_(0.0),
          dim_(0),
          kdtree_valid_(false)
    {
        if (epsilon <= 0 || lambda <= 0 || mu <= 0 ||
            beta_merge <= 0 || beta_noise <= 0 || max_clusters == 0) {
            throw std::invalid_argument("DBSTREAM: All parameters must be > 0");
        }
    }

    void Insert(const arma::vec& point, double timestamp) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (point.n_elem == 0) {
            std::cerr << "DBSTREAM: Error: Invalid point with zero dimension.\n";
            return;
        }
        if (dim_ == 0) {
            dim_ = point.n_elem;
        } else if (point.n_elem != dim_) {
            throw std::invalid_argument("DBSTREAM: Data point dimension mismatch");
        }

        if (timestamp < last_seen_ts_) {
            std::cerr << "DBSTREAM: Warning: Non-monotonic timestamp " << timestamp
                      << " (last: " << last_seen_ts_ << "). Clamped.\n";
            timestamp = last_seen_ts_;
        } else {
            last_seen_ts_ = timestamp;
        }

        // 1. Try to absorb into existing micro-cluster
        bool absorbed = false;
        for (auto& mc : clusters_) {
            double dist_sq = arma::dot(point - mc.center, point - mc.center);
            if (dist_sq <= epsilon_sq_) {
                mc.center = (mc.center * mc.weight + point) / (mc.weight + 1.0);
                mc.weight += 1.0;
                mc.lastUpdate = timestamp;
                absorbed = true;
                break;
            }
        }

        // 2. Create new micro-cluster if not absorbed
        if (!absorbed) {
            if (clusters_.size() >= max_clusters_) {
                RemoveNoise(timestamp);
                if (clusters_.size() >= max_clusters_) {
                    if (!ForceMergeClosest(timestamp)) {
                        std::cerr << "DBSTREAM: Warning: Max clusters (" << max_clusters_
                                  << ") reached. Dropping new outlier point.\n";
                        return;
                    }
                }
            }
            clusters_.emplace_back(point, timestamp, 1.0);
            kdtree_valid_ = false; // 新增簇，KDTree失效
        }

        // ✅ 修复3: 每次插入后立即合并 (但使用高效的KD-Tree)
        MergeClusters(timestamp); // 不再使用 merge_frequency_

        // 4. Remove noise
        if (clusters_.size() > max_clusters_ * 0.8) {
            RemoveNoise(timestamp);
        }
    }

    std::vector<MicroCluster> GetClusters(double timestamp) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<MicroCluster> active;
        active.reserve(clusters_.size());
        for (const auto& mc : clusters_) {
            if (DecayedWeight(mc, timestamp) >= mu_) {
                active.emplace_back(mc);
            }
        }
        return active;
    }

    std::size_t ClusterCount(double timestamp) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::count_if(clusters_.begin(), clusters_.end(),
                             [&](const MicroCluster& mc) {
                                 return DecayedWeight(mc, timestamp) >= mu_;
                             });
    }

    std::vector<MicroCluster> AllMicroClusters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return clusters_;
    }

    std::size_t TotalMicroClusters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return clusters_.size();
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        clusters_.clear();
        last_seen_ts_ = 0.0;
        dim_ = 0;
    }

    double QueryDecayedWeight(const MicroCluster& cluster, double timestamp) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return DecayedWeight(cluster, timestamp);
    }

private:
    /*==================== Parameters ====================*/
    double epsilon_;
    double epsilon_sq_;
    double lambda_;
    double mu_;
    double beta_merge_;
    double beta_noise_;
    size_t max_clusters_;
    double last_seen_ts_;
    size_t dim_;

    std::vector<MicroCluster> clusters_;
    mutable std::mutex mutex_;

    // ✅ KDTree 优化所需的数据结构
    mutable arma::mat kdtree_data_;        
    mutable bool kdtree_valid_;           

    /*==================== KDTree Management ====================*/
    void BuildKDTree() const {
        if (clusters_.empty()) {
            kdtree_valid_ = true;
            return;
        }

        kdtree_data_.set_size(dim_, clusters_.size());
        for (size_t i = 0; i < clusters_.size(); ++i) {
            kdtree_data_.col(i) = clusters_[i].center;
        }

        kdtree_valid_ = true;
    }

    void EnsureKDTreeUpToDate() const {
        if (!kdtree_valid_) {
            BuildKDTree();
        }
    }

    /*==================== Utility Functions ====================*/

    double DecayedWeight(const MicroCluster& mc, double now) const noexcept {
        double age = now - mc.lastUpdate;
        if (age < 0) return mc.weight;
        return mc.weight * std::exp(-lambda_ * age);
    }

    // ✅ 修复1: 使用最小值作为共享密度
    double SharedDensity(const MicroCluster& a,
                         const MicroCluster& b,
                         double now) const noexcept {
        arma::vec diff = a.center - b.center;
        double dist_sq = arma::dot(diff, diff);
        if (dist_sq > epsilon_sq_) return 0.0;
        double wA = DecayedWeight(a, now);
        double wB = DecayedWeight(b, now);
        return std::min(wA, wB); // 使用最小值
    }

    // ✅ 修复2: 合并时使用衰减权重
    MicroCluster Merge(const MicroCluster& a,
                       const MicroCluster& b,
                       double now) const {
        double wA = DecayedWeight(a, now);
        double wB = DecayedWeight(b, now);
        arma::vec newCenter = (a.center * wA + b.center * wB) / (wA + wB);
        double newRawWeight = a.weight + b.weight;
        return MicroCluster(std::move(newCenter), now, newRawWeight);
    }

    // ✅ 优化后的 MergeClusters：使用 mlpack::RangeSearch 进行范围搜索
    void MergeClusters(double now) {
        if (clusters_.size() < 2) return;
        EnsureKDTreeUpToDate();

        mlpack::RangeSearch<> rangeSearch(kdtree_data_, true, false); // singleMode=true
        mlpack::Range queryRange(0.0, epsilon_);
        std::vector<std::pair<size_t, size_t>> to_merge;

        for (size_t i = 0; i < clusters_.size(); ++i) {
            arma::vec query_point = clusters_[i].center;

            std::vector<std::vector<size_t>> neighbors;
            std::vector<std::vector<double>> distances;

            rangeSearch.Search(query_point, queryRange, neighbors, distances);
            if (!neighbors.empty() && !neighbors[0].empty()) {
                for (size_t neighbor_idx : neighbors[0]) {
                    size_t j = neighbor_idx;
                    if (i == j) continue;
                    if (i < j) {
                        double density = SharedDensity(clusters_[i], clusters_[j], now);
                        if (density >= beta_merge_) {
                            // 额外检查：至少一个簇是核心簇（可选，增强稳健性）
                            if (DecayedWeight(clusters_[i], now) >= mu_ || DecayedWeight(clusters_[j], now) >= mu_) {
                                to_merge.emplace_back(i, j);
                            }
                        }
                    }
                }
            }
        }

        std::vector<bool> merged(clusters_.size(), false);
        bool any_merged = false;

        for (const auto& [i, j] : to_merge) {
            if (merged[i] || merged[j]) continue;

            clusters_[i] = Merge(clusters_[i], clusters_[j], now);
            clusters_[j] = std::move(clusters_.back());
            clusters_.pop_back();
            merged[j] = true;
            any_merged = true;
        }

        if (any_merged) {
            kdtree_valid_ = false;
        }
    }

    void RemoveNoise(double now) {
        auto pred = [&](const MicroCluster& mc) {
            return DecayedWeight(mc, now) < beta_noise_;
        };
        auto new_end = std::remove_if(clusters_.begin(), clusters_.end(), pred);
        bool removed = new_end != clusters_.end();
        clusters_.erase(new_end, clusters_.end());
        if (removed) {
            kdtree_valid_ = false;
        }
    }

    bool ForceMergeClosest(double now) {
        if (clusters_.size() < 2) return false;
        EnsureKDTreeUpToDate();

        size_t best_i = 0, best_j = 1;
        double max_density = SharedDensity(clusters_[0], clusters_[1], now);

        for (size_t i = 0; i < clusters_.size(); ++i) {
            for (size_t j = i + 1; j < clusters_.size(); ++j) {
                double density = SharedDensity(clusters_[i], clusters_[j], now);
                if (density > max_density) {
                    max_density = density;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (max_density > 0.0) {
            clusters_[best_i] = Merge(clusters_[best_i], clusters_[best_j], now);
            clusters_[best_j] = std::move(clusters_.back());
            clusters_.pop_back();
            kdtree_valid_ = false;
            return true;
        }
        return false;
    }
};

} // namespace stream