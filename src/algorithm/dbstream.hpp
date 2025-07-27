#pragma once

#include <armadillo>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <iostream>

namespace stream {

/**
 * @brief Micro-cluster structure for summarizing local density in data streams
 */
struct MicroCluster
{
    arma::vec center;     // Centroid (weighted average position)
    double    weight;     // Raw weight (cumulative un-decayed "equivalent points")
    double    lastUpdate; // Timestamp of last update

    /**
     * @param pt        Data point
     * @param ts        Timestamp
     * @param w         Initial weight (default 1.0)
     */
    MicroCluster(arma::vec pt, double ts, double w = 1.0)
        : center(std::move(pt)), weight(w), lastUpdate(ts) {}
};

/**
 * @class DBSTREAM
 * @brief Density-based streaming clustering algorithm
 *
 * Key Features:
 * - Supports discovery of arbitrarily shaped clusters
 * - Handles noise and outliers
 * - Adapts to concept drift (time decay)
 * - Single-pass, memory-controlled, thread-safe
 *
 * Reference: Ntoutsi et al., "Density-based clustering of data streams", DEXA 2012
 */
class DBSTREAM
{
public:
    /**
     * @param epsilon       Neighborhood radius (distance threshold)
     * @param lambda        Time decay rate λ > 0 (higher means faster forgetting)
     * @param mu            Core cluster threshold: decayed weight >= μ to be valid
     * @param beta_merge    Cluster merge threshold: shared density >= beta_merge to merge
     * @param beta_noise    Noise cluster removal threshold: decayed weight < beta_noise to remove
     * @param max_clusters  Maximum number of micro-clusters (memory control)
     */
    DBSTREAM(double epsilon,
             double lambda,
             double mu,
             double beta_merge,
             double beta_noise,
             size_t max_clusters = 1000)
        : epsilon_(epsilon),
          lambda_(lambda),
          mu_(mu),
          beta_merge_(beta_merge),
          beta_noise_(beta_noise),
          max_clusters_(max_clusters),
          last_seen_ts_(0.0),
          dim_(0)
    {
        if (epsilon <= 0 || lambda <= 0 || mu <= 0 ||
            beta_merge <= 0 || beta_noise <= 0 || max_clusters == 0) {
            throw std::invalid_argument("DBSTREAM: All parameters must be > 0");
        }
    }

    /*--------------------------------------------------------------
     * Insert a data point
     *--------------------------------------------------------------*/
    void Insert(const arma::vec& point, double timestamp)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            if (point.n_elem == 0) {
                return;
            }

            if (dim_ == 0) {
                dim_ = point.n_elem;
            } else if (point.n_elem != dim_) {
                throw std::invalid_argument("DBSTREAM: Data point dimension mismatch");
            }

            // Handle non-monotonic timestamps
            if (timestamp < last_seen_ts_) {
                std::cerr << "DBSTREAM: Warning: Non-monotonic timestamp " << timestamp
                          << " (last: " << last_seen_ts_ << "). Clamped.\n";
                timestamp = last_seen_ts_;
            } else {
                last_seen_ts_ = timestamp;
            }

            // 1. Try to absorb into existing micro-cluster
            bool absorbed = false;
            for (auto& mc : clusters_)
            {
                double dist = arma::norm(point - mc.center);
                if (dist <= epsilon_)
                {
                    double w_effective = DecayedWeight(mc, timestamp);
                    mc.center = (mc.center * w_effective + point) / (w_effective + 1.0);
                    mc.weight += 1.0;
                    mc.lastUpdate = timestamp;
                    absorbed = true;
                    break;
                }
            }

            // 2. If not absorbed, create new micro-cluster (with memory control)
            if (!absorbed) {
                // Try to free space
                if (clusters_.size() >= max_clusters_) {
                    RemoveNoise(timestamp);
                    if (clusters_.size() >= max_clusters_) {
                        // Optional: Force merge weakest pair
                        // For now, drop outlier
                        std::cerr << "DBSTREAM: Warning: Max clusters (" << max_clusters_
                                  << ") reached. Dropping new outlier point.\n";
                        return;
                    }
                }
                clusters_.emplace_back(point, timestamp, 1.0);
            }

            // 3. Merge micro-clusters (safe O(n²) with early restart)
            MergeClusters(timestamp);

            // 4. Remove noise (more aggressive)
            if (clusters_.size() > max_clusters_ * 0.8 ||  // >80% full
                clusters_.size() >= 100 && clusters_.size() % 50 == 0) {  // every 50 after 100
                RemoveNoise(timestamp);
            }

        } catch (const std::exception& e) {
            std::cerr << "DBSTREAM: Exception in Insert: " << e.what() << std::endl;
        }
    }

    /*--------------------------------------------------------------
     * Query Interfaces
     *--------------------------------------------------------------*/

    /**
     * @brief Get current valid core clusters (decayed weight >= mu)
     */
    std::vector<MicroCluster> GetClusters(double timestamp) const
    {
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

    /**
     * @brief Get current core cluster count
     */
    std::size_t ClusterCount(double timestamp) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::count_if(clusters_.begin(), clusters_.end(),
                             [&](const MicroCluster& mc) {
                                 return DecayedWeight(mc, timestamp) >= mu_;
                             });
    }

    /**
     * @brief Get all micro-clusters (including noise, for debugging)
     */
    const std::vector<MicroCluster>& AllMicroClusters() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return clusters_;
    }

    /**
     * @brief Get total micro-cluster count (including noise)
     */
    std::size_t TotalMicroClusters() const noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return clusters_.size();
    }

    /**
     * @brief Reset all state
     */
    void Reset() noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        clusters_.clear();
        last_seen_ts_ = 0.0;
        dim_ = 0;
    }

private:
    /*==================== Parameters ====================*/
    double epsilon_;       // Neighborhood radius
    double lambda_;        // Decay rate
    double mu_;            // Core cluster threshold
    double beta_merge_;    // Merge threshold
    double beta_noise_;    // Noise removal threshold
    size_t max_clusters_;  // Maximum micro-cluster count
    double last_seen_ts_;  // Latest timestamp
    size_t dim_;           // Data dimension

    std::vector<MicroCluster> clusters_;
    mutable std::mutex mutex_;  // Thread safety

    /*==================== Utility Functions ====================*/

    /**
     * @brief Calculate decayed weight of a micro-cluster at current time
     */
    double DecayedWeight(const MicroCluster& mc, double now) const noexcept
    {
        double age = now - mc.lastUpdate;
        if (age < 0) return mc.weight;  // Safety
        return mc.weight * std::exp(-lambda_ * age);
    }

    /**
     * @brief Calculate shared density between two micro-clusters
     *        Defined as average of decayed weights (if within epsilon)
     */
    double SharedDensity(const MicroCluster& a,
                         const MicroCluster& b,
                         double now) const noexcept
    {
        double dist = arma::norm(a.center - b.center);
        if (dist > epsilon_) return 0.0;

        double wA = DecayedWeight(a, now);
        double wB = DecayedWeight(b, now);
        return 0.5 * (wA + wB);
    }

    /**
     * @brief Merge two micro-clusters into one
     */
    MicroCluster Merge(const MicroCluster& a,
                       const MicroCluster& b,
                       double now) const
    {
        double wA = DecayedWeight(a, now);
        double wB = DecayedWeight(b, now);

        arma::vec newCenter = (a.center * wA + b.center * wB) / (wA + wB);
        double newRawWeight = a.weight + b.weight;

        return MicroCluster(std::move(newCenter), now, newRawWeight);
    }

    /**
     * @brief Perform micro-cluster merging
     *        Uses O(n²) with early restart to avoid iterator invalidation
     */
    void MergeClusters(double now)
    {
        bool changed;
        do {
            changed = false;
            for (size_t i = 0; i < clusters_.size(); ++i)
            {
                for (size_t j = i + 1; j < clusters_.size(); )
                {
                    if (SharedDensity(clusters_[i], clusters_[j], now) >= beta_merge_)
                    {
                        clusters_[i] = Merge(clusters_[i], clusters_[j], now);
                        clusters_.erase(clusters_.begin() + j);
                        changed = true;
                        break;  // Restart outer loop
                    }
                    else
                    {
                        ++j;
                    }
                }
                if (changed) break;
            }
        } while (changed);
    }

    /**
     * @brief Remove noise micro-clusters (decayed weight < beta_noise)
     */
    void RemoveNoise(double now)
    {
        auto pred = [&](const MicroCluster& mc) {
            return DecayedWeight(mc, now) < beta_noise_;
        };
        clusters_.erase(
            std::remove_if(clusters_.begin(), clusters_.end(), pred),
            clusters_.end()
        );
    }
};

} // namespace stream