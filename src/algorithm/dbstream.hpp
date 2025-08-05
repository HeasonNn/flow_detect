#pragma once

#include <armadillo>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <limits>
#include <map>

// ================= 微簇结构体 =================
struct MicroCluster {
    arma::vec center;
    double    weight = 0.0;
    double    lastUpdate = 0.0;

    explicit MicroCluster(const arma::vec& pt, double ts, double w = 1.0)
        : center(pt), weight(w), lastUpdate(ts) {}

    void decay(double lambda, double now) {
        double dt = now - lastUpdate;
        if (dt <= 0) return;
        double f = std::exp(-lambda * dt);
        weight *= f;
        lastUpdate = now;
    }

    double getDistance(const arma::vec& point) const {
        return arma::norm(point - center);
    }

    double getDecayedWeight(double lambda, double now) const {
        double dt = now - lastUpdate;
        return weight * std::exp(-lambda * std::max(dt, 0.0));
    }

    bool isCore(double lambda, double now, double mu) const {
        return getDecayedWeight(lambda, now) >= mu;
    }
};

// ================= 邻接表的Key与Value =================
struct MCLinkKey {
    const MicroCluster* a;
    const MicroCluster* b;
    MCLinkKey(const MicroCluster* x, const MicroCluster* y)
        : a(x < y ? x : y), b(x < y ? y : x) {}

    bool operator==(const MCLinkKey& o) const {
        return a == o.a && b == o.b;
    }
};

namespace std {
template <>
struct hash<MCLinkKey> {
    size_t operator()(const MCLinkKey& k) const noexcept {
        return std::hash<const MicroCluster*>()(k.a) ^ (std::hash<const MicroCluster*>()(k.b) << 1);
    }
};
}

struct MCLinkValue {
    double weight = 0.0;
    double connection = 0.0;
    double lastUpdate = 0.0;
};

// ================= DBSTREAM 主类 =================
class DBSTREAM {
public:
    explicit DBSTREAM(double epsilon, double lambda, double mu, double beta_noise, size_t max_clusters, double eta = 0.1)
        : epsilon_(epsilon), 
          lambda_(lambda), 
          mu_(mu), 
          beta_noise_(beta_noise), 
          max_clusters_(max_clusters), 
          eta_(eta),
          last_seen_ts_(0.0), 
          dim_(0)
    {
        if (epsilon_ <= 0 || lambda_ <= 0 || mu_ <= 0 || beta_noise_ <= 0 || max_clusters_ == 0)
            throw std::invalid_argument("DBSTREAM: All parameters must be > 0");
    }

    // ====== 主插入流程 ======
    void Insert(const arma::vec& point, double ts) {
        if (point.empty()) {
            std::cerr << "DBSTREAM: Invalid point with zero dimension.\n";
            return;
        }
        if (ts < last_seen_ts_) return;

        if (dim_ == 0) dim_ = point.n_elem;
        else if (point.n_elem != dim_) {
            throw std::invalid_argument("Dimension mismatch");
        }

        // 1. 邻域搜索
        std::vector<std::shared_ptr<MicroCluster>> N;
        for (auto& mc : clusters_) {
            mc->decay(lambda_, ts);
            if (mc->getDistance(point) < epsilon_) N.push_back(mc);
        }

        // 2. 创建新微簇或竞争学习更新
        if (N.empty()) {
            if (clusters_.size() >= max_clusters_) {
                RemoveNoise(ts);   // 先清理弱簇
                if (clusters_.size() >= max_clusters_) {
                    std::cerr << "Max clusters hit, dropping point.\n";
                    return;
                }
            }
            clusters_.emplace_back(std::make_shared<MicroCluster>(point, ts));
        } else {
            // —— 竞争学习新中心（高斯核） ——
            std::vector<arma::vec> new_centers;
            for (auto& mc : N) {
                mc->weight += 1.0;
                mc->lastUpdate = ts;
                double d = arma::norm(mc->center - point);
                double h = std::exp(-d*d / (2 * epsilon_ * epsilon_));
                new_centers.push_back(mc->center + eta_ * h * (point - mc->center));
            }
            // 检查是否塌陷（所有新中心之间距离 >= epsilon_）
            bool move_allowed = true;
            for (size_t i = 0; i < new_centers.size(); ++i)
                for (size_t j = i + 1; j < new_centers.size(); ++j)
                    if (arma::norm(new_centers[i] - new_centers[j]) < epsilon_)
                        move_allowed = false;
            // 批量move
            if (move_allowed)
                for (size_t i = 0; i < N.size(); ++i) N[i]->center = new_centers[i];
            
            // 更新共享密度图
            if (N.size() > 1) {
                for (size_t i = 0; i < N.size(); ++i)
                    for (size_t j = i + 1; j < N.size(); ++j) {
                        MCLinkKey key(N[i].get(), N[j].get());
                        auto& val = adj_[key];
                        double decay = std::exp(-lambda_ * (ts - val.lastUpdate));
                        val.weight *= decay;
                        val.weight += 1.0;
                        val.connection = 2 * (val.weight) / (N[i].get()->weight + N[j].get()->weight);
                        val.lastUpdate = ts;
                    }
            }
        }

        // Step 3: 定期清理弱簇和弱边（比如每隔20步清理一次）
        if (++insert_counter_ % 100 == 0) RemoveNoise(ts);

        last_seen_ts_ = ts;
    }

    // ====== 获取全部核心微簇 ======
    std::vector<std::shared_ptr<MicroCluster>> CoreMicroClusters(double ts) const {
        std::vector<std::shared_ptr<MicroCluster>> res;
        for (const auto& mc : clusters_)
            if (mc->isCore(lambda_, ts, mu_)) res.push_back(mc);
        return res;
    }

    // ====== 全局聚类标签输出（核心微簇间连通分量） ======
    std::map<const MicroCluster*, int> GlobalClusterLabels(double ts, double connection_threshold = 0.3) const {
        // 1. 只保留核心微簇
        std::vector<const MicroCluster*> core_mcs;
        for (const auto& mc : clusters_)
            if (mc->isCore(lambda_, ts, mu_)) core_mcs.push_back(mc.get());

        // 2. 并查集Union-Find找连通分量
        std::map<const MicroCluster*, const MicroCluster*> parent;
        for (const auto& mc : core_mcs) parent[mc] = mc;
        auto find = [&](const MicroCluster* x) {
            while (x != parent[x]) x = parent[x];
            return x;
        };
        auto unite = [&](const MicroCluster* x, const MicroCluster* y) {
            parent[find(x)] = find(y);
        };
        for (const auto& [key, val] : adj_) {
            if (val.connection >= connection_threshold &&
                parent.count(key.a) && parent.count(key.b)) {
                unite(key.a, key.b);
            }
        }
        // 3. 分配label
        std::map<const MicroCluster*, int> cluster_label;
        int next_label = 0;
        for (const auto& mc : core_mcs) {
            auto root = find(mc);
            if (!cluster_label.count(root))
                cluster_label[root] = next_label++;
        }
        std::map<const MicroCluster*, int> result;
        for (const auto& mc : core_mcs)
            result[mc] = cluster_label[find(mc)];
        return result;
    }

    // ====== 数据点归属聚类分配 ======
    int AssignCluster(const arma::vec& point, double ts, const std::map<const MicroCluster*, int>& labels, bool core_only=true) const {
        double min_dist = std::numeric_limits<double>::max();
        const MicroCluster* best = nullptr;
        for (const auto& mc : clusters_) {
            if (core_only && !mc->isCore(lambda_, ts, mu_)) continue;
            double d = mc->getDistance(point);
            if (d < min_dist) {
                min_dist = d;
                best = mc.get();
            }
        }
        // 仅当最近簇“足够近”时才归属
        if (best && labels.count(best) && min_dist <= epsilon_) return labels.at(best);
        return -1;
    }

    std::vector<std::shared_ptr<MicroCluster>> AllMicroClusters() const {
        return clusters_;
    }

    std::size_t TotalMicroClusters() const {
        return clusters_.size();
    }

    void Reset() {
        clusters_.clear();
        adj_.clear();
        last_seen_ts_ = 0.0;
        dim_ = 0;
        insert_counter_ = 0;
    }

private:
    double epsilon_, lambda_, mu_, beta_noise_, eta_;
    size_t max_clusters_;
    double last_seen_ts_;
    size_t dim_;
    size_t insert_counter_ = 0;

    std::vector<std::shared_ptr<MicroCluster>> clusters_;
    std::unordered_map<MCLinkKey, MCLinkValue> adj_;

    void RemoveNoise(double ts) {
        auto end = std::remove_if(clusters_.begin(), clusters_.end(),
            [&](const std::shared_ptr<MicroCluster>& mc) {
                if (mc->getDecayedWeight(lambda_, ts) < beta_noise_) {
                    RemoveEdgesFor(mc.get());
                    return true;
                }
                return false;
            });
        clusters_.erase(end, clusters_.end());
    }

    void RemoveEdgesFor(const MicroCluster* mc) {
        std::vector<MCLinkKey> to_remove;
        for (const auto& [key, _] : adj_) {
            if (key.a == mc || key.b == mc)
                to_remove.push_back(key);
        }
        for (const auto& key : to_remove)
            adj_.erase(key);
    }
};