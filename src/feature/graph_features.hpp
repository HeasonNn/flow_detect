#pragma once

#include <utility>
#include <cstdint>
#include <armadillo>
#include <deque>
#include <queue>

#include "robin_hood.hpp"
#include "../common.hpp"
#include "flow_feature.hpp"

struct hash_pair {
    std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const noexcept {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(p.first) << 32) | p.second);
    }
};

struct EdgeInfo
{
    size_t count = 0;
    uint64_t last_ts = 0;
};

class GraphFeatureExtractor {
private:
    const json& config_;

    /* 核心图结构 */
    using IPSet = std::unordered_set<uint32_t>;
    std::unordered_map<uint32_t, IPSet> adj_out;
    std::unordered_map<uint32_t, IPSet> adj_in;

    /* 活跃度计数 */
    std::unordered_map<uint32_t, size_t> node_activity;

    /* 边信息表 */
    std::unordered_map<std::pair<uint32_t, uint32_t>, EdgeInfo, hash_pair> edge_activity;

    /* 出度分布 */
    std::multiset<size_t> out_degree_distribution;
    std::unordered_map<uint32_t, std::multiset<size_t>::iterator> out_degree_iter;

    /* 时间与资源控制 */
    uint64_t current_ts_ = 0;
    uint64_t update_count_ = 0;
    uint64_t prune_interval_;
    uint64_t time_window_;
    size_t max_nodes_;
    size_t max_edges_;
    size_t simple_step_;
    size_t update_batch_count_ = 0;

    /* FIFO 队列：用于硬性上限驱逐 (使用 find_if 进行安全删除) */
    std::deque<std::pair<uint64_t, uint32_t>> node_fifo_queue_;       // <last_active_ts, node_ip>
    std::deque<std::pair<uint64_t, std::pair<uint32_t, uint32_t>>> edge_fifo_queue_; // <last_active_ts, edge_key>

    void prune_inactive();
    void real_update(const FlowRecord& f);

    // FIFO辅助函数 (O(N) 但绝对安全)
    void remove_from_node_fifo(uint32_t ip);
    void remove_from_edge_fifo(const std::pair<uint32_t, uint32_t>& edge);

public:
    explicit GraphFeatureExtractor(const json& config);

    void advance_time(uint64_t ts);
    void updateGraph(const FlowRecord& flow);
    arma::vec extract(const FlowRecord& flow);

    static bool is_well_known_port(uint16_t port);

    inline void update_out_degree(GraphFeatureExtractor& self, uint32_t ip, size_t new_deg) {
        auto& ms = self.out_degree_distribution;
        auto& iter_map = self.out_degree_iter;

        if (iter_map.count(ip)) {
            ms.erase(iter_map[ip]);
        }
        iter_map[ip] = ms.insert(new_deg);
    }

    size_t active_nodes() const { return node_activity.size(); }
    size_t active_edges() const { return edge_activity.size(); }
};