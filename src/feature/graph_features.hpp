#pragma once

#include <utility>
#include <cstdint>
#include <armadillo>

#include "../common.hpp"
#include "flow_feature.hpp"
#include "fixed_rolling_counter.hpp"

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

    /* 图结构 */
    using IPSet = robin_hood::unordered_flat_set<uint32_t>;
    std::unordered_map<uint32_t, IPSet> adj_out;   // src→dst
    std::unordered_map<uint32_t, IPSet> adj_in;    // dst←src

    /* 活跃度计数 */
    std::unordered_map<uint32_t, int> node_activity;
    robin_hood::unordered_flat_map<std::pair<uint32_t, uint32_t>, EdgeInfo, hash_pair> edge_activity;

    /* 出度分布维护 */
    std::multiset<size_t> out_degree_distribution;
    std::unordered_map<uint32_t, std::multiset<size_t>::iterator> out_degree_iter;

    /* 滚动计数器 */
    std::shared_ptr<FixedRollingCounter> dst_counter;
    std::shared_ptr<FixedRollingCounter> src_counter;

    /* 时间控制 */
    uint64_t current_ts = 0;
    uint64_t update_count = 0;
    uint64_t prune_interval_;
    uint64_t time_window_;

    size_t edge_limit_;
    size_t simple_step_;
    size_t update_batch_count_ = 0;

    void prune_inactive();
    void real_update(const FlowRecord& f);

public:
    explicit GraphFeatureExtractor(const json& config) : config_(config) 
    {
        const auto& extractor_config = config_["extractor"];
        prune_interval_ = extractor_config.value("prune_interval", 10000);
        time_window_    = extractor_config.value("time_window", 300000);
        edge_limit_     = extractor_config.value("edge_limit", 1000);
        simple_step_    = extractor_config.value("simple_step", 1000);
        
        size_t rolling_counter_size = extractor_config.value("rolling_counter_size", 65536);
        dst_counter = std::make_shared<FixedRollingCounter>(time_window_, rolling_counter_size);
        src_counter = std::make_shared<FixedRollingCounter>(time_window_, rolling_counter_size);
    };

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
};