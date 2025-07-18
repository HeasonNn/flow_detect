#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <armadillo>

#include "flow_feature.hpp"
#include "../common.hpp"

using namespace std;

struct hash_pair {
    size_t operator()(const pair<uint32_t, uint32_t>& p) const {
        return hash<uint64_t>()(((uint64_t)p.first << 32) | p.second);
    }
};

struct EdgeInfo {
    int count = 0;
    uint64_t last_ts = 0;
};


class GraphFeatureExtractor {
private:
    unordered_map<uint32_t, unordered_set<uint32_t>> adj;
    unordered_map<uint32_t, int> node_activity;
    unordered_map<pair<uint32_t, uint32_t>, EdgeInfo, hash_pair> edge_activity;
    
    uint64_t current_ts = 0;
    uint64_t update_count = 0;
    const uint64_t PRUNE_INTERVAL = 10000;
    const uint64_t TIME_WINDOW = 300;
    
    void prune_inactive();

public:
    GraphFeatureExtractor() = default;

    void advance_time(uint64_t ts);
    void updateGraph(const FlowRecord& flow);
    arma::vec extract(const FlowRecord& flow);
    
    static bool is_well_known_port(uint16_t port);
    static double protocol_to_onehot(uint16_t proto);
};