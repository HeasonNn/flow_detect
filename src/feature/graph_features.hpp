#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <armadillo>

#include "../common.hpp"

using namespace std;

class GraphFeatureExtractor {
private:
    unordered_map<uint32_t, unordered_set<uint32_t>> adj;

    unordered_map<uint32_t, int> node_activity;
    unordered_map<pair<uint32_t, uint32_t>, int> edge_activity;
    const int ACTIVITY_THRESHOLD = 10;
    uint update_count = 0;

public:
    void updateGraph(const uint32_t src, const uint32_t dst);
    arma::vec extract(const uint32_t src, const uint32_t dst);
};