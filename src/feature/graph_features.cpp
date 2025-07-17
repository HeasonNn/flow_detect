#include "graph_features.hpp"

void GraphFeatureExtractor::updateGraph(const uint32_t src, const uint32_t dst) {
    node_activity[src]++;
    node_activity[dst]++;

    if (adj[src].find(dst) != adj[src].end()) {
        edge_activity[{src, dst}]++;
    } else {
        adj[src].insert(dst);
        edge_activity[{src, dst}] = 1;
    }

}

arma::vec GraphFeatureExtractor::extract(const std::uint32_t src, std::uint32_t dst) {
    int src_deg = adj[src].size();
    int dst_deg = adj[dst].size();
    int mutual = 0;

    for (const auto& n : adj[src]) {
        if (adj[dst].count(n)) mutual++;
    }

    arma::vec v(3);
    v(0) = src_deg;
    v(1) = dst_deg;
    v(2) = mutual;
    
    return v;
}