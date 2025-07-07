#include "graph_features.hpp"

void GraphFeatureExtractor::updateGraph(const std::string& src, const std::string& dst) {
    node_activity[src]++;
    node_activity[dst]++;

    if (adj[src].find(dst) != adj[src].end()) {
        edge_activity[{src, dst}]++;
    } else {
        adj[src].insert(dst);
        edge_activity[{src, dst}] = 1;
    }

    update_count++;

    if (update_count >= 100) {
        pruneInactiveElements();
        update_count = 0;
    }
}

void GraphFeatureExtractor::pruneInactiveElements() {
    for (auto it = node_activity.begin(); it != node_activity.end();) {
        if (it->second < ACTIVITY_THRESHOLD) {
            adj.erase(it->first);
            it = node_activity.erase(it); 
        } else {
            ++it;
        }
    }

    for (auto it = edge_activity.begin(); it != edge_activity.end();) {
        if (it->second < ACTIVITY_THRESHOLD) {
            adj[std::get<0>(it->first)].erase(std::get<1>(it->first)); 
            it = edge_activity.erase(it);
        } else {
            ++it;
        }
    }
}

arma::vec GraphFeatureExtractor::extract(const std::string& src, const std::string& dst) {
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