#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <armadillo>

namespace std {
    template <>
    struct hash<std::tuple<std::string, std::string>> {
        size_t operator()(const std::tuple<std::string, std::string>& t) const {
            size_t h1 = std::hash<std::string>{}(std::get<0>(t));
            size_t h2 = std::hash<std::string>{}(std::get<1>(t));
            return h1 ^ (h2 << 1);
        }
    };
}

class GraphFeatureExtractor {
private:
    std::unordered_map<std::string, std::unordered_set<std::string>> adj;

    std::unordered_map<std::string, int> node_activity;
    std::unordered_map<std::tuple<std::string, std::string>, int> edge_activity;
    const int ACTIVITY_THRESHOLD = 10;
    uint update_count = 0;

public:
    void updateGraph(const std::string& src, const std::string& dst);
    void pruneInactiveElements();

    arma::vec extract(const std::string& src, const std::string& dst);
};