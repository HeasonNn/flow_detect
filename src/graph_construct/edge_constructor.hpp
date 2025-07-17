#pragma once

#include <mlpack/core.hpp>
#include "../feature/flow_feature.hpp"
#include "../common.hpp"

using namespace std;

struct AggregatedFlowInfo {
    arma::vec feat;
    vector<size_t> flow_idx;
};


class EdgeConstructor
{
private:
    shared_ptr<vector<pair<FlowRecord, size_t>>> all_flow_vec_ptr_;
    shared_ptr<vector<size_t>> long_flowid_vec_ptr_;
    shared_ptr<vector<size_t>> short_flowid_vec_ptr_;

    u_int16_t EDGE_LONG_LINE = 15;

    void aggregateByCosine(double kSimilarityThreshold = 0.95);
    void aggregateBySrcDstIP();

public:
    explicit EdgeConstructor(const shared_ptr<vector<pair<FlowRecord, size_t>>> all_flow_vec_ptr)
        : long_flowid_vec_ptr_(make_shared<vector<size_t>>()),
          short_flowid_vec_ptr_(make_shared<vector<size_t>>()),
          all_flow_vec_ptr_(all_flow_vec_ptr) {};

    void ClassifyFlow();
    void AggregateFlow(); 

    ~EdgeConstructor() = default;
};
