#include "edge_constructor.hpp"

void EdgeConstructor::ClassifyFlow(){
     size_t sum_short = 0, sum_long = 0;
    for (size_t idx = 0; idx < all_flow_vec_ptr_->size(); idx++) {
        const auto& [flow, label] = all_flow_vec_ptr_->at(idx);
        if (flow.packets > EDGE_LONG_LINE) {
            long_flowid_vec_ptr_->push_back(idx);
            sum_long += flow.packets;
        } else {
            short_flowid_vec_ptr_->push_back(idx);
            sum_short += flow.packets;
        }
    }
    LOGF("Before aggregation: %ld short edges [%ld pkts], %ld long edges [%ld pkts].", 
            short_flowid_vec_ptr_->size(), sum_short, long_flowid_vec_ptr_->size(), sum_long);
}


void EdgeConstructor::aggregateBySrcDstIP(){
    using FlowKey = std::pair<uint32_t, uint32_t>;
    auto flow_map = std::make_shared<std::unordered_map<FlowKey, std::vector<size_t>>>();
    for (const auto idx : *short_flowid_vec_ptr_) {
        const auto& [flow, label] = (*all_flow_vec_ptr_)[idx];
        FlowKey key = {flow.src_ip, flow.dst_ip};
        (*flow_map)[key].push_back(idx);
    }

    size_t short_flow_sum2 = flow_map->size();
    LOGF("Before aggregation: %ld short flow, After IP-only aggregation: %ld groups (↓ %.2f%%)", 
        short_flowid_vec_ptr_->size(), 
        short_flow_sum2,
        100.0 * (1.0 - static_cast<double>(short_flow_sum2) / short_flowid_vec_ptr_->size())
    );
}


void EdgeConstructor::aggregateByCosine(double kSimilarityThreshold) {
    auto flow_feat_map = make_shared<unordered_map<pair<uint32_t, uint32_t>,  vector<AggregatedFlowInfo>>>();

    const auto extract_flow_feature_fn = [&](const FlowRecord& flow) -> arma::vec {
        arma::vec v(5);
        v(0) = static_cast<double>(flow.src_port) / 65535.0;
        v(1) = static_cast<double>(flow.dst_port) / 65535.0;
        v(2) = std::log1p(flow.get_duration());
        v(3) = std::log1p(flow.packets);
        v(4) = std::log1p(flow.bytes) - std::log1p(flow.packets);
        return v;
    };

    auto cosine_similarity_fn = [&](const arma::vec& x, const arma::vec& y) -> double {
        const double kEpsilon = 1e-12;
        return arma::dot(x, y) / (arma::norm(x, 2) * arma::norm(y, 2) + kEpsilon);
    };

    for (const auto idx : * short_flowid_vec_ptr_) {
        const auto& [flow, label] = (*all_flow_vec_ptr_)[idx];
        std::pair<uint32_t, uint32_t> flow_key = {flow.src_ip, flow.dst_ip};
        arma::vec flow_feat = extract_flow_feature_fn(flow);
        auto& agg_info_vec = (*flow_feat_map)[flow_key];

        bool matched = false;
        for (auto& agg_info : agg_info_vec) {
            if (cosine_similarity_fn(flow_feat, agg_info.feat) >= kSimilarityThreshold) {
                agg_info.feat = (agg_info.feat * agg_info.flow_idx.size() + flow_feat) 
                                / static_cast<double>(agg_info.flow_idx.size() + 1);
                agg_info.flow_idx.emplace_back(idx);
                matched = true;
                break;
            }
        }

        if (!matched) {
            agg_info_vec.push_back({flow_feat, {idx}});
        }
    }

    size_t short_flow_sum = 0;
    for(auto it = flow_feat_map->begin(); it != flow_feat_map->end(); it++){
        short_flow_sum += it->second.size();
    }

    LOGF("Before aggregation: %ld short flows, After Cosine-Similarity aggregation: %ld groups (↓ %.2f%%), Threshold = %.4f", 
        short_flowid_vec_ptr_->size(), 
        short_flow_sum,
        100.0 * (1.0 - static_cast<double>(short_flow_sum) / short_flowid_vec_ptr_->size()),
        kSimilarityThreshold);
}


void EdgeConstructor::AggregateFlow(){
    aggregateBySrcDstIP();
    aggregateByCosine(0.5);
    aggregateByCosine(0.7);
    aggregateByCosine(0.9);
}