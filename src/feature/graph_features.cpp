#include "graph_features.hpp"
#include <algorithm>
#include <omp.h>      


/* ---------- advance_time() ---------- */
void GraphFeatureExtractor::advance_time(uint64_t ts) {
    if (ts > current_ts) {
        ++update_count;
        if ((ts - current_ts > 1000) || (update_count % prune_interval_ == 0)) {
            prune_inactive();
        }
        current_ts = ts;
    }
}


/* ---------- real_update() ---------- */
void GraphFeatureExtractor::real_update(const FlowRecord& f) {
    uint32_t src = f.src_ip;
    uint32_t dst = f.dst_ip;
    uint64_t ts  = f.ts_end.tv_sec * 1000ULL + f.ts_end.tv_nsec / 1000000ULL;

    advance_time(ts);

    /* 更新图 */
    size_t old_out_size = adj_out[src].size();
    adj_out[src].insert(dst);
    size_t new_out_size = adj_out[src].size();
    if (new_out_size != old_out_size) {
        update_out_degree(*this, src, new_out_size);
    }

    adj_in[dst].insert(src);
    node_activity[src]++;
    node_activity[dst]++;

    auto edge_key = std::make_pair(src, dst);
    auto it = edge_activity.find(edge_key);
    if (it != edge_activity.end()) {
        it->second.count++;
        it->second.last_ts = current_ts;
    } else {
        edge_activity[edge_key] = EdgeInfo{1, current_ts};
    }
}


/* ---------- updateGraph() ---------- */
void GraphFeatureExtractor::updateGraph(const FlowRecord& flow) {
    ++update_batch_count_;
    size_t edge_num = edge_activity.size();
    if (edge_num <= edge_limit_) {
        real_update(flow);
    } else {
        if (update_batch_count_ % simple_step_ == 0) {
            real_update(flow);
        }
    }
}


/* ---------- prune_inactive ---------- */
void GraphFeatureExtractor::prune_inactive() {
    for (auto it = edge_activity.begin(); it != edge_activity.end();) {
        if (current_ts - it->second.last_ts > time_window_) {
            auto [src, dst] = it->first;
            if (adj_out.count(src)) {
                size_t old_out_size = adj_out[src].size();
                adj_out[src].erase(dst);
                size_t new_out_size = adj_out[src].size();
                if (new_out_size != old_out_size) {
                    update_out_degree(*this, src, new_out_size);
                }
                if (adj_out[src].empty()) adj_out.erase(src);
            }
            if (adj_in.count(dst)) {
                adj_in[dst].erase(src);
                if (adj_in[dst].empty()) adj_in.erase(dst);
            }
            it = edge_activity.erase(it);
        } else {
            ++it;
        }
    }

    /* 清理孤立节点 */
    robin_hood::unordered_flat_set<uint32_t> active;
    active.reserve(adj_out.size() + adj_in.size());
    for (const auto& [k, v] : adj_out) active.insert(k);
    for (const auto& [k, v] : adj_in) active.insert(k);
    for (auto it = node_activity.begin(); it != node_activity.end();) {
        if (active.count(it->first) == 0) it = node_activity.erase(it);
        else ++it;
    }
}


/* ---------- extract ---------- */
arma::vec GraphFeatureExtractor::extract(const FlowRecord& flow) {
    const uint32_t src = flow.src_ip;
    const uint32_t dst = flow.dst_ip;
    const uint64_t ts = flow.ts_end.tv_sec * 1000ULL + flow.ts_end.tv_nsec / 1000000ULL;

    arma::vec v(20, arma::fill::zeros);

    /* === [图结构特征] === */
    size_t out_src = adj_out.count(src) ? adj_out[src].size() : 0;
    size_t in_dst  = adj_in.count(dst)  ? adj_in[dst].size()  : 0;

    v(0) = static_cast<double>(out_src);
    v(1) = static_cast<double>(in_dst);

    dst_counter->add(ts, dst);
    src_counter->add(ts, src);
    v(2) = dst_counter->unique() ? v(0) / dst_counter->unique() : 0.0;
    v(3) = src_counter->unique() ? v(1) / src_counter->unique() : 0.0;

    /* v(4): max Jaccard 相似度 */
    double max_in_sim = 0.0;
    if (adj_in.count(dst)) {
        const auto& my_in = adj_in[dst];
        const size_t my_deg = my_in.size();
        if (my_deg <= 128 && adj_in.size() <= 5000) {
            #pragma omp parallel for reduction(max:max_in_sim)
            for (auto it = adj_in.begin(); it != adj_in.end(); ++it) {
                if (it->first == dst || it->second.size() > 128) continue;
                const auto& other_in = it->second;
                size_t inter = 0;
                for (uint32_t s : my_in) inter += other_in.count(s);
                size_t uni = my_in.size() + other_in.size() - inter;
                if (uni > 0) {
                    double sim = static_cast<double>(inter) / uni;
                    if (sim > max_in_sim) max_in_sim = sim;
                }
            }
        }
    }
    v(4) = max_in_sim;

    v(5) = edge_activity.count({dst, src}) ? 1.0 : 0.0;
    v(6) = edge_activity.count({src, dst}) ? edge_activity[{src, dst}].count : 0.0;
    v(7) = edge_activity.count({dst, src}) ? edge_activity[{dst, src}].count : 0.0;

    v(8) = (out_src > 0) ? std::log2(static_cast<double>(out_src)) : 0.0;

    if (adj_out.count(src) && adj_out.count(dst)) {
        const auto& s_neigh = adj_out[src];
        const auto& d_neigh = adj_out[dst];
        size_t inter = 0;
        for (auto n : s_neigh) inter += d_neigh.count(n);
        size_t uni = s_neigh.size() + d_neigh.size() - inter;
        v(9) = (uni > 0) ? static_cast<double>(inter) / uni : 0.0;
        v(11) = static_cast<double>(inter);
    }

    if (adj_out.count(src)) {
        const auto& neighbors = adj_out[src];
        size_t links = 0;
        for (auto i : neighbors) {
            for (auto j : neighbors) {
                if (i < j && adj_out.count(i) && adj_out[i].count(j)) ++links;
            }
        }
        size_t d = neighbors.size();
        size_t possible = d * (d - 1) / 2;
        v(10) = (possible > 0) ? static_cast<double>(links) / possible : 0.0;
    }

    v(12) = static_cast<double>(out_src) / (1.0 + in_dst);

    /* v(13): 出度 rank percentile */
    if (!out_degree_distribution.empty()) {
        auto it = out_degree_distribution.lower_bound(out_src);
        v(13) = static_cast<double>(std::distance(out_degree_distribution.begin(), it)) /
                out_degree_distribution.size();
    }

    /* === [流量统计特征] === */
    const double duration = flow.get_duration();
    const double avg_len = (flow.packets > 0) ? static_cast<double>(flow.bytes) / flow.packets : 0.0;
    v(14) = flow.proto;
    v(15) = is_well_known_port(flow.src_port) || is_well_known_port(flow.dst_port) ? 1.0 : 0.0;
    v(16) = std::log1p(std::max(0.0, duration));
    v(17) = std::log1p(avg_len);
    v(18) = std::log1p(flow.packets);
    v(19) = std::log1p(flow.bytes);

    return v;
}


/* ---------- static helpers ---------- */
// double GraphFeatureExtractor::protocol_to_onehot(uint16_t proto) {
//     if (proto == 6)  return 1.0;
//     if (proto == 17) return 1.0;
//     return 0.0;
// }

bool GraphFeatureExtractor::is_well_known_port(uint16_t port) {
    static const std::unordered_set<uint16_t> well_known_ports = {
        20, 21, 22, 23, 25, 53, 80, 110, 123, 143, 161, 443, 3306, 8080
    };
    return well_known_ports.count(port) > 0;
}