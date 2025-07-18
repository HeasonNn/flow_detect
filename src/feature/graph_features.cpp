#include "graph_features.hpp"


void GraphFeatureExtractor::advance_time(uint64_t ts) {
    // 仅当当前时间较前次时间戳大时才更新
    if (ts > current_ts) {
        if ((ts - current_ts > 10) || (update_count % PRUNE_INTERVAL == 0)) {
            prune_inactive();
        }
        current_ts = ts;
    }
}


void GraphFeatureExtractor::updateGraph(const FlowRecord& flow) {
    uint32_t src = flow.src_ip;
    uint32_t dst = flow.dst_ip;

    uint64_t ts_ms = flow.ts_end.tv_sec * 1000ULL + flow.ts_end.tv_nsec / 1000000ULL;
    advance_time(ts_ms);

    // 更新邻接表（双向边）
    adj[src].insert(dst);
    // adj[dst].insert(src); // 若需要无向图，保留此行；若需有向图，删除此行

    node_activity[src]++;
    node_activity[dst]++;

    auto edge_key = make_pair(src, dst);
    auto it = edge_activity.find(edge_key);
    if (it != edge_activity.end()) {
        it->second.count++;
        it->second.last_ts = current_ts;
    } else {
        edge_activity[edge_key] = EdgeInfo{1, current_ts};
    }

    ++update_count;
}


void GraphFeatureExtractor::prune_inactive() {
    for (auto it = edge_activity.begin(); it != edge_activity.end(); ) {
        const auto& info = it->second;
        if ((current_ts - info.last_ts) > TIME_WINDOW) {
            const auto& [src, dst] = it->first;
            // 检查 src 是否存在并且是非空
            if (adj.find(src) != adj.end()) {
                adj[src].erase(dst);  // 清除掉该边
            }
            it = edge_activity.erase(it);  // 删除该边活动记录
        } else {
            ++it;
        }
    }

    for (auto it = node_activity.begin(); it != node_activity.end();) {
        if (adj.find(it->first) != adj.end() && adj[it->first].empty()) {
            it = node_activity.erase(it);
        } else {
            ++it;
        }
    }
}


arma::vec GraphFeatureExtractor::extract(const FlowRecord& flow) {
    // === 图结构特征（6维） ===
    uint32_t src = flow.src_ip;
    uint32_t dst = flow.dst_ip;

    int src_deg = adj.count(src) ? adj[src].size() : 0;
    int dst_deg = adj.count(dst) ? adj[dst].size() : 0;
    int mutual = 0;

    if (adj.count(src) && adj.count(dst)) {
        if (adj[src].size() > adj[dst].size()) swap(src, dst);
        for (const auto& n : adj[src]) {
            if (adj[dst].count(n)) mutual++;
        }
    }

    int edge_count = 0;
    auto it = edge_activity.find({src, dst});
    if (it != edge_activity.end()) {
        edge_count = it->second.count;
    }

    bool has_reverse = (adj.count(dst) && adj[dst].count(src));

    double jaccard = 0.0;
    int union_size = src_deg + dst_deg - mutual;
    if (union_size > 0) jaccard = static_cast<double>(mutual) / union_size;

    arma::vec v(12);  // 总共12维特征

    // 图结构特征（6维）
    v(0) = src_deg;
    v(1) = dst_deg;
    v(2) = mutual;
    v(3) = edge_count;
    v(4) = jaccard;
    v(5) = has_reverse ? 1.0 : 0.0;

    // === 流量行为特征（6维） ===
    // 协议类型 One-hot 编码
    double duration = flow.get_duration();
    double avg_len = (flow.packets > 0) ? static_cast<double>(flow.bytes) / flow.packets : 0.0;
    
    v(6) = flow.proto;
    v(7) = is_well_known_port(flow.src_port) || is_well_known_port(flow.dst_port) ? 1.0 : 0.0;
    v(8) = std::log1p(std::max(0.0, duration));
    v(9) = std::log1p(avg_len);
    v(10) = std::log1p(flow.packets);
    v(11) = std::log1p(flow.bytes);
    return v;
}

double GraphFeatureExtractor::protocol_to_onehot(uint16_t proto) {
    if (proto == 6) return 1.0;   // TCP
    if (proto == 17) return 1.0;  // UDP
    return 0.0;                  // Other
}

bool GraphFeatureExtractor::is_well_known_port(uint16_t port) {
    static const unordered_set<uint16_t> well_known_ports = {
        20, 21, 22, 23, 25, 53, 80, 110, 123, 143, 161, 443, 3306, 8080
    };
    return well_known_ports.count(port) > 0;
}