#include "graph_features.hpp"
#include <algorithm>
#include <omp.h>      


/* ---------- GraphFeatureExtractor ---------- */
GraphFeatureExtractor::GraphFeatureExtractor(const json& config)
    : config_(config)
{
    const auto& extractor_config = config_["extractor"];
    prune_interval_ = extractor_config.value("prune_interval", 1000);
    time_window_    = extractor_config.value("time_window", 300000);
    max_nodes_      = extractor_config.value("max_nodes", 10000);
    max_edges_      = extractor_config.value("max_edges", 50000);
    simple_step_    = extractor_config.value("simple_step", 1000);

    std::cout << "prune_interval: " << prune_interval_
              << ", time_window: " << time_window_
              << ", max_nodes: " << max_nodes_
              << ", max_edges: " << max_edges_
              << ", simple_step: " << simple_step_ << "\n" << std::flush;
}

/* ---------- advance_time() ---------- */
void GraphFeatureExtractor::advance_time(uint64_t ts) {
    if (ts >= current_ts_) {
        ++update_count_;
        current_ts_ = ts;
        if ((update_count_ % prune_interval_ == 0)) {
            prune_inactive();
        }
    }
}

/* ---------- real_update() ---------- */
void GraphFeatureExtractor::real_update(const FlowRecord& f) {
    uint32_t src = f.src_ip;
    uint32_t dst = f.dst_ip;
    uint64_t ts  = GET_DOUBLE_TS(f.ts_end);

    auto edge_key = std::make_pair(src, dst);
    bool new_edge = false;

    // --- 更新边 ---
    auto eit = edge_activity.find(edge_key);
    if (eit != edge_activity.end()) {
        // 已存在：更新计数和时间戳
        eit->second.count++;
        eit->second.last_ts = ts;
        
        // 更新FIFO队列：先移除旧位置，再添加到尾部
        remove_from_edge_fifo(edge_key);
        edge_fifo_queue_.push_back({ts, edge_key});

    } else {
        // 新边：检查硬上限
        while (edge_activity.size() >= max_edges_ && !edge_fifo_queue_.empty()) {
            auto [oldest_ts, oldest_edge] = edge_fifo_queue_.front();
            edge_fifo_queue_.pop_front();

            auto it = edge_activity.find(oldest_edge);
            if (it != edge_activity.end()) {
                uint32_t s = oldest_edge.first, d = oldest_edge.second;
                
                // 清理邻接表
                if (adj_out.count(s)) {
                    size_t old_deg = adj_out[s].size();
                    adj_out[s].erase(d);
                    size_t new_deg = adj_out[s].size();
                    if (new_deg != old_deg) {
                        update_out_degree(*this, s, new_deg);
                    }
                    if (adj_out[s].empty()) {
                        out_degree_iter.erase(s);
                        adj_out.erase(s);
                    }
                }
                if (adj_in.count(d)) {
                    adj_in[d].erase(s);
                    if (adj_in[d].empty()) {
                        adj_in.erase(d);
                    }
                }

                // 安全降低节点活跃度
                if (node_activity.count(s)) {
                    node_activity[s]--;
                    if (node_activity[s] <= 0) node_activity.erase(s);
                }
                if (node_activity.count(d)) {
                    node_activity[d]--;
                    if (node_activity[d] <= 0) node_activity.erase(d);
                }

                edge_activity.erase(it);
            }
        }

        // 插入新边
        edge_activity[edge_key] = EdgeInfo{1, ts};
        edge_fifo_queue_.push_back({ts, edge_key});
        new_edge = true;
    }

    // --- 更新图结构和节点活跃度 ---
    if (new_edge) {
        adj_out[src].insert(dst);
        adj_in[dst].insert(src);

        size_t new_out_size = adj_out[src].size();
        update_out_degree(*this, src, new_out_size);

        // 注册节点活跃性
        node_activity[src]++;
        node_activity[dst]++;

        // 加入 FIFO（用于 LRU 管理）
        remove_from_node_fifo(src);
        remove_from_node_fifo(dst);
        node_fifo_queue_.push_back({ts, src});
        node_fifo_queue_.push_back({ts, dst});

    } else {
        // 刷新已有边的时间顺序
        remove_from_node_fifo(src);
        remove_from_node_fifo(dst);
        node_fifo_queue_.push_back({ts, src});
        node_fifo_queue_.push_back({ts, dst});
    }

    // size_t num_nodes = node_activity.size();
    // size_t num_edges = edge_activity.size();

    // uint64_t strat_ts  = GET_DOUBLE_TS(f.ts_start);
    // uint64_t end_ts    = GET_DOUBLE_TS(f.ts_end);
    // std::cout << "[Graph Update] src: " << src 
    //         << " | dst: " << dst 
    //         << " | start_ts: " << strat_ts
    //         << " | end_ts: " << end_ts 
    //         << " | current ts: " << current_ts_ 
    //         << " | Nodes: " << num_nodes 
    //         << " | Edges: " << num_edges 
    //         << " | FIFO Node Size: " << node_fifo_queue_.size() 
    //         << " | FIFO Edge Size: " << edge_fifo_queue_.size() 
    //         << "\n" << std::flush;
}

/* ---------- updateGraph() ---------- */
void GraphFeatureExtractor::updateGraph(const FlowRecord& flow) {
    ++update_batch_count_;
    if (update_batch_count_ % simple_step_ == 0) {
        real_update(flow);
    }
}

/* ---------- prune_inactive() ---------- */
void GraphFeatureExtractor::prune_inactive() {
    // === 第一阶段：收集并清理过期的边 ===
    std::vector<std::pair<uint32_t, uint32_t>> expired_edges;

    for (auto& [edge, info] : edge_activity) {
        if (current_ts_ - info.last_ts > time_window_) {
            expired_edges.push_back(edge);
        }
    }

    for (const auto& [src, dst] : expired_edges) {
        auto ekey = std::make_pair(src, dst);
        remove_from_edge_fifo(ekey); // 从FIFO队列移除
        edge_activity.erase(ekey);   // 从边表移除

        // 同步清理图结构
        if (adj_out.count(src)) {
            size_t old_deg = adj_out[src].size();
            adj_out[src].erase(dst);
            size_t new_deg = adj_out[src].size();
            if (new_deg != old_deg) {
                update_out_degree(*this, src, new_deg);
            }
            if (adj_out[src].empty()) {
                out_degree_iter.erase(src);
                adj_out.erase(src);
            }
        }
        if (adj_in.count(dst)) {
            adj_in[dst].erase(src);
            if (adj_in[dst].empty()) {
                adj_in.erase(dst);
            }
        }

        // 安全地降低节点活跃度
        if (node_activity.count(src)) {
            node_activity[src]--;
            if (node_activity[src] <= 0) node_activity.erase(src);
        }
        if (node_activity.count(dst)) {
            node_activity[dst]--;
            if (node_activity[dst] <= 0) node_activity.erase(dst);
        }
    }

    // === 第二阶段：清理因无连接而失效的孤立节点 ===
    std::vector<uint32_t> expired_nodes;
    for (auto it = node_activity.begin(); it != node_activity.end();) {
        uint32_t ip = it->first;
        bool has_connection = adj_out.count(ip) || adj_in.count(ip);
        
        if (!has_connection) {
            expired_nodes.push_back(ip);
            it = node_activity.erase(it);
        } else {
            ++it;
        }
    }

    for (uint32_t ip : expired_nodes) {
        remove_from_node_fifo(ip);
    }

    // === 第三阶段：强制执行资源上限 (Hard Limit) ===
    // 驱逐节点
    size_t node_evictions = 0;
    while (node_activity.size() > max_nodes_ && !node_fifo_queue_.empty()) {
        auto [_, ip] = node_fifo_queue_.front();
        node_fifo_queue_.pop_front();
        if (node_activity.count(ip) && !adj_out.count(ip) && !adj_in.count(ip)) {
            node_activity.erase(ip);
            out_degree_iter.erase(ip);
            node_evictions++;
        }
    }

    // 驱逐边
    size_t edge_evictions = 0;
    while (edge_activity.size() > max_edges_ && !edge_fifo_queue_.empty()) {
        auto [_, edge] = edge_fifo_queue_.front();
        edge_fifo_queue_.pop_front();
        auto it = edge_activity.find(edge);
        if (it != edge_activity.end()) {
            // 复用清理逻辑
            uint32_t s = edge.first, d = edge.second;
            if (adj_out.count(s)) {
                size_t old_deg = adj_out[s].size();
                adj_out[s].erase(d);
                size_t new_deg = adj_out[s].size();
                if (new_deg != old_deg) {
                    update_out_degree(*this, s, new_deg);
                }
                if (adj_out[s].empty()) {
                    out_degree_iter.erase(s);
                    adj_out.erase(s);
                }
            }
            if (adj_in.count(d)) {
                adj_in[d].erase(s);
                if (adj_in[d].empty()) {
                    adj_in.erase(d);
                }
            }
            if (node_activity.count(s)) {
                node_activity[s]--;
                if (node_activity[s] <= 0) node_activity.erase(s);
            }
            if (node_activity.count(d)) {
                node_activity[d]--;
                if (node_activity[d] <= 0) node_activity.erase(d);
            }
            edge_activity.erase(it);
            edge_evictions++;
        }
    }

    // std::cout << "[Graph Pruning] 驱逐边数量: " << edge_evictions << std::endl;
    // size_t num_nodes = node_activity.size();
    // size_t num_edges = edge_activity.size();

    // std::cout << "[Graph Pruning] 当前节点数: " << num_nodes
    //           << " | 当前边数: " << num_edges
    //           << " | FIFO 节点队列大小: " << node_fifo_queue_.size()
    //           << " | FIFO 边队列大小: " << edge_fifo_queue_.size()
    //           << "\n" << std::flush;
}


/* ---------- FIFO 辅助函数 (O(N) 但绝对安全) ---------- */
void GraphFeatureExtractor::remove_from_node_fifo(uint32_t ip) {
    auto it = std::find_if(node_fifo_queue_.begin(), node_fifo_queue_.end(),
                           [ip](const auto& item) { return item.second == ip; });
    if (it != node_fifo_queue_.end()) {
        node_fifo_queue_.erase(it);
    }
    // 如果找不到，静默失败
}

void GraphFeatureExtractor::remove_from_edge_fifo(const std::pair<uint32_t, uint32_t>& edge) {
    auto it = std::find_if(edge_fifo_queue_.begin(), edge_fifo_queue_.end(),
                           [&edge](const auto& item) { return item.second == edge; });
    if (it != edge_fifo_queue_.end()) {
        edge_fifo_queue_.erase(it);
    }
    // 如果找不到，静默失败
}

/* ---------- extract 和 helpers 保持不变 ---------- */
arma::vec GraphFeatureExtractor::extract(const FlowRecord& flow) {
    const uint32_t src = flow.src_ip;
    const uint32_t dst = flow.dst_ip;
    const uint64_t ts = flow.ts_end.tv_sec * 1000ULL + flow.ts_end.tv_nsec / 1000000ULL;

    arma::vec v(16, arma::fill::zeros);
    size_t out_src = adj_out.count(src) ? adj_out[src].size() : 0;
    size_t in_dst  = adj_in.count(dst)  ? adj_in[dst].size()  : 0;

    v(0) = static_cast<double>(out_src);
    v(1) = static_cast<double>(in_dst);
    v(2) = node_activity.size() ? v(0) / node_activity.size() : 0.0;
    v(3) = node_activity.size() ? v(1) / node_activity.size() : 0.0;
    v(4) = edge_activity.count({dst, src}) ? 1.0 : 0.0;
    v(5) = edge_activity.count({src, dst}) ? edge_activity[{src, dst}].count : 0.0;
    v(6) = edge_activity.count({dst, src}) ? edge_activity[{dst, src}].count : 0.0;
    v(7) = (out_src > 0) ? std::log2(static_cast<double>(out_src)) : 0.0;
    v(8) = static_cast<double>(out_src) / (1.0 + in_dst);

    if (!out_degree_distribution.empty()) {
        auto it = out_degree_distribution.lower_bound(out_src);
        v(9) = static_cast<double>(std::distance(out_degree_distribution.begin(), it)) /
               out_degree_distribution.size();
    }

    const double duration = flow.get_duration();
    const double avg_len = (flow.packets > 0) ? static_cast<double>(flow.bytes) / flow.packets : 0.0;
    v(10) = flow.proto;
    v(11) = is_well_known_port(flow.src_port) || is_well_known_port(flow.dst_port) ? 1.0 : 0.0;
    v(12) = std::log1p(std::max(0.0, duration));
    v(13) = std::log1p(avg_len);
    v(14) = std::log1p(flow.packets);
    v(15) = std::log1p(flow.bytes);

    return v;
}

bool GraphFeatureExtractor::is_well_known_port(uint16_t port) {
    static const std::unordered_set<uint16_t> well_known_ports = {
        20, 21, 22, 23, 25, 53, 80, 110, 123, 143, 161, 443, 3306, 8080
    };
    return well_known_ports.count(port) > 0;
}