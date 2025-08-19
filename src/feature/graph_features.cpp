#include "graph_features.hpp"
#include <algorithm>


/* ================= GraphFeatureExtractor ================= */
GraphFeatureExtractor::GraphFeatureExtractor(const json& cfg, Time start_time) 
    : cfg_(cfg), last_prune_ts_(start_time)
{
    const auto& extractor_config = cfg_["extractor"];
    const auto ttl = extractor_config.value("ttl_ms", 30000);
    const auto wheel_granularity = extractor_config.value("wheel_granularity_ms", 1000);
    const auto prune_interval = cfg.value("prune_interval", 1000);

    prune_interval_ = Dur(prune_interval);

    g_ = std::make_unique<GraphMaintainer>(Dur(ttl), Dur(wheel_granularity), start_time);

    // std::cout << "GraphFeatureExtractor initialized.\n"
    //           << "  ttl: "               << ttl << "ms\n"
    //           << "  wheel_granularity: " << wheel_granularity << "ms\n"
    //           << "  prune_interval: "    << prune_interval << "ms\n"
    //           << std::flush;
}

void GraphFeatureExtractor::maybe_prune(Time now) {
    // now 是当前流的时间
    auto interval = (now - last_prune_ts_).count();
    // std::cout << "Intelval: " << interval << "  prune_interval_: " << prune_interval_.count() << "\n";
    if (now - last_prune_ts_ > prune_interval_) {
        // std::cout << "exec g_->prune(now);" << "\n";
        g_->prune(now);
        last_prune_ts_ = now; // 更新为流时间
        
        // std::cout << "[Graph Stats] Active Nodes: " << g_->active_nodes() << " | Active Edges: " << g_->active_edges() << "\n";
    }
}

void GraphFeatureExtractor::updateGraph(const FlowRecord& f) {
    Time ts_start = to_time_point(f.ts_start); // 获取流的真实开始时间
    Time ts_end   = to_time_point(f.ts_end);
    maybe_prune(ts_start); // 传入流时间
    g_->update(f.src_ip, f.dst_ip, ts_start, ts_end);

    // std::cout << std::fixed << std::setprecision(15);
    // std::cout << f.src_ip << ", " << f.src_ip << ", " << GET_DOUBLE_TS(f.ts_start) << "\n";
}

arma::vec GraphFeatureExtractor::extract(const FlowRecord& f) const {
    arma::vec v(20, arma::fill::zeros);
    Time now = to_time_point(f.ts_start); // 特征提取也基于流时间

    double duration_ms = std::max(f.get_duration(), 1.0); // 防止除零

    const NodeMeta* ns = g_->node(f.src_ip, now);
    const NodeMeta* nd = g_->node(f.dst_ip, now);
    const EdgeMeta* ed = g_->edge(f.src_ip, f.dst_ip, now);

    size_t active_nodes = g_->active_nodes();
    size_t active_edges = g_->active_edges();

    // ==================== 1. 节点级特征 ====================
    v(0) = ns ? std::log1p(static_cast<double>(ns->out_deg)) : 0.0;
    v(1) = ns ? std::log1p(static_cast<double>(ns->in_deg))  : 0.0;
    v(2) = nd ? std::log1p(static_cast<double>(nd->in_deg))  : 0.0;
    v(3) = nd ? std::log1p(static_cast<double>(nd->out_deg)) : 0.0;

    if (ns) {
        auto diff = static_cast<int64_t>(ns->out_deg) - static_cast<int64_t>(ns->in_deg);
        v(4) = std::abs(static_cast<double>(diff));
    }
    if (nd) {
        auto diff = static_cast<int64_t>(nd->out_deg) - static_cast<int64_t>(nd->in_deg);
        v(5) = std::abs(static_cast<double>(diff));
    }

    // ==================== 2. 边级特征 ====================
    v(6) = ed ? std::log1p(static_cast<double>(ed->count)) : 0.0;

    // ==================== 3. 流量特征 ====================
    v(7) = std::log1p(static_cast<double>(f.bytes));
    v(8) = std::log1p(static_cast<double>(f.packets));
    v(9) = std::log1p(duration_ms);
    v(10) = std::log1p(static_cast<double>(f.bytes) / duration_ms);
    v(11) = std::log1p(static_cast<double>(f.packets) / duration_ms);
    v(12) = f.packets > 0 ? static_cast<double>(f.bytes) / f.packets : 0.0;
    v(13) = static_cast<double>(f.proto);

    // ==================== 4. 全局上下文特征 ====================
    v(14) = std::log1p(static_cast<double>(active_nodes));
    v(15) = std::log1p(static_cast<double>(active_edges));
    v(16) = (active_nodes > 0) ? (static_cast<double>(active_edges) / active_nodes) : 0.0;

    // ==================== 5. 高级行为特征 ====================
    if (ns && ns->out_deg > 0) {
        double ratio = static_cast<double>(ns->in_deg) / (ns->out_deg + ns->in_deg + 1);
        v(17) = std::log1p(static_cast<double>(ns->out_deg)) * (1.0 - ratio);
    }

    v(18) = (ns && ns->out_deg > 10) ? 1.0 : 0.0;

    double global_density = v(16);
    double edge_frequency = v(6);
    v(19) = (1.0 / (global_density + 1.0)) * (1.0 / (edge_frequency + 1.0));

    return v;
}