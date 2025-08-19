#include "detector.hpp"

namespace fs = filesystem;


MixDetector::MixDetector(shared_ptr<DataLoader> loader, 
                         const json& config)
    : Detector(loader, config) 
{
    const json& iforest_config_ = config_["detector"]["iforest_config"];
    random_seed_       = iforest_config_.value("random_seed", 2025);
    n_trees_           = iforest_config_.value("n_trees", 100);
    sample_size_       = iforest_config_.value("sample_size", 64);
    max_depth_         = iforest_config_.value("max_depth", 10);
    outline_threshold_ = iforest_config_.value("outline_threshold", 0.1);

    const json& dbstream_config = config_["detector"]["dbstream_config"];
    epsilon_              = dbstream_config.value("epsilon", 0.05);
    lambda_               = dbstream_config.value("lambda", 0.01);
    mu_                   = dbstream_config.value("mu", 3);
    beta_noise_           = dbstream_config.value("beta_noise", 1.5);
    max_clusters_         = dbstream_config.value("max_clusters", 500);
    eta_                  = dbstream_config.value("eta", 0.1);
    connection_threshold_ = dbstream_config.value("connection_threshold", 0.3);
    
    std::cout << "[MixDetector Initialized Parameters]\n";
    std::cout << "🔧 IForest Config:\n";
    std::cout << "  - random_seed        = " << random_seed_ << "\n";
    std::cout << "  - n_trees            = " << n_trees_ << "\n";
    std::cout << "  - sample_size        = " << sample_size_ << "\n";
    std::cout << "  - max_depth          = " << max_depth_ << "\n";
    std::cout << "  - outline_threshold  = " << outline_threshold_ << "\n";

    std::cout << "🧠 DBSTREAM Config:\n";
    std::cout << "  - epsilon            = " << epsilon_ << "\n";
    std::cout << "  - lambda             = " << lambda_ << "\n";
    std::cout << "  - mu                 = " << mu_ << "\n";
    std::cout << "  - beta_noise         = " << beta_noise_ << "\n";
    std::cout << "  - max_clusters       = " << max_clusters_ << "\n";

    flows_ = flows_ = loader_->getAllData();;
}


void MixDetector::execPCAProccess(const PartitionData& pt) {
    mlpack::PCA pca;
    arma::mat reduced;
    pca.Apply(*pt.norm_data, reduced, 2);

    const size_t n = reduced.n_cols;
    if (pt.idx_vec->size() != n || pt.assignments->size() != n) {
        std::cerr << "[execPCAProccess] Size mismatch! idx_vec/assignments/reduced columns not aligned.\n";
        return;
    }

    std::ofstream fout("result/dbstream_pca_pid_" + std::to_string(pt.pid) + ".csv");
    if (!fout.is_open()) {
        std::cerr << "Failed to open output file for PCA result.\n";
        return;
    }

    fout << "x,y,assignments,label,is_outlier\n";
    for (size_t i = 0; i < n; ++i) {
        size_t idx = pt.idx_vec->at(i);
        size_t raw_label = flows_->at(idx).second;
        int assign = pt.assignments->at(i);
        bool is_outlier = pt.outlier->count(idx) > 0;  // 这里的 i 是否应为 pt.idx_vec[i]，视你的 outlier 设计

        fout << std::setprecision(6)
             << reduced(0, i) << "," 
             << reduced(1, i) << ","
             << assign        << "," 
             << raw_label     << "," 
             << is_outlier    << "\n";
    }
}

void MixDetector::execDBstreamDetect(const PartitionData& pt) {
    if (!pt.sample_vec || pt.sample_vec->empty()) {
        std::cerr << "[execDBstreamDetect] partition " << pt.pid << " has no samples, skip.\n";
        return;
    }
    if (!pt.idx_vec || pt.idx_vec->size() != pt.sample_vec->size()) {
        std::cerr << "[execDBstreamDetect] idx_vec size mismatch, skip.\n";
        return;
    }

    const size_t dim = pt.sample_vec->at(0).n_elem;
    const size_t n_samples = pt.sample_vec->size();
    arma::mat data(dim, n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        data.col(i) = pt.sample_vec->at(i);
    }

    // 标准化
    scaler_.Fit(data);
    scaler_.Transform(data, *pt.norm_data);

    // DBSTREAM 实例
    auto dbstream = std::make_unique<DBSTREAM>(
        epsilon_, lambda_, mu_, beta_noise_, max_clusters_, eta_
    );

    std::cout << "Starting DBSTREAM clustering..." << std::endl;
    for (size_t i = 0; i < n_samples; ++i) {
        size_t idx = pt.idx_vec->at(i);
        double now = GET_DOUBLE_TS(flows_->at(idx).first.ts_start);
        dbstream->Insert(pt.norm_data->col(i), now);
    }
    std::cout << std::endl;

    // 获取聚类结果
    size_t last_idx =pt.idx_vec->back();
    double final_ts = GET_DOUBLE_TS(flows_->at(last_idx).first.ts_start);
    auto macro_labels = dbstream->GlobalClusterLabels(final_ts, connection_threshold_);
    auto all_micro_clusters = dbstream->AllMicroClusters();

    pt.assignments->assign(n_samples, -1); 
    const double thr2 = (epsilon_ * 1.5) * (epsilon_ * 1.5);
    for (size_t i = 0; i < n_samples; ++i) {
        const arma::vec& x = pt.norm_data->col(i);
        double best_d2 = std::numeric_limits<double>::infinity();
        int best_label = -1;

        for (const auto& mc : all_micro_clusters) {
            // 用平方距离，少一次 sqrt
            const double d2 = arma::accu(arma::square(x - mc->center));
            if (d2 <= thr2 && d2 < best_d2) {
                if (auto it = macro_labels.find(mc.get()); it != macro_labels.end()) {
                    best_d2 = d2;
                    best_label = it->second;
                }
            }
        }

        if (best_label != -1) {
            (*pt.assignments)[i] = best_label;
        }
    }

    // 1) 统计各簇频次（忽略 -1）
    std::unordered_map<int, size_t> freq;
    freq.reserve(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        int lbl = (*pt.assignments)[i];
        if (lbl != -1) ++freq[lbl];
    }

    // 2) 找出占比最大的簇（若全是 -1，则 majority 仍为 -1）
    int majority = -1;
    size_t max_cnt = 0;
    for (const auto& kv : freq) {
        if (kv.second > max_cnt) {
            max_cnt = kv.second;
            majority = kv.first;
        }
    }

    // 3) 将“最大簇”与“未分配(-1)”合并为异常
    //   （如需只追加异常，可去掉 clear；若希望重置异常集合，保留 clear）
    pt.outlier->clear();
    for (size_t i = 0; i < n_samples; ++i) {
        int lbl = (*pt.assignments)[i];
        // if (lbl == -1 || lbl == majority) {
        //     pt.outlier->insert(pt.idx_vec->at(i));  // 记录为异常（用全局索引）
        // }
        if (lbl != -1 || lbl == majority) {
            pt.outlier->insert(pt.idx_vec->at(i));  // 记录为异常（用全局索引）
        }
    }

    return;
}


// void MixDetector::Detect() {
//     using std::cout;
//     using std::endl;

//     if (flows_->empty()) {
//         cout << "No training flows_.\n";
//         return;
//     }

//     Time start_time = to_time_point(flows_->front().first.ts_start);
//     const size_t total = flows_->size();
//     size_t count = 0;
//     const size_t print_interval = 1000;
//     const size_t kNumPartitions = 100;

//     // === 签名字段索引（确保与你的 SubgraphProfiler::finalizeSignature 顺序一致）===
//     enum SigIdx : size_t {
//         F1_flows_cnt = 0,
//         F5_bw_p95    = 4,
//         F6_pps_p95   = 5,
//         G3_role_imb  = 9,
//         P1_proto_H   = 16,

//         // KL_self（若要使用可读；本实现仅用 KL_global）
//         KL_bw_self   = 18,
//         KL_pps_self  = 19,
//         KL_pkt_self  = 20,
//         KL_rar_self  = 21,
//         KL_scn_self  = 22,
//         KL_pr_self   = 23,

//         // 关键：KL_global（本次打分使用它们）
//         KL_bw_global   = 24,
//         KL_pps_global  = 25,
//         KL_pkt_global  = 26,
//         KL_rar_global  = 27,
//         KL_scn_global  = 28,
//         KL_pr_global   = 29
//     };

//     // 直方图/分箱配置（与 SubgraphProfiler 一致）
//     SubgraphProfileConfig s_cfg;
//     s_cfg.bw_hi  = 14.0;
//     s_cfg.pps_hi = 14.0;
//     s_cfg.pkt_hi = 2000.0;

//     // === 初始化分区 ===
//     std::vector<PartitionData> partition_data(kNumPartitions);
//     for (size_t i = 0; i < kNumPartitions; ++i) {
//         auto& P = partition_data[i];
//         P.pid = static_cast<int>(i);
//         P.bloom_filter = std::make_unique<BloomFilter>(1u << 21, 8);
//         P.extractor    = std::make_unique<GraphFeatureExtractor>(config_, start_time);
//         P.profiler     = std::make_unique<SubgraphProfiler>(s_cfg);

//         P.sample_vec   = std::make_shared<std::vector<arma::vec>>();
//         P.norm_data    = std::make_unique<arma::mat>();
//         P.idx_vec      = std::make_unique<std::vector<size_t>>();
//         P.outlier      = std::make_unique<std::unordered_set<size_t>>();
//         P.assignments  = std::make_unique<std::vector<size_t>>();
//     }

//     // === 分区选择（Bloom-LDG）===
//     auto select_partition_fn = [&](const FlowRecord& flow) -> size_t {
//         std::vector<size_t> neighbor_cnt(kNumPartitions, 0);
//         for (size_t i = 0; i < kNumPartitions; ++i) {
//             if (partition_data[i].bloom_filter->contains(flow.src_ip)) neighbor_cnt[i]++;
//             if (partition_data[i].bloom_filter->contains(flow.dst_ip)) neighbor_cnt[i]++;
//         }
//         const double lambda = 1.0;
//         int best = 0;
//         double max_score = -1e18;
//         size_t max_cap = std::max_element(
//             partition_data.begin(), partition_data.end(),
//             [](const PartitionData& a, const PartitionData& b){ return a.load < b.load; }
//         )->load + 1;

//         for (size_t i = 0; i < kNumPartitions; ++i) {
//             double score = static_cast<double>(neighbor_cnt[i])
//                          - lambda * (double(partition_data[i].load) / double(max_cap));
//             if (score > max_score) { max_score = score; best = static_cast<int>(i); }
//         }
//         return static_cast<size_t>(best);
//     };

//     // ========== Stage A：LDG 分割 + 特征提取（只缓存） ==========
//     size_t idx = 0;
//     for (const auto& pr : *flows_) {
//         const FlowRecord& flow = pr.first;

//         size_t pid = select_partition_fn(flow);
//         auto& P = partition_data[pid];

//         P.extractor->updateGraph(flow);
//         arma::vec v = P.extractor->extract(flow);

//         P.bloom_filter->insert(flow.src_ip);
//         P.bloom_filter->insert(flow.dst_ip);

//         if (v.is_empty()) continue;

//         P.sample_vec->emplace_back(v);
//         P.idx_vec->emplace_back(idx++);
//         ++P.load;

//         if (++count % print_interval == 0 || count == total) {
//             cout << "\rPartitioning & feature extraction... "
//                  << count << " / " << total << " samples." << std::flush;
//         }
//     }
//     cout << endl;

//     // ========== Stage B：批量聚合（非分布统计） ==========
//     // 仅保留非空分区
//     std::vector<size_t> nonempty; nonempty.reserve(kNumPartitions);
//     for (size_t i = 0; i < kNumPartitions; ++i) if (partition_data[i].load > 0) nonempty.push_back(i);
//     if (nonempty.empty()) { cout << "No non-empty partitions.\n"; return; }

//     // 先 buildFromSamples（不带 KL_global）
//     for (auto pid : nonempty) {
//         auto& P = partition_data[pid];
//         P.profiler->buildFromSamples(P.sample_vec);
//     }

//     // —— 选择 Top10% 候选子图 与 背景候选池（用于构造背景） ——
//     std::sort(nonempty.begin(), nonempty.end(), [&](size_t a, size_t b){
//         return partition_data[a].load > partition_data[b].load;
//     });
//     const size_t n_nonempty = nonempty.size();
//     const size_t n_top = std::max<size_t>(1, (size_t)std::floor(0.10 * n_nonempty));
//     std::vector<size_t> top_idx(nonempty.begin(), nonempty.begin() + std::min(n_top, n_nonempty));
//     std::vector<size_t> pool_idx(nonempty.begin() + std::min(n_top, n_nonempty), nonempty.end());

//     // ---- 工具：分位上的载荷阈值 ----
//     auto nth_load = [&](const std::vector<size_t>& idxs, double q)->size_t {
//         if (idxs.empty()) return 0;
//         std::vector<size_t> loads; loads.reserve(idxs.size());
//         for (auto id : idxs) loads.push_back(partition_data[id].load);
//         size_t k = (size_t)std::floor(std::clamp(q,0.0,1.0) * (loads.size()-1));
//         std::nth_element(loads.begin(), loads.begin()+k, loads.end());
//         return loads[k];
//     };
//     size_t q20 = nth_load(pool_idx, 0.20);
//     size_t q80 = nth_load(pool_idx, 0.80);

//     // ---- 均匀度（多维熵平均） ----
//     auto protoPdfFromCounts = [](const std::unordered_map<int,size_t>& cnt)->arma::vec {
//         if (cnt.empty()) return arma::vec(1, arma::fill::ones);
//         int maxk = 0; for (auto& kv:cnt) maxk = std::max(maxk, kv.first);
//         arma::vec p(maxk+1, arma::fill::zeros);
//         for (auto& kv:cnt) p(kv.first) = kv.second;
//         p += 1e-12; p /= arma::accu(p);
//         return p;
//     };
//     auto uhist_pdf = [&](std::shared_ptr<std::vector<arma::vec>> samples,
//                          size_t bins, double lo, double hi, size_t feat_idx)->arma::vec {
//         Histogram1D h(bins, lo, hi);
//         for (const auto& v : *samples) { if (v.n_elem > feat_idx) h.add(v(feat_idx)); }
//         return h.pdf();
//     };
//     auto u_proto_pdf = [&](std::shared_ptr<std::vector<arma::vec>> samples)->arma::vec {
//         std::unordered_map<int,size_t> cnt;
//         for (const auto& v : *samples) if (v.n_elem > 13) ++cnt[(int)v(13)];
//         return protoPdfFromCounts(cnt);
//     };
//     auto entropy01 = [&](const arma::vec& p)->double {
//         if (p.n_elem==0) return 0.0;
//         arma::vec q = p + 1e-12; q /= arma::accu(q);
//         double H = -arma::accu(q % arma::log(q));
//         double Hmax = std::log((double)q.n_elem);
//         return (Hmax>0)? (H/Hmax) : 0.0;
//     };

//     // 计算背景候选池中每个分区的均匀度
//     std::vector<double> uniformity(kNumPartitions, 0.0);
//     for (auto pid : pool_idx) {
//         const auto& P = partition_data[pid];
//         if (P.sample_vec->empty()) continue;

//         arma::vec Pbw  = uhist_pdf(P.sample_vec, s_cfg.bins, s_cfg.bw_lo,  s_cfg.bw_hi,  10);
//         arma::vec Ppps = uhist_pdf(P.sample_vec, s_cfg.bins, s_cfg.pps_lo, s_cfg.pps_hi, 11);
//         arma::vec Ppkt = uhist_pdf(P.sample_vec, s_cfg.bins, s_cfg.pkt_lo, s_cfg.pkt_hi, 12);
//         arma::vec Prar = uhist_pdf(P.sample_vec, s_cfg.bins, s_cfg.rar_lo, s_cfg.rar_hi, 19);
//         arma::vec Pscn = uhist_pdf(P.sample_vec, s_cfg.bins, s_cfg.scn_lo, s_cfg.scn_hi, 18);
//         arma::vec Ppr  = u_proto_pdf(P.sample_vec);

//         double u = 0.0;
//         u += entropy01(Pbw);
//         u += entropy01(Ppps);
//         u += entropy01(Ppkt);
//         u += entropy01(Prar);
//         u += entropy01(Pscn);
//         u += entropy01(Ppr);
//         uniformity[pid] = u / 6.0;
//     }

//     // 20–80 分位 + 均匀度阈值 0.70 过滤背景；不够再放宽
//     double tau_bg = 0.70;
//     std::vector<size_t> bg_idx; bg_idx.reserve(pool_idx.size());
//     for (auto pid : pool_idx) {
//         size_t load = partition_data[pid].load;
//         if (load >= q20 && load <= q80 && uniformity[pid] >= tau_bg)
//             bg_idx.push_back(pid);
//     }
//     if (bg_idx.size() < std::max<size_t>(10, n_nonempty/10)) {
//         size_t q10 = nth_load(pool_idx, 0.10);
//         size_t q90 = nth_load(pool_idx, 0.90);
//         bg_idx.clear();
//         for (auto pid : pool_idx) {
//             size_t load = partition_data[pid].load;
//             if (load >= q10 && load <= q90 && uniformity[pid] >= 0.60)
//                 bg_idx.push_back(pid);
//         }
//         if (bg_idx.empty()) bg_idx = pool_idx;
//     }

//     // —— 构造“均匀背景”参考分布 ——（只算一次）
//     Histogram1D bg_bw (s_cfg.bins, s_cfg.bw_lo,  s_cfg.bw_hi);
//     Histogram1D bg_pps(s_cfg.bins, s_cfg.pps_lo, s_cfg.pps_hi);
//     Histogram1D bg_pkt(s_cfg.bins, s_cfg.pkt_lo, s_cfg.pkt_hi);
//     Histogram1D bg_rar(s_cfg.bins, s_cfg.rar_lo, s_cfg.rar_hi);
//     Histogram1D bg_scn(s_cfg.bins, s_cfg.scn_lo, s_cfg.scn_hi);
//     std::unordered_map<int, size_t> bg_proto_cnt;

//     for (auto pid : bg_idx) {
//         const auto& P = partition_data[pid];
//         for (const auto& v : *P.sample_vec) {
//             if (v.n_elem < 20) continue;
//             bg_bw .add( v(10) );
//             bg_pps.add( v(11) );
//             bg_pkt.add( v(12) );
//             bg_rar.add( v(19) );
//             bg_scn.add( std::clamp((double)v(18), 0.0, 1.0) );
//             ++bg_proto_cnt[(int)v(13)];
//         }
//     }
//     const arma::vec B_bw  = bg_bw.pdf();
//     const arma::vec B_pps = bg_pps.pdf();
//     const arma::vec B_pkt = bg_pkt.pdf();
//     const arma::vec B_rar = bg_rar.pdf();
//     const arma::vec B_scn = bg_scn.pdf();
//     const arma::vec B_pr  = protoPdfFromCounts(bg_proto_cnt);

//     // —— 把背景注入每个 profiler，并“统一”产出含 KL_global 的签名 ——
//     for (auto pid : nonempty) {
//         auto& P = partition_data[pid];
//         P.profiler->setGlobalRefBw  (B_bw);
//         P.profiler->setGlobalRefPps (B_pps);
//         P.profiler->setGlobalRefPkt (B_pkt);
//         P.profiler->setGlobalRefRar (B_rar);
//         P.profiler->setGlobalRefScn (B_scn);
//         P.profiler->setGlobalRefProto(bg_proto_cnt);

//         P.signature = P.profiler->finalizeSignature(/*include_global=*/true);
//     }

//     // ========== Stage C：打分（稳健 z + ReLU²） ==========
//     auto collect_subset = [&](const std::vector<size_t>& idxs, auto getter)->std::vector<double>{
//         std::vector<double> xs; xs.reserve(idxs.size());
//         for (auto pid : idxs) {
//             double v = getter(partition_data[pid]);
//             if (std::isfinite(v)) xs.push_back(v);
//         }
//         if (xs.empty()) xs.push_back(0.0);
//         return xs;
//     };

//     // KL 维度直接来自 signature (KL_global)
//     auto xs_kl_bw    = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>KL_bw_global)? P.signature(KL_bw_global) : 0.0; });
//     auto xs_kl_pps   = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>KL_pps_global)? P.signature(KL_pps_global) : 0.0; });
//     auto xs_kl_pkt   = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>KL_pkt_global)? P.signature(KL_pkt_global) : 0.0; });
//     auto xs_kl_rar   = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>KL_rar_global)? P.signature(KL_rar_global) : 0.0; });
//     auto xs_kl_scn   = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>KL_scn_global)? P.signature(KL_scn_global) : 0.0; });
//     auto xs_kl_proto = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>KL_pr_global)? P.signature(KL_pr_global) : 0.0; });

//     // 非分布参考项（还是从 signature 读）
//     auto xs_bw95  = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>F5_bw_p95)? P.signature(F5_bw_p95):0.0; });
//     auto xs_pps95 = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>F6_pps_p95)? P.signature(F6_pps_p95):0.0; });
//     auto xs_role  = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>G3_role_imb)? P.signature(G3_role_imb):0.0; });
//     auto xs_H     = collect_subset(nonempty, [&](const PartitionData& P){
//         return (!P.signature.is_empty() && P.signature.n_elem>P1_proto_H)? P.signature(P1_proto_H):0.0; });

//     // 稳健 z-score + 兜底
//     auto zscore_robust = [&](const std::vector<double>& xs, double x) {
//         auto median_of = [&](const std::vector<double>& a){
//             std::vector<double> t; t.reserve(a.size());
//             for (double v : a) if (std::isfinite(v)) t.push_back(v);
//             if (t.empty()) return 0.0;
//             size_t n = t.size();
//             std::nth_element(t.begin(), t.begin()+n/2, t.end());
//             double med = t[n/2];
//             if ((n & 1) == 0) { auto it = std::max_element(t.begin(), t.begin()+n/2); med = 0.5*(med + *it); }
//             return med;
//         };
//         auto mad_of = [&](const std::vector<double>& a, double med){
//             std::vector<double> d; d.reserve(a.size());
//             for (double v : a) if (std::isfinite(v)) d.push_back(std::abs(v - med));
//             if (d.empty()) return 0.0;
//             size_t n = d.size();
//             std::nth_element(d.begin(), d.begin()+n/2, d.end());
//             double mad = d[n/2];
//             if ((n & 1) == 0) { auto it = std::max_element(d.begin(), d.begin()+n/2); mad = 0.5*(mad + *it); }
//             return 1.4826 * mad;
//         };

//         double med = median_of(xs);
//         double mad = mad_of(xs, med);
//         if (std::isfinite(x) && mad >= 1e-12) return (x - med) / mad;

//         // STD 兜底
//         double mean = 0.0, var = 0.0;
//         {   double s1=0.0, s2=0.0; size_t n=0;
//             for (double v : xs) if (std::isfinite(v)) { s1 += v; s2 += v*v; ++n; }
//             if (n > 0) { mean = s1/n; var = std::max(0.0, s2/n - mean*mean); }
//         }
//         double sd = std::sqrt(var);
//         if (std::isfinite(x) && sd >= 1e-12) return (x - mean) / sd;

//         if (!std::isfinite(x)) return 0.0;
//         return (x > med) ? 1.0 : ((x < med) ? -1.0 : 0.0);
//     };
//     auto relu = [](double t){ return t > 0.0 ? t : 0.0; };
//     auto contrib = [&](const std::vector<double>& xs, double x, double w, double dir){
//         double z = zscore_robust(xs, x) * dir;
//         double a = relu(z);
//         return w * a * a;
//     };

//     // 权重
//     const double w_kl_bw=1.0, w_kl_pps=1.0, w_kl_pkt=0.6, w_kl_rar=0.8, w_kl_scn=0.8, w_kl_pr=0.8;
//     const double w_bw=0.4,  w_pps=0.4,  w_role=0.2, w_H=0.2;

//     // 计算分数
//     for (auto& P : partition_data) {
//         if (P.load <= 100) { P.score = 0.0; continue; }

//         const auto KLbw  = (!P.signature.is_empty() && P.signature.n_elem>KL_bw_global  ) ? P.signature(KL_bw_global)   : 0.0;
//         const auto KLpps = (!P.signature.is_empty() && P.signature.n_elem>KL_pps_global ) ? P.signature(KL_pps_global)  : 0.0;
//         const auto KLpkt = (!P.signature.is_empty() && P.signature.n_elem>KL_pkt_global ) ? P.signature(KL_pkt_global)  : 0.0;
//         const auto KLrar = (!P.signature.is_empty() && P.signature.n_elem>KL_rar_global ) ? P.signature(KL_rar_global)  : 0.0;
//         const auto KLscn = (!P.signature.is_empty() && P.signature.n_elem>KL_scn_global ) ? P.signature(KL_scn_global)  : 0.0;
//         const auto KLpr  = (!P.signature.is_empty() && P.signature.n_elem>KL_pr_global  ) ? P.signature(KL_pr_global)   : 0.0;

//         double S = 0.0;
//         S += contrib(xs_kl_bw,    KLbw,  w_kl_bw,   +1.0);
//         S += contrib(xs_kl_pps,   KLpps, w_kl_pps,  +1.0);
//         S += contrib(xs_kl_pkt,   KLpkt, w_kl_pkt,  +1.0);
//         S += contrib(xs_kl_rar,   KLrar, w_kl_rar,  +1.0);
//         S += contrib(xs_kl_scn,   KLscn, w_kl_scn,  +1.0);
//         S += contrib(xs_kl_proto, KLpr,  w_kl_pr,   +1.0);

//         S += contrib(xs_bw95,  (!P.signature.is_empty()? P.signature(F5_bw_p95):0.0),  w_bw,  +1.0);
//         S += contrib(xs_pps95, (!P.signature.is_empty()? P.signature(F6_pps_p95):0.0), w_pps, +1.0);
//         S += contrib(xs_role,  (!P.signature.is_empty()? P.signature(G3_role_imb):0.0),w_role,+1.0);
//         S += contrib(xs_H,     (!P.signature.is_empty()? P.signature(P1_proto_H):0.0), w_H,   -1.0); // 熵低更可疑

//         P.score = S;
//     }

//     // ===== 打印（仅非空）=====
//     cout << "\n=== First-level Partition Stats ===\n";
//     cout << std::setw(12) << "Partition" << " | "
//          << std::setw(8)  << "Samples"   << " | "
//          << std::setw(6)  << "Pos"       << " | "
//          << std::setw(6)  << "Neg"       << " | "
//          << std::setw(8)  << "bw_p95"    << " | "
//          << std::setw(8)  << "pps_p95"   << " | "
//          << std::setw(8)  << "KL_bw"     << " | "
//          << std::setw(8)  << "KL_pps"    << " | "
//          << std::setw(8)  << "KL_proto"  << " | "
//          << std::setw(10) << "Score"
//          << "\n";
//     cout << std::string(100, '-') << "\n";

//     for (size_t pid : nonempty) {
//         const auto& P = partition_data[pid];
//         size_t samples = P.sample_vec->size();
//         size_t pos = std::count_if(P.idx_vec->begin(), P.idx_vec->end(), [&](size_t i){
//             return flows_->at(i).second;
//         });
//         size_t neg = samples - pos;

//         auto bw95   = (!P.signature.is_empty() && P.signature.n_elem > F5_bw_p95)  ? P.signature(F5_bw_p95)  : 0.0;
//         auto pps95  = (!P.signature.is_empty() && P.signature.n_elem > F6_pps_p95) ? P.signature(F6_pps_p95) : 0.0;
//         auto klbw   = (!P.signature.is_empty() && P.signature.n_elem > KL_bw_global)? P.signature(KL_bw_global):0.0;
//         auto klpps  = (!P.signature.is_empty() && P.signature.n_elem > KL_pps_global)?P.signature(KL_pps_global):0.0;
//         auto klpr   = (!P.signature.is_empty() && P.signature.n_elem > KL_pr_global)? P.signature(KL_pr_global):0.0;

//         std::ostringstream part_id; part_id << "root-" << pid;

//         cout << std::setw(12) << std::left << part_id.str() << " | "
//              << std::setw(8)  << samples << " | "
//              << std::setw(6)  << pos << " | "
//              << std::setw(6)  << neg << " | "
//              << std::setw(8)  << std::fixed << std::setprecision(2) << bw95  << " | "
//              << std::setw(8)  << std::fixed << std::setprecision(2) << pps95 << " | "
//              << std::setw(8)  << std::fixed << std::setprecision(3) << klbw  << " | "
//              << std::setw(8)  << std::fixed << std::setprecision(3) << klpps << " | "
//              << std::setw(8)  << std::fixed << std::setprecision(3) << klpr  << " | "
//              << std::setw(10) << std::fixed << std::setprecision(3) << P.score
//              << "\n";
//     }
//     cout << std::string(100, '=') << "\n";

//     // ===== 动态阈值选嫌疑分区（只在非空里）=====
//     std::vector<double> scores; scores.reserve(nonempty.size());
//     for (auto pid : nonempty) scores.push_back(partition_data[pid].score);

//     auto median_of = [&](std::vector<double> v){
//         if (v.empty()) return 0.0;
//         size_t n = v.size();
//         std::nth_element(v.begin(), v.begin()+n/2, v.end());
//         double med = v[n/2];
//         if ((n & 1) == 0) { auto it = std::max_element(v.begin(), v.begin()+n/2); med = 0.5*(med + *it); }
//         return med;
//     };
//     auto mad_of = [&](const std::vector<double>& v, double med){
//         if (v.empty()) return 0.0;
//         std::vector<double> d; d.reserve(v.size());
//         for (double x : v) if (std::isfinite(x)) d.push_back(std::abs(x - med));
//         if (d.empty()) return 0.0;
//         size_t n = d.size();
//         std::nth_element(d.begin(), d.begin()+n/2, d.end());
//         double mad = d[n/2];
//         if ((n & 1) == 0) { auto it = std::max_element(d.begin(), d.begin()+n/2); mad = 0.5*(mad + *it); }
//         return 1.4826 * mad;
//     };
//     auto quantile = [&](std::vector<double> v, double q) {
//         if (v.empty()) return 0.0;
//         q = std::clamp(q, 0.0, 1.0);
//         size_t k = (size_t)std::floor(q * (v.size()-1));
//         std::nth_element(v.begin(), v.begin()+k, v.end());
//         return v[k];
//     };

//     double med = median_of(scores);
//     double mad = mad_of(scores, med);
//     double thr_mad = med + 3.0 * mad;
//     double thr_q95 = quantile(scores, 0.95);
//     double tau = std::max(thr_mad, thr_q95);

//     std::vector<size_t> ord = nonempty;
//     std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b){
//         return partition_data[a].score > partition_data[b].score;
//     });

//     std::vector<size_t> sel;
//     for (auto pid : ord) if (partition_data[pid].score >= tau) sel.push_back(pid);

//     const size_t min_k = std::max<size_t>(5, std::max<size_t>(1, n_nonempty / 100));
//     const size_t max_k = std::max<size_t>(1, (size_t)std::floor(0.10 * n_nonempty));
//     if (sel.size() < min_k) {
//         sel.assign(ord.begin(), ord.begin() + std::min(min_k, ord.size()));
//     } else if (sel.size() > max_k) {
//         sel.resize(max_k);
//     }

//     std::cout << "Suspicious subgraphs (dynamic threshold, tau="
//               << std::fixed << std::setprecision(3) << tau
//               << ", selected=" << sel.size() << " / " << n_nonempty << "):\n";
//     for (size_t i = 0; i < sel.size(); ++i) {
//         size_t pid = sel[i];
//         const auto& P = partition_data[pid];
//         std::cout << "  #" << (i+1)
//                   << "  pid=" << pid
//                   << "  samples=" << P.load
//                   << "  score=" << std::fixed << std::setprecision(3) << P.score
//                   << "\n";
//     }

//     // ===== Evaluate only on selected partitions =====
//     size_t TP = 0, FP = 0, FN = 0, TN = 0;

//     std::vector<char> is_sel(kNumPartitions, 0);
//     for (auto pid : sel) is_sel[pid] = 1;

//     // 选中分区：跑检测
//     for (auto pid : sel) {
//         auto& pt = partition_data[pid];
//         if (pt.load == 0) continue;

//         execDBstreamDetect(pt);
//         execPCAProccess(pt);

//         for (const auto& i_flow : *pt.idx_vec) {
//             const size_t pred  = pt.outlier->count(i_flow) ? 1u : 0u;
//             const size_t label = static_cast<size_t>(flows_->at(i_flow).second);

//             TP += (pred & label);
//             FP += (pred & (1u - label));
//             TN += ((1u - pred) & (1u - label));
//             FN += ((1u - pred) & label);
//         }
//     }

//     // 非选中分区：pred=0
//     for (size_t pid = 0; pid < kNumPartitions; ++pid) {
//         if (is_sel[pid] || partition_data[pid].load == 0) continue;
//         const auto& pt = partition_data[pid];

//         size_t pos = 0;
//         for (const auto& i_flow : *pt.idx_vec) pos += static_cast<size_t>(flows_->at(i_flow).second);
//         const size_t neg = pt.idx_vec->size() - pos;

//         TN += neg;
//         FN += pos;
//     }

//     size_t sum       = TP + TN + FP + FN;
//     double accuracy  = (double)(TP + TN) / std::max<size_t>(1, sum);
//     double precision = (TP + FP) ? (double)TP / (TP + FP) : 0.0;
//     double recall    = (TP + FN) ? (double)TP / (TP + FN) : 0.0;
//     double fpr       = (FP + TN) ? (double)FP / (FP + TN) : 0.0;
//     double f1        = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;

//     cout << "📊 Final Evaluation (DBSTREAM):\n";
//     cout << "✅ Accuracy  : " << accuracy * 100 << "%\n";
//     cout << "🎯 Precision : " << precision * 100 << "%\n";
//     cout << "📥 Recall    : " << recall * 100 << "%\n";
//     cout << "🚨 FPR       : " << fpr * 100 << "%\n";
//     cout << "📈 F1-Score  : " << f1 * 100 << "%\n";

//     cout << "\n📊 Confusion Matrix:\n";
//     cout << "Predicted       0        1\n";
//     cout << "Actual 0 | " << std::setw(6) << TN << "  | " << std::setw(6) << FP << "\n";
//     cout << "Actual 1 | " << std::setw(6) << FN << "  | " << std::setw(6) << TP << "\n";
// }


void MixDetector::Detect() {
    using std::cout;
    using std::endl;

    if (flows_->empty()) {
        cout << "No training flows_.\n";
        return;
    }

    Time start_time = to_time_point(flows_->front().first.ts_start);
    const size_t total = flows_->size();
    size_t count = 0;
    const size_t print_interval = 1000;
    const size_t kNumPartitions = 100;

    // 与 SubgraphProfiler::finalizeSignature 的索引（打印用）
    enum SigIdx : size_t {
        F1_flows_cnt = 0,
        F5_bw_p95    = 4,
        F6_pps_p95   = 5,
        G3_role_imb  = 9,
        P1_proto_H   = 16
    };

    // 直方图/分箱配置（与 SubgraphProfiler 一致）
    SubgraphProfileConfig s_cfg;
    s_cfg.bw_hi  = 14.0;
    s_cfg.pps_hi = 14.0;
    s_cfg.pkt_hi = 2000.0;

    // 初始化分区
    std::vector<PartitionData> partition_data(kNumPartitions);
    for (size_t i = 0; i < kNumPartitions; ++i) {
        auto& P = partition_data[i];
        P.pid = static_cast<int>(i);
        P.bloom_filter = std::make_unique<BloomFilter>(1u << 21, 8);
        P.extractor    = std::make_unique<GraphFeatureExtractor>(config_, start_time);
        P.profiler     = std::make_unique<SubgraphProfiler>(s_cfg);

        P.sample_vec   = std::make_shared<std::vector<arma::vec>>();
        P.norm_data    = std::make_unique<arma::mat>();
        P.idx_vec      = std::make_unique<std::vector<size_t>>();
        P.outlier      = std::make_unique<std::unordered_set<size_t>>();
        P.assignments  = std::make_unique<std::vector<size_t>>();
        P.score        = 0.0; // 将用于承载 iForest 分数
    }

    // 分区选择（Bloom-LDG）
    auto select_partition_fn = [&](const FlowRecord& flow) -> size_t {
        std::vector<size_t> neighbor_cnt(kNumPartitions, 0);
        for (size_t i = 0; i < kNumPartitions; ++i) {
            if (partition_data[i].bloom_filter->contains(flow.src_ip)) neighbor_cnt[i]++;
            if (partition_data[i].bloom_filter->contains(flow.dst_ip)) neighbor_cnt[i]++;
        }
        const double lambda = 1.0;
        int best = 0;
        double max_score = -1e18;
        size_t max_cap = std::max_element(
            partition_data.begin(), partition_data.end(),
            [](const PartitionData& a, const PartitionData& b){ return a.load < b.load; }
        )->load + 1;

        for (size_t i = 0; i < kNumPartitions; ++i) {
            double score = static_cast<double>(neighbor_cnt[i])
                         - lambda * (double(partition_data[i].load) / double(max_cap));
            if (score > max_score) { max_score = score; best = static_cast<int>(i); }
        }
        return static_cast<size_t>(best);
    };

    // ========== Stage A：LDG 分割 + 特征提取（只缓存） ==========
    size_t idx = 0;
    for (const auto& pr : *flows_) {
        const FlowRecord& flow = pr.first;

        size_t pid = select_partition_fn(flow);
        auto& P = partition_data[pid];

        P.extractor->updateGraph(flow);
        arma::vec v = P.extractor->extract(flow);

        P.bloom_filter->insert(flow.src_ip);
        P.bloom_filter->insert(flow.dst_ip);

        if (v.is_empty()) continue;

        P.sample_vec->emplace_back(v);
        P.idx_vec->emplace_back(idx++);
        ++P.load;

        if (++count % print_interval == 0 || count == total) {
            cout << "\rPartitioning & feature extraction... "
                 << count << " / " << total << " samples." << std::flush;
        }
    }
    cout << endl;

    // 仅保留非空分区
    std::vector<size_t> nonempty; nonempty.reserve(kNumPartitions);
    for (size_t i = 0; i < kNumPartitions; ++i) {
        if (partition_data[i].load > 0) nonempty.push_back(i);
    }
    if (nonempty.empty()) {
        cout << "No non-empty partitions.\n";
        return;
    }

    // ========== Stage B：每子图批量聚合（取 signature 前 18 维） ==========
    for (auto pid : nonempty) {
        auto& P = partition_data[pid];
        P.profiler->buildFromSamples(P.sample_vec);
        P.signature = P.profiler->finalizeSignature(/*include_global=*/false); // 这里包含 18 基础维 + 若干 KLself，但我们只用前 18 维
    }

    // 过滤：仅在 load>=100 且 signature 维度>=18 的分区上跑 iForest
    std::vector<size_t> used; used.reserve(nonempty.size());
    for (auto pid : nonempty) {
        const auto& P = partition_data[pid];
        if (P.load >= 100 && !P.signature.is_empty() && P.signature.n_elem >= 18) {
            used.push_back(pid);
        }
    }
    if (used.empty()) {
        cout << "No partitions meet the condition (load>=100 & sig>=18).\n";
        return;
    }

    // 组装 18xN 特征矩阵（每列一个子图）
    arma::mat X(18, used.size(), arma::fill::zeros);
    for (size_t j = 0; j < used.size(); ++j) {
        const auto& sig = partition_data[used[j]].signature;
        X.col(j) = sig.head(18); // 仅 18 维统计量
    }

    // —— 鲁棒缩放：逐维中位数/MAD（提升 iForest 稳健性）——
    auto robust_scale = [&](arma::mat& M) {
        for (arma::uword r = 0; r < M.n_rows; ++r) {
            arma::rowvec v = M.row(r);
            double med = arma::median(v);
            arma::rowvec absdev = arma::abs(v - med);
            double mad = arma::median(absdev) * 1.4826; // 正态一致
            if (!(mad >= 1e-12)) mad = 1.0;
            M.row(r) = (M.row(r) - med) / mad;
        }
    };
    robust_scale(X);

    // —— 训练 Isolation Forest ——（使用你提供的实现）
    size_t nTrees     = 200;
    size_t sampleSize = std::min<size_t>(256, X.n_cols);
    size_t maxDepth   = (size_t)std::ceil(std::log2(std::max<size_t>(2, sampleSize)));
    IsolationForest ifor(/*seed=*/42, nTrees, sampleSize, maxDepth);
    ifor.Fit(X);

    // —— 得分（越大越异常）——
    std::vector<double> if_scores(used.size(), 0.0);
    for (size_t j = 0; j < used.size(); ++j) {
        if_scores[j] = ifor.AnomalyScore(X.col(j));
    }
    // 回填至分区
    for (size_t j = 0; j < used.size(); ++j) {
        partition_data[used[j]].score = if_scores[j];
    }

    // ========== 动态阈值（MAD 与 P95 更严格者） + 护栏 ==========
    auto quantile = [&](std::vector<double> v, double q){
        if (v.empty()) return 0.0;
        q = std::clamp(q, 0.0, 1.0);
        size_t k = (size_t)std::floor(q * (v.size()-1));
        std::nth_element(v.begin(), v.begin()+k, v.end());
        return v[k];
    };
    auto median_of = [&](std::vector<double> v){
        if (v.empty()) return 0.0;
        size_t n=v.size(); std::nth_element(v.begin(), v.begin()+n/2, v.end());
        double med=v[n/2]; if(!(n & 1)){auto it=std::max_element(v.begin(), v.begin()+n/2); med=0.5*(med+*it);}
        return med;
    };
    auto mad_of = [&](const std::vector<double>& v, double med){
        if (v.empty()) return 0.0;
        std::vector<double> d; d.reserve(v.size());
        for (double x : v) d.push_back(std::abs(x - med));
        size_t n=d.size(); std::nth_element(d.begin(), d.begin()+n/2, d.end());
        double mad=d[n/2]; if(!(n & 1)){auto it=std::max_element(d.begin(), d.begin()+n/2); mad=0.5*(mad+*it);}
        return 1.4826*mad;
    };

    double med = median_of(if_scores);
    double mad = mad_of(if_scores, med);
    double thr_mad = med + 3.0 * mad;        // 可调 2.5~4
    double thr_q95 = quantile(if_scores, 0.95);
    double tau = std::max(thr_mad, thr_q95);

    // 生成按分数降序的索引（仅 used）
    std::vector<size_t> ord = used;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b){
        return partition_data[a].score > partition_data[b].score;
    });

    // 先用阈值筛
    std::vector<size_t> sel;
    sel.reserve(ord.size());
    for (auto pid : ord) {
        if (partition_data[pid].score >= tau) sel.push_back(pid);
    }

    // 护栏：至少 min_k，至多 10%
    const size_t n_used = used.size();
    const size_t min_k  = std::max<size_t>(5, std::max<size_t>(1, n_used / 100)); // ≥5 或 ≥1%
    const size_t max_k  = std::max<size_t>(1, (size_t)std::floor(0.10 * n_used)); // ≤10%

    if (sel.size() < min_k) {
        sel.assign(ord.begin(), ord.begin() + std::min(min_k, ord.size()));
    } else if (sel.size() > max_k) {
        sel.resize(max_k);
    }

    // ===== 打印（仅非空）=====
    cout << "\n=== First-level Partition Stats ===\n";
    cout << std::setw(12) << "Partition" << " | "
         << std::setw(8)  << "Samples"   << " | "
         << std::setw(6)  << "Pos"       << " | "
         << std::setw(6)  << "Neg"       << " | "
         << std::setw(8)  << "bw_p95"    << " | "
         << std::setw(8)  << "pps_p95"   << " | "
         << std::setw(10) << "IF_score"
         << "\n";
    cout << std::string(80, '-') << "\n";

    for (size_t pid : nonempty) {
        const auto& P = partition_data[pid];
        size_t samples = P.sample_vec->size();
        size_t pos = std::count_if(P.idx_vec->begin(), P.idx_vec->end(), [&](size_t i){
            return flows_->at(i).second;
        });
        size_t neg = samples - pos;

        auto bw95   = (!P.signature.is_empty() && P.signature.n_elem > F5_bw_p95)  ? P.signature(F5_bw_p95)  : 0.0;
        auto pps95  = (!P.signature.is_empty() && P.signature.n_elem > F6_pps_p95) ? P.signature(F6_pps_p95) : 0.0;

        std::ostringstream part_id; part_id << "root-" << pid;

        cout << std::setw(12) << std::left << part_id.str() << " | "
             << std::setw(8)  << samples << " | "
             << std::setw(6)  << pos << " | "
             << std::setw(6)  << neg << " | "
             << std::setw(8)  << std::fixed << std::setprecision(2) << bw95  << " | "
             << std::setw(8)  << std::fixed << std::setprecision(2) << pps95 << " | "
             << std::setw(10) << std::fixed << std::setprecision(3) << P.score
             << "\n";
    }
    cout << std::string(80, '=') << "\n";

    // 打印选中的嫌疑分区
    std::cout << "Suspicious subgraphs (iForest, tau="
              << std::fixed << std::setprecision(3) << tau
              << ", selected=" << sel.size() << " / " << n_used << "):\n";
    for (size_t i = 0; i < sel.size(); ++i) {
        size_t pid = sel[i];
        const auto& P = partition_data[pid];
        std::cout << "  #" << (i+1)
                  << "  pid=" << pid
                  << "  samples=" << P.load
                  << "  score=" << std::fixed << std::setprecision(3) << P.score
                  << "\n";
    }

    // ===== Evaluate only on selected partitions =====
    size_t TP = 0, FP = 0, FN = 0, TN = 0;

    // 标记选中分区
    std::vector<char> is_sel(kNumPartitions, 0);
    for (auto pid : sel) is_sel[pid] = 1;

    // 1) 选中分区：跑 DBSTREAM/PCA（你的原函数）
    for (auto pid : sel) {
        auto& pt = partition_data[pid];
        if (pt.load == 0) continue;

        execDBstreamDetect(pt);
        execPCAProccess(pt);

        for (const auto& i_flow : *pt.idx_vec) {
            const size_t pred  = pt.outlier->count(i_flow) ? 1u : 0u;
            const size_t label = static_cast<size_t>(flows_->at(i_flow).second);

            TP += (pred & label);
            FP += (pred & (1u - label));
            TN += ((1u - pred) & (1u - label));
            FN += ((1u - pred) & label);
        }
    }

    // 2) 非选中（含 load<100 或 sig<18 的，或被 iForest 排除的）：统一 pred=0
    for (size_t pid = 0; pid < kNumPartitions; ++pid) {
        if (partition_data[pid].load == 0 || is_sel[pid]) continue;
        const auto& pt = partition_data[pid];

        size_t pos = 0;
        for (const auto& i_flow : *pt.idx_vec) pos += static_cast<size_t>(flows_->at(i_flow).second);
        const size_t neg = pt.idx_vec->size() - pos;

        TN += neg;   // pred=0 & label=0
        FN += pos;   // pred=0 & label=1
    }

    // 汇总指标
    size_t sum       = TP + TN + FP + FN;
    double accuracy  = (double)(TP + TN) / std::max<size_t>(1, sum);
    double precision = (TP + FP) ? (double)TP / (TP + FP) : 0.0;
    double recall    = (TP + FN) ? (double)TP / (TP + FN) : 0.0;
    double fpr       = (FP + TN) ? (double)FP / (FP + TN) : 0.0;
    double f1        = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;

    cout << "📊 Final Evaluation (iForest→DBSTREAM on selected):\n";
    cout << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    cout << "🎯 Precision : " << precision * 100 << "%\n";
    cout << "📥 Recall    : " << recall * 100 << "%\n";
    cout << "🚨 FPR       : " << fpr * 100 << "%\n";
    cout << "📈 F1-Score  : " << f1 * 100 << "%\n";

    cout << "\n📊 Confusion Matrix:\n";
    cout << "Predicted       0        1\n";
    cout << "Actual 0 | " << std::setw(6) << TN << "  | " << std::setw(6) << FP << "\n";
    cout << "Actual 1 | " << std::setw(6) << FN << "  | " << std::setw(6) << TP << "\n";
}


void MixDetector::Train(){

}

void MixDetector::run() {
    Detect();
}