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
        bool is_outlier = pt.outlier->count(idx) > 0;

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
        if (lbl != -1 || lbl == majority) {
            pt.outlier->insert(pt.idx_vec->at(i));  // 记录为异常（用全局索引）
        }
    }

    return;
}

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

    // ===== 分箱配置（与 SubgraphProfileConfig 一致）=====
    SubgraphProfileConfig s_cfg;
    s_cfg.bins   = 32;
    s_cfg.bw_hi  = 14.0;
    s_cfg.pps_hi = 14.0;
    s_cfg.pkt_hi = 2000.0;
    s_cfg.deg_hi = 14.0;
    s_cfg.imb_hi = 14.0;

    // ===== 初始化分区 =====
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
        P.score        = 0.0; // iForest 分数
    }

    // ===== 分区选择（Bloom-LDG）=====
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

    // ===== Stage A：LDG 分割 + 特征提取（缓存样本）=====
    size_t idx = 0;
    for (const auto& pr : *flows_) {
        const FlowRecord& flow = pr.first;

        size_t pid = select_partition_fn(flow);
        auto& P = partition_data[pid];

        P.extractor->updateGraph(flow);
        arma::vec v = P.extractor->extract(flow);

        P.bloom_filter->insert(flow.src_ip);
        P.bloom_filter->insert(flow.dst_ip);
        ++P.load;

        if (v.is_empty()) continue;

        P.sample_vec->emplace_back(v);
        P.idx_vec->emplace_back(idx++);

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

    // ===== Stage B：子图聚合（使用新 24 维 signature）=====
    for (auto pid : nonempty) {
        auto& P = partition_data[pid];
        P.profiler->buildFromSamples(P.sample_vec);
        P.signature = P.profiler->finalizeSignature(); // 24 维
    }

    // 使用特征名动态获取维度 d（与 SubgraphProfiler::featureNames() 完全对齐）
    const auto featNames = SubgraphProfiler::featureNames();
    const arma::uword d = static_cast<arma::uword>(featNames.size()); // 24

    // 过滤
    std::vector<size_t> used; used.reserve(nonempty.size());
    for (auto pid : nonempty) {
        const auto& P = partition_data[pid];
        if (P.load != 0 && !P.signature.is_empty() && P.signature.n_elem == d) {
            used.push_back(pid);
        }
    }
    if (used.empty()) {
        cout << "No partitions meet the condition (load>=100 & sig==d).\n";
        return;
    }

    // 组装 d x N 特征矩阵（每列一个子图），使用 24 维
    arma::mat X(d, used.size(), arma::fill::zeros);
    for (size_t j = 0; j < used.size(); ++j) {
        const auto& sig = partition_data[used[j]].signature;
        X.col(j) = sig.head(d);
    }

    mlpack::data::MinMaxScaler scaler;
    scaler.Fit(X);
    arma::mat X_scaled(X.n_rows, X.n_cols);
    scaler.Transform(X, X_scaled);

    // ========= Pairwise Jensen–Shannon divergence (JS) ==========
    // JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M=(P+Q)/2
    auto js_div = [](const arma::vec& p_raw, const arma::vec& q_raw) -> double {
        const double eps = 1e-12;
        if (p_raw.n_elem == 0 || q_raw.n_elem == 0) return 0.0;

        // 对齐长度（理论上相等；若不等则截断到较短长度，或扩充）
        arma::uword n = std::min(p_raw.n_elem, q_raw.n_elem);
        arma::vec p = p_raw.head(n) + eps;
        arma::vec q = q_raw.head(n) + eps;
        p /= arma::accu(p);
        q /= arma::accu(q);
        arma::vec m = 0.5 * (p + q);

        // KL(p||m) 与 KL(q||m)
        arma::vec kl_pm = p % (arma::log(p) - arma::log(m));
        arma::vec kl_qm = q % (arma::log(q) - arma::log(m));
        double js = 0.5 * arma::accu(kl_pm) + 0.5 * arma::accu(kl_qm);

        // 数值护栏
        if (!std::isfinite(js) || js < 0) js = 0.0;
        return js;
    };

    // 你可以按需改权重；默认等权
    struct JSParts {
        double w_bw   = 1.0, w_pps  = 1.0, w_pkt  = 1.0, w_rar = 1.0, w_scn = 1.0;
        double w_sdeg = 1.0, w_ddeg = 1.0, w_proto= 1.0;
    } w;

    // 计算 N×N 的 pairwise JS 距离矩阵的上三角，并聚合为每个分区的异常分数
    const size_t N = used.size();
    std::vector<double> js_scores(N, 0.0);

    // 选择一种鲁棒聚合（中位数或 90 分位）
    auto quantile = [](std::vector<double> v, double q){
        if (v.empty()) return 0.0;
        q = std::clamp(q, 0.0, 1.0);
        size_t k = (size_t)std::floor(q * (v.size()-1));
        std::nth_element(v.begin(), v.begin()+k, v.end());
        return v[k];
    };

    for (size_t a = 0; a < N; ++a) {
        auto& Pa = partition_data[used[a]];
        std::vector<double> row_js; row_js.reserve(N-1);

        // 预取 a 的分布
        arma::vec a_bw   = Pa.profiler->dist_bw();
        arma::vec a_pps  = Pa.profiler->dist_pps();
        arma::vec a_pkt  = Pa.profiler->dist_pkt();
        arma::vec a_rar  = Pa.profiler->dist_rar();
        arma::vec a_scn  = Pa.profiler->dist_scn();
        const arma::vec& a_sdeg = Pa.profiler->dist_srcdeg();
        const arma::vec& a_ddeg = Pa.profiler->dist_dstdeg();
        const arma::vec& a_prot = Pa.profiler->dist_proto();

        for (size_t b = 0; b < N; ++b) if (a != b) {
            auto& Pb = partition_data[used[b]];

            // 预取 b 的分布
            arma::vec b_bw   = Pb.profiler->dist_bw();
            arma::vec b_pps  = Pb.profiler->dist_pps();
            arma::vec b_pkt  = Pb.profiler->dist_pkt();
            arma::vec b_rar  = Pb.profiler->dist_rar();
            arma::vec b_scn  = Pb.profiler->dist_scn();
            const arma::vec& b_sdeg = Pb.profiler->dist_srcdeg();
            const arma::vec& b_ddeg = Pb.profiler->dist_dstdeg();
            const arma::vec& b_prot = Pb.profiler->dist_proto();

            // 分布级 JS，最后做加权平均
            double js_bw   = js_div(a_bw,   b_bw);
            double js_pps  = js_div(a_pps,  b_pps);
            double js_pkt  = js_div(a_pkt,  b_pkt);
            double js_rar  = js_div(a_rar,  b_rar);
            double js_scn  = js_div(a_scn,  b_scn);
            double js_sdeg = js_div(a_sdeg, b_sdeg);
            double js_ddeg = js_div(a_ddeg, b_ddeg);
            double js_prot = js_div(a_prot, b_prot);

            double wsum = w.w_bw + w.w_pps + w.w_pkt + w.w_rar + w.w_scn
                        + w.w_sdeg + w.w_ddeg + w.w_proto;
            double js_ab = (w.w_bw*js_bw + w.w_pps*js_pps + w.w_pkt*js_pkt +
                            w.w_rar*js_rar + w.w_scn*js_scn +
                            w.w_sdeg*js_sdeg + w.w_ddeg*js_ddeg +
                            w.w_proto*js_prot) / std::max(1e-12, wsum);

            row_js.push_back(js_ab);
        }

        js_scores[a] = quantile(row_js, 0.5); // median
    }

    // 写回到 partition 分数字段
    for (size_t j = 0; j < N; ++j) {
        partition_data[used[j]].score = js_scores[j];
    }

    // ===== 动态阈值（MAD 与 P95 取更严格者）=====
    auto median_of = [&](std::vector<double> v){
        if (v.empty()) return 0.0;
        size_t n=v.size();
        std::nth_element(v.begin(), v.begin()+n/2, v.end());
        double m2 = v[n/2];
        if (n & 1) return m2;
        std::nth_element(v.begin(), v.begin()+n/2-1, v.end());
        double m1 = v[n/2-1];
        return 0.5*(m1+m2);
    };
    auto mad_of = [&](const std::vector<double>& v, double med){
        if (v.empty()) return 0.0;
        std::vector<double> d; d.reserve(v.size());
        for (double x : v) d.push_back(std::abs(x - med));
        size_t n=d.size();
        std::nth_element(d.begin(), d.begin()+n/2, d.end());
        double md2 = d[n/2];
        if (n & 1) return 1.4826*md2;
        std::nth_element(d.begin(), d.begin()+n/2-1, d.end());
        double md1 = d[n/2-1];
        return 1.4826*0.5*(md1+md2);
    };

    double med = median_of(js_scores);
    double mad = mad_of(js_scores, med);
    double thr_mad = med + 2.5 * mad;        // 可调 2.5~4
    double thr_q95 = quantile(js_scores, 0.95);
    double tau = std::max(thr_mad, thr_q95);

    // 按分数降序
    std::vector<size_t> ord = used;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b){
        return partition_data[a].score > partition_data[b].score;
    });

    // 初筛
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

    // 动态表头
    cout << std::setw(12) << "Partition" << " | "
        << std::setw(8)  << "Samples"   << " | "
        << std::setw(6)  << "Pos"       << " | "
        << std::setw(6)  << "Neg";

    for (const auto& name : featNames) {
        cout << " | " << std::setw(10) << name;
    }
    cout << " | " << std::setw(10) << "Score" << "\n";

    cout << std::string(12+8+6+6 + (featNames.size()+1)*13, '-') << "\n";

    // 排序（异常分数降序）
    std::vector<size_t> sorted_nonempty = nonempty;
    std::sort(sorted_nonempty.begin(), sorted_nonempty.end(),
        [&](size_t a, size_t b) {
            return partition_data[a].score > partition_data[b].score;
        });

    // 遍历分区
    for (size_t pid : sorted_nonempty) {
        const auto& P = partition_data[pid];
        size_t samples = P.sample_vec->size();
        size_t pos = std::count_if(P.idx_vec->begin(), P.idx_vec->end(), [&](size_t i){
            return flows_->at(i).second;
        });
        size_t neg = samples - pos;

        std::ostringstream part_id; part_id << "root-" << pid;

        cout << std::setw(12) << std::left << part_id.str() << " | "
            << std::setw(8)  << samples << " | "
            << std::setw(6)  << pos << " | "
            << std::setw(6)  << neg;

        // 动态输出每个特征
        if (!P.signature.is_empty()) {
            for (size_t i = 0; i < featNames.size(); ++i) {
                double val = (P.signature.n_elem > i) ? P.signature(i) : 0.0;
                cout << " | " << std::setw(10) << std::fixed << std::setprecision(3) << val;
            }
        } else {
            for (size_t i = 0; i < featNames.size(); ++i)
                cout << " | " << std::setw(10) << "NA";
        }

        cout << " | " << std::setw(10) << std::fixed << std::setprecision(3) << P.score
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

    // 2) 非选中：统一 pred=0
    for (size_t pid = 0; pid < kNumPartitions; ++pid) {
        if (partition_data[pid].load == 0 || is_sel[pid]) continue;
        const auto& pt = partition_data[pid];

        size_t pos = 0;
        for (const auto& i_flow : *pt.idx_vec) pos += static_cast<size_t>(flows_->at(i_flow).second);
        const size_t neg = pt.idx_vec->size() - pos;

        TN += neg;   // pred=0 & label=0
        FN += pos;   // pred=0 & label=1
    }

    // ===== 汇总指标 =====
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