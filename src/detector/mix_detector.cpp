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
    epsilon_      = dbstream_config.value("epsilon", 0.05);
    lambda_       = dbstream_config.value("lambda", 0.01);
    mu_           = dbstream_config.value("mu", 3);
    beta_noise_   = dbstream_config.value("beta_noise", 1.5);
    max_clusters_ = dbstream_config.value("max_clusters", 500);
    eta_          = dbstream_config.value("eta", 0.1);

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
}

void MixDetector::Train(){

}

void MixDetector::Detect(){
    const auto& train_flows = *loader_->getTrainData();
    size_t total = train_flows.size();
    size_t count = 0;
    size_t print_interval = 1000;

    Time start_time = to_time_point(train_flows.front().first.ts_start);
    auto graphExtractor = std::make_unique<GraphFeatureExtractor>(config_, start_time);

    // === Step 1: 特征提取 ===
    std::vector<arma::vec> raw_samples;
    std::vector<double> timestamps;
    for (const auto &[flow, label] : train_flows) {
        graphExtractor->updateGraph(flow);
        arma::vec graphVec = graphExtractor->extract(flow);
        if (graphVec.is_empty()) continue;
        raw_samples.push_back(graphVec);
        timestamps.emplace_back(GET_DOUBLE_TS(flow.ts_start));
        if (++count % print_interval == 0 || count == total) {
            cout << "\rExtracting features... " << count << " / " << total << " samples." << flush;
        }
    }
    cout << endl;

    if (raw_samples.empty()) {
        cout << "No valid samples extracted. Exiting.\n";
        return;
    }

    // === Step 2: 数据标准化 ===
    arma::mat flow_data(raw_samples[0].n_elem, raw_samples.size());
    for (size_t i = 0; i < raw_samples.size(); ++i)
        flow_data.col(i) = raw_samples[i];

    scaler_.Fit(flow_data);
    arma::mat norm_data;
    scaler_.Transform(flow_data, norm_data);

    // === Step 3: DBSTREAM 聚类 ===
    DBSTREAM dbstream(epsilon_, lambda_, mu_, beta_noise_, max_clusters_, eta_);
    const size_t total_samples = norm_data.n_cols;
    std::cout << "Starting DBSTREAM clustering..." << std::endl;
    for (size_t i = 0; i < total_samples; ++i) {
        dbstream.Insert(norm_data.col(i), timestamps[i]);
        if (i % print_interval == 0 || i == total_samples - 1) {
            double progress = double(i + 1) / total_samples * 100.0;
            std::cout << "\rClustering progress: " << (i + 1) << " / " << total_samples
                      << " (" << std::fixed << std::setprecision(1) << progress << "%)" << std::flush;
        }
    }
    std::cout << std::endl;

    const double final_ts = timestamps.back();
    auto core_clusters = dbstream.CoreMicroClusters(final_ts);
    auto all_micro_clusters = dbstream.AllMicroClusters();
    auto macro_labels = dbstream.GlobalClusterLabels(final_ts, 0.3); // connection_threshold 可调整,论文中建议 2 维时为 0.3

    // === Step 4: assignment & 异常检测（按宏簇 label） ===
    std::unordered_set<size_t> outlier_ids;
    std::vector<int> assignments(norm_data.n_cols, -1); // -1 = 未分配
    std::vector<double> distances(norm_data.n_cols, -1.0);

    for (size_t i = 0; i < norm_data.n_cols; ++i) {
        const arma::vec& pt = norm_data.col(i);
        double min_dist = epsilon_ * 2; // 吸收半径
        const MicroCluster* best_mc = nullptr;
        for (const auto& mc : all_micro_clusters) {
            double d = arma::norm(pt - mc->center);
            if (d < min_dist) {
                min_dist = d;
                best_mc = mc.get();
            }
        }
        if (best_mc && macro_labels.count(best_mc)) {
            assignments[i] = macro_labels.at(best_mc); // 宏簇 label
            distances[i] = min_dist;
        } else {
            outlier_ids.insert(i); // 噪声点
        }
    }

    // === Step 5: 异常增强（可选：对离群簇再做一次孤立点检测） ===
    const double cluster_dispersion_threshold = 0.5;
    for (const auto& kv : macro_labels) {
        int macro_label = kv.second;
        std::vector<std::pair<size_t, double>> points_in_cluster;
        for (size_t i = 0; i < assignments.size(); ++i)
            if (assignments[i] == macro_label)
                points_in_cluster.emplace_back(i, distances[i]);
        if (points_in_cluster.size() < 2) continue;
        double avg_dist = 0.0;
        for (const auto& pd : points_in_cluster) avg_dist += pd.second;
        avg_dist /= points_in_cluster.size();
        if (avg_dist > cluster_dispersion_threshold) {
            auto max_it = std::max_element(points_in_cluster.begin(), points_in_cluster.end(),
                                           [](const auto& a, const auto& b) { return a.second < b.second; });
            outlier_ids.insert(max_it->first);
        }
    }

    // === Step 6: 评估与输出 ===
    fs::create_directory("result");
    std::string output_file = "result/dbstream_predict_center.csv";
    std::ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Dist,Pred,Label\n";
    size_t TP = 0, TN = 0, FP = 0, FN = 0;
    for (size_t i = 0; i < train_flows.size(); ++i) {
        const auto& [flow, label] = train_flows[i];
        double dist = distances[i];
        size_t pred = outlier_ids.count(i) ? 1 : 0;
        if (pred == 1 && label == 1) TP++;
        else if (pred == 0 && label == 0) TN++;
        else if (pred == 1 && label == 0) FP++;
        else if (pred == 0 && label == 1) FN++;
        ofs << flow.src_ip << "," << flow.dst_ip << "," << dist << "," << pred << "," << label << "\n";
    }
    ofs.close();

    double accuracy = (double)(TP + TN) / (TP + TN + FP + FN);
    double precision = TP + FP ? (double)TP / (TP + FP) : 0.0;
    double recall = TP + FN ? (double)TP / (TP + FN) : 0.0;
    double fpr = (FP + TN) ? (double)FP / (FP + TN) : 0.0;
    double f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;

    cout << "📊 Final Evaluation (DBSTREAM):\n";
    cout << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    cout << "🎯 Precision : " << precision * 100 << "%\n";
    cout << "📥 Recall    : " << recall * 100 << "%\n";
    cout << "🚨 FPR       : " << fpr * 100 << "%\n";
    cout << "📈 F1-Score  : " << f1 * 100 << "%\n";
    cout << "📁 Results written to: " << output_file << "\n";

    // === 可视化降维输出 ===
    mlpack::PCA pca;
    arma::mat reduced;
    pca.Apply(norm_data, reduced, 2);

    std::ofstream fout("result/dbstream_pca_result.csv");
    fout << "x,y,assignments,label,is_outlier\n";
    for (size_t i = 0; i < reduced.n_cols; ++i) {
        const size_t label = train_flows[i].second;
        bool is_outlier = outlier_ids.count(i) > 0;
        int assign = assignments[i];
        fout << reduced(0, i) << "," << reduced(1, i) << "," << assign << "," << label << "," << is_outlier << "\n";
    }
    fout.close();

    // === Debug 信息 ===
    std::cout << "\n=== Debug Info ===\n";
    std::cout << "⦿ 微簇总数       : " << dbstream.TotalMicroClusters() << "\n";
    std::cout << "⦿ 核心簇个数     : " << core_clusters.size() << "\n";
    std::cout << "⦿ 当前时间戳     : " << final_ts << "\n";
    std::cout << "⦿ 全局宏簇数     : " << std::set<int>([&]{
        std::set<int> s;
        for (auto& kv : macro_labels) s.insert(kv.second);
        return s;
    }()).size() << " 个宏簇\n";
    std::cout << "=====================\n\n";
}

void MixDetector::run() {
    Detect();
}