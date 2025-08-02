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
    beta_merge_   = dbstream_config.value("beta_merge", 2.0);
    beta_noise_   = dbstream_config.value("beta_noise", 1.5);
    max_clusters_ = dbstream_config.value("max_clusters", 500);

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
    std::cout << "  - beta_merge         = " << beta_merge_ << "\n";
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

    // === Step 1: 提取所有特征并构建数据集 ===
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

    // === Step 3: 使用修复后的 DBSTREAM 进行聚类 ===
    // 使用我们推荐的、修复后的参数
    stream::DBSTREAM dbstream(epsilon_, lambda_, mu_, beta_merge_, beta_noise_, max_clusters_);

    size_t total_samples = norm_data.n_cols;
    cout << "Starting DBSTREAM clustering..." << endl;
    for (size_t i = 0; i < norm_data.n_cols; ++i) {
        dbstream.Insert(norm_data.col(i), timestamps[i]);

        if (i % print_interval == 0 || i == total_samples - 1) {
            double progress = (static_cast<double>(i + 1) / total_samples) * 100.0;
            cout << "\rClustering progress: " << i + 1 << " / " << total_samples 
                << " (" << std::fixed << std::setprecision(1) << progress << "%)" << flush;
        }
    }
    cout << endl; // 在进度条结束后换行
    auto core_clusters = dbstream.GetClusters(timestamps.back());
    cout << "DBSTREAM found " << core_clusters.size() << " core clusters.\n";

    // ✅ 修复1: 使用 AllMicroClusters() 获取所有微簇，而非 GetClusters()
    // 原因：修复后的 DBSTREAM 中，beta_noise_ 被设置得很高（如5.0），用于有效清理内存。
    //       因此，GetClusters() 返回的只是衰减权重 >= mu_ 的簇，而许多“候选”但尚未成熟的核心簇不会被返回。
    //       我们需要检查所有现存的微簇来判断一个点是否“被吸收”。
    auto all_micro_clusters = dbstream.AllMicroClusters(); // 获取所有现存的微簇
    double current_timestamp = timestamps.back();

    // === Step 4: 基于 DBSTREAM 结果进行异常检测 ===
    unordered_set<size_t> outlier_ids;
    std::vector<int> assignments(norm_data.n_cols, -1); // -1 表示未分配（即异常）
    std::vector<double> distances_to_cluster(norm_data.n_cols, -1.0);

    // 遍历每个数据点
    for (size_t i = 0; i < norm_data.n_cols; ++i) {
        const arma::vec& pt = norm_data.col(i);
        double best_dist = epsilon_; // 使用 epsilon 作为吸收半径
        int best_idx = -1;

        // ✅ 修复2: 遍历所有微簇，而不仅仅是“核心”簇
        // 目标：判断点是否被任何一个微簇在 epsilon 邻域内吸收。
        for (size_t j = 0; j < all_micro_clusters.size(); ++j) {
            const auto& mc = all_micro_clusters[j];
            double dist = arma::norm(pt - mc.center);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<int>(j);
            }
        }

        if (best_idx >= 0) {
            // 点被某个微簇吸收
            assignments[i] = best_idx;
            distances_to_cluster[i] = best_dist;
        } else {
            // 点未被任何微簇吸收 → 直接标记为异常
            outlier_ids.insert(i);
        }
    }

    // === Step 5: 识别“弱”或“离散”微簇中的异常点 (可选增强) ===
    // 这个步骤可以根据需要保留，用于处理边界情况。
    // 但请注意，由于 beta_noise_ 设置得较高，很多“弱”簇在被识别前就已被清理。
    // 因此，这一步的必要性降低，但可以作为二次过滤。

    const double cluster_dispersion_threshold = 0.5; 
    const double cluster_strength_threshold = mu_ * 1.5; 

    for (size_t cid = 0; cid < all_micro_clusters.size(); ++cid) {
        const auto& cluster = all_micro_clusters[cid];
        double decayed_weight = dbstream.QueryDecayedWeight(cluster, current_timestamp);
        
        std::vector<std::pair<size_t, double>> points_in_cluster;
        for (size_t i = 0; i < assignments.size(); ++i) {
            if (assignments[i] == static_cast<int>(cid)) {
                points_in_cluster.emplace_back(i, distances_to_cluster[i]);
            }
        }

        if (points_in_cluster.size() < 2) continue; // 至少有两个点才计算离散度

        double total_dist = 0.0;
        for (const auto& [pid, dist] : points_in_cluster) total_dist += dist;
        double avg_dist = total_dist / points_in_cluster.size();

        // ✅ 修复3: 调整逻辑。如果簇很弱，但点都在中心，可能没问题。
        //          如果簇很离散，即使很强，也可能有问题。
        //          这里我们更关注“离散度”，因为它更能反映内部的一致性。
        if (avg_dist > cluster_dispersion_threshold) {
            auto max_it = std::max_element(points_in_cluster.begin(), points_in_cluster.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            // ✅ 优化: 只标记最远的1-2个点，而不是整个簇
            outlier_ids.insert(max_it->first);
            // 如果想更激进，可以标记距离大于 avg_dist + std_dev 的所有点
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
        double dist = distances_to_cluster[i];
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

    // === 降维可视化输出 ===
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
}

void MixDetector::run() {
    Detect();
}