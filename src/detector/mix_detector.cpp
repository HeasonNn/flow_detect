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

    auto graphExtractor = std::make_unique<GraphFeatureExtractor>(config_);

    for (const auto &[flow, label] : train_flows) {
        graphExtractor->advance_time(GET_DOUBLE_TS(flow.ts_start));
        graphExtractor->updateGraph(flow);
        arma::vec graphVec = graphExtractor->extract(flow);

        if (graphVec.is_empty()) continue;
        addSample(graphVec);

        if (++count % print_interval == 0 || count == total) {
            cout << "\rProcessed " << count << " / " << total << " samples." << flush;
        }
    }
    cout << endl; 
    printSamples();

    stream::DBSTREAM dbstream(epsilon_, lambda_, mu_, beta_merge_, beta_noise_, max_clusters_);

    arma::mat flow_data(sample_vecs_[0].n_elem, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i)
        flow_data.col(i) = sample_vecs_[i];
    
    scaler_.Fit(flow_data);
    arma::mat norm_data;
    scaler_.Transform(flow_data, norm_data);

    std::vector<double> timestamps;
    for (const auto &[flow, label] : train_flows) 
        timestamps.emplace_back(GET_DOUBLE_TS(flow.ts_start));
    
    for (size_t i = 0; i < norm_data.n_cols; ++i)
        dbstream.Insert(norm_data.col(i), timestamps[i]);

    auto core_clusters = dbstream.GetClusters(timestamps.back());
    
    unordered_map<size_t, vector<pair<size_t, double>>> cluster_points;
    unordered_map<size_t, double> point_to_dist;
    unordered_map<size_t, size_t> point_to_cluster;
    std::vector<int> assignments(norm_data.n_cols, -1);

    for (size_t i = 0; i < norm_data.n_cols; ++i) {
        const arma::vec& pt = norm_data.col(i);
        double best_dist = epsilon_;  // 定义 epsilon 内才归类
        int best_idx = -1;

        for (size_t j = 0; j < core_clusters.size(); ++j) {
            double dist = arma::norm(pt - core_clusters[j].center);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }

        if (best_idx >= 0) {
            assignments[i] = best_idx;
            cluster_points[best_idx].emplace_back(i, best_dist);
            point_to_dist[i] = best_dist;
            point_to_cluster[i] = best_idx;
        }
    }

    // === Step 4: 异常点检测逻辑（仿照 DBSCAN） ===
    unordered_set<size_t> outlier_ids;

    // === 4.1: 每个簇中最远的点 ===
    for (const auto& [cid, points] : cluster_points) {
        auto max_it = std::max_element(points.begin(), points.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
        outlier_ids.insert(max_it->first);  // 最远点为异常
    }

    // === 4.2: 找出最离散簇（平均距离最大）===
    double max_avg_dist = -1.0;
    size_t extreme_cluster = SIZE_MAX;
    for (const auto& [cid, points] : cluster_points) {
        double total = 0.0;
        for (const auto& [_, dist] : points) total += dist;
        double avg = total / points.size();
        if (avg > max_avg_dist) {
            max_avg_dist = avg;
            extreme_cluster = cid;
        }
    }

    // === 4.3: 将最离散簇的所有点标记为异常 ===
    if (cluster_points.count(extreme_cluster)) {
        for (const auto& [pid, _] : cluster_points[extreme_cluster])
            outlier_ids.insert(pid);
    }


    // === 4.4: 标记极端密集但小型的簇为异常 ===
    const double dense_threshold = 0.05; // 可调：簇内平均距离小于此值
    const size_t min_cluster_size = 10;  // 可调：点数少于此认为是异常

    for (const auto& [cid, points] : cluster_points) {
        if (points.size() > min_cluster_size) continue;

        double total_dist = 0.0;
        for (const auto& [_, dist] : points) total_dist += dist;
        double avg_dist = total_dist / points.size();

        if (avg_dist < dense_threshold) {
            // 这是一个小且密集的簇 → 异常
            for (const auto& [pid, _] : points)
                outlier_ids.insert(pid);
        }
    }

    // === 4.5: 根据簇大小判断异常 ===
    const size_t too_small_threshold = 5;    // 过小簇点数阈值
    const size_t too_large_threshold = 500; // 过大簇点数阈值

    for (const auto& [cid, points] : cluster_points) {
        if (points.size() <= too_small_threshold || points.size() >= too_large_threshold) {
            // 将该簇中的所有点标记为异常
            for (const auto& [pid, _] : points)
                outlier_ids.insert(pid);
        }
    }

    // === Step 5: 评估 + 输出 ===
    fs::create_directory("result");
    std::string output_file = "result/dbstream_predict_center.csv";
    std::ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Dist,Pred,Label\n";

    size_t TP = 0, TN = 0, FP = 0, FN = 0;
    for (size_t i = 0; i < train_flows.size(); ++i) {
        const auto& [flow, label] = train_flows[i];
        double dist = point_to_dist.count(i) ? point_to_dist[i] : -1.0;
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
    double fpr       = (FP + TN) ? (double)FP / (FP + TN) : 0.0;
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