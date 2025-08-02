#include "detector.hpp"

namespace fs = filesystem;


void MiniBatchKMeansDetector::Train() {
    if (trained_ || sample_vecs_.empty()) return;

    arma::mat data(sample_vecs_[0].n_rows, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i)
        data.col(i) = sample_vecs_[i];

    // 训练 MiniBatchKMeans
    mbk_.Train(data);

    trained_ = true;

    for (size_t i = 0; i < data.n_cols; ++i) {
        arma::vec x = data.col(i);
        auto [cluster, dist] = mbk_.Predict(x);
        cluster_dists_[cluster].push_back(dist);
    }

    // ==== 计算每簇的距离统计（均值 + 标准差） ====
    cluster_mean_.clear();
    cluster_stddev_.clear();
    for (const auto& [cluster, dists] : cluster_dists_) {
        double mean = arma::mean(arma::vec(dists));
        double stddev = arma::stddev(arma::vec(dists));
        cluster_mean_[cluster] = mean;
        cluster_stddev_[cluster] = stddev;
    }

    cluster_thresholds_.clear();
    for (const auto& [cluster, dists] : cluster_dists_) {
        if (dists.size() < 5) {  // 小样本簇使用全局阈值
            cluster_thresholds_[cluster] = global_threshold_;
            continue;
        }
        
        // 计算95%分位数
        std::vector<double> sorted_dists = dists;
        std::sort(sorted_dists.begin(), sorted_dists.end());
        size_t idx = std::min(static_cast<size_t>(sorted_dists.size() * 0.995), sorted_dists.size()-1);
        cluster_thresholds_[cluster] = sorted_dists[idx];
    }

    std::cout << "📊 Cluster Thresholds:\n";
    for (size_t i = 0; i < cluster_thresholds_.size(); ++i) {
        std::cout << "  - Cluster " << i << ": Threshold = " << cluster_thresholds_[i] << "\n";
    }
}


size_t MiniBatchKMeansDetector::Predict(const arma::vec& X) {
    const auto [cluster, dist] = mbk_.Predict(X);

    auto it = cluster_thresholds_.find(cluster);
    if (it != cluster_thresholds_.end() && dist > it->second) {
        return 1;
    }

    return 0;
}

void MiniBatchKMeansDetector::run_detection(void) {
    const auto data_path = loader_->getDataPath();
    const auto& test_flows = *loader_->getTestData();
    size_t TP = 0, TN = 0, FP = 0, FN = 0;

    fs::create_directory("result");
    string current_time = get_current_time_str();
    string base = fs::path(data_path).stem().string();
    string output_file = "result/" + base + "_mini-batch-kmeans_pred_" + current_time + ".csv";
    string metric_file = "result/" + base + "_mini-batch-kmeans_metrics_" + current_time + ".txt";

    ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Pred,Label\n";

    cout << "\n🔎 Running Mini-Batch-KMeans Detection:\n";

    Time start_time = to_time_point(test_flows.front().first.ts_start);
    auto graphExtractor = std::make_unique<GraphFeatureExtractor>(config_, start_time);
    auto flowExtractor = std::make_unique<FlowFeatureExtractor>();

    for (const auto& pair : test_flows) {
        const FlowRecord& flow = pair.first;
        arma::vec flowVec = flowExtractor->extract(flow);
        arma::vec graphVec = graphExtractor->extract(flow);

        arma::vec feat = arma::join_vert(flowVec, graphVec);
        size_t pred = Predict(feat);
        size_t label = pair.second;

        if (pred == 1 && label == 1) TP++;
        else if (pred == 0 && label == 0) TN++;
        else if (pred == 1 && label == 0) FP++;
        else if (pred == 0 && label == 1) FN++;

        ofs << flow.src_ip << "," << flow.dst_ip << "," << pred << "," << label << "\n";
    }

    ofs.close();

    size_t total = TP + TN + FP + FN;
    double accuracy = (double)(TP + TN) / total;
    double precision = (TP + FP) ? (double)TP / (TP + FP) : 0.0;
    double recall = (TP + FN) ? (double)TP / (TP + FN) : 0.0;
    double f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;
    double f2 = (precision + recall) ? 5 * precision * recall / (4 * precision + recall) : 0.0;

    cout << "\n📊 Evaluation Metrics:\n";
    cout << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    cout << "🎯 Precision : " << precision * 100 << "%\n";
    cout << "📥 Recall    : " << recall * 100 << "%\n";
    cout << "📈 F1-Score  : " << f1 * 100 << "%\n";
    cout << "📈 F2-Score  : " << f2 * 100 << "%\n";

    cout << "\n📊 Confusion Matrix:\n";
    cout << "             Predicted\n";
    cout << "            0        1\n";
    cout << "Actual 0 | " << setw(6) << TN << "  | " << setw(6) << FP << "\n";
    cout << "Actual 1 | " << setw(6) << FN << "  | " << setw(6) << TP << "\n";
    cout << "\n📁 Results written to: " << output_file << "\n";

    ofstream mfs(metric_file);
    mfs << fixed << setprecision(4);

    mfs << "📊 Evaluation Metrics:\n";
    mfs << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    mfs << "🎯 Precision : " << precision * 100 << "%\n";
    mfs << "📥 Recall    : " << recall * 100 << "%\n";
    mfs << "📈 F1-Score  : " << f1 * 100 << "%\n";
    mfs << "📈 F2-Score  : " << f2 * 100 << "%\n\n";

    mfs << "📊 Confusion Matrix:\n";
    mfs << "             Predicted\n";
    mfs << "            0        1\n";
    mfs << "Actual 0 | " << setw(6) << TN << "  | " << setw(6) << FP << "\n";
    mfs << "Actual 1 | " << setw(6) << FN << "  | " << setw(6) << TP << "\n";

    mfs.close();
    cout << "📄 Metrics written to: " << metric_file << "\n";
}


void MiniBatchKMeansDetector::run(void) {
    const auto& train_flows = *loader_->getTrainData();

    size_t total = train_flows.size();
    size_t count = 0;
    size_t print_interval = 1000;

    Time start_time = to_time_point(train_flows.front().first.ts_start);
    auto graphExtractor = std::make_unique<GraphFeatureExtractor>(config_, start_time);
    auto flowExtractor = std::make_unique<FlowFeatureExtractor>();

    for (const auto &[flow, label] : train_flows)
    {
        if(label) continue;
        graphExtractor->updateGraph(flow);
        arma::vec flowVec = flowExtractor->extract(flow);
        arma::vec graphVec = graphExtractor->extract(flow);

        if (flowVec.is_empty() || graphVec.is_empty()) continue;
        addSample(arma::join_vert(flowVec, graphVec));

        if (++count % print_interval == 0 || count == total) {
            cout << "\rProcessed " << count << " / " << total << " samples." << flush;
        }
    }
    cout << endl; 

    cout << " 🔄 Start train: " << "\n";
    
    Train();

    PrintClusteDetail();

    run_detection();

    SaveTrainClusterResult("train_clusters.csv");
    SaveTestAbnormalClusterResult("test_clusters.csv");

    // PerformPCAVisualization();
}


void MiniBatchKMeansDetector::PerformPCAVisualization() {
    if (sample_vecs_.empty()) {
        std::cerr << "No samples available for PCA." << std::endl;
        return;
    }

    const auto& train_flows = *loader_->getTrainData();
    
    // 验证样本一致性
    if (sample_vecs_.size() != train_flows.size()) {
        std::cerr << "Sample count mismatch: vectors=" << sample_vecs_.size()
                  << " labels=" << train_flows.size() << std::endl;
        return;
    }

    // 构建数据矩阵：每列一个样本
    arma::mat data(sample_vecs_[0].n_elem, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i) {
        data.col(i) = sample_vecs_[i];  // 列存储样本
    }

    // 特征缩放
    mlpack::data::MinMaxScaler scaler;
    arma::mat norm_data;
    scaler.Fit(data);
    scaler.Transform(data, norm_data);

    // 执行PCA：输入[特征数×样本数]，输出[2×样本数]
    arma::mat transformedData;
    mlpack::PCA pca;
    pca.Apply(norm_data, transformedData, 2);

    // 准备输出矩阵：每行一个样本，列分别为PCA1, PCA2, Cluster
    arma::mat final_output(transformedData.n_cols, 3);  // [N, 3]
    
    // 填充PCA结果（转置后每行对应一个样本）
    final_output.col(0) = transformedData.row(0).t();
    final_output.col(1) = transformedData.row(1).t();
    
    // 填充聚类标签
    for (size_t i = 0; i < transformedData.n_cols; ++i) {
        final_output(i, 2) = train_flows[i].second;
    }

    final_output = final_output.t();

    // 保存结果
    std::string filename = "pca_result.csv";
    if (!mlpack::data::Save(filename, final_output)) {
        std::cerr << "❌ Failed to save PCA output to " << filename << std::endl;
    } else {
        std::cout << "✅ PCA result saved to " << filename << std::endl;
        std::cout << "Saving matrix shape: " << final_output.n_rows 
                  << " x " << final_output.n_cols << std::endl;
        std::cout << "👉 CSV format: [PCA1, PCA2, Cluster] per row." << std::endl;
    }
}

void MiniBatchKMeansDetector::PrintClusteDetail() {    
    cluster_counts_.clear();

    std::cout << "📊 Cluster Statistics:\n";
    for (size_t i = 0; i < mbk_.K(); ++i) {
        auto it = cluster_dists_.find(i);
        if (it == cluster_dists_.end()) {
            std::cout << "Cluster " << i << ": (empty)\n";
            continue;
        }

        const std::vector<double>& dists = it->second;
        arma::vec arma_dists(dists);

        double mean = arma::mean(arma_dists);
        double stddev = arma::stddev(arma_dists);
        double min_dist = arma::min(arma_dists);
        double max_dist = arma::max(arma_dists);
        size_t count = it->second.size();

        std::cout << "🔹 Cluster " << i
                  << " | Size: " << count
                  << " | Mean: " << mean
                  << " | StdDev: " << stddev
                  << " | Min: " << min_dist
                  << " | Max: " << max_dist
                  << '\n';
    }
}

void MiniBatchKMeansDetector::SaveTrainClusterResult(const std::string& filename) {
    std::ofstream ofs(filename);
    ofs << "Index,Cluster,Distance\n";

    size_t idx = 0;
    for (const auto& [cluster, dists] : cluster_dists_) {
        for (double dist : dists) {
            ofs << idx++ << "," << cluster << "," << dist << "\n";
        }
    }

    std::cout << "✅ Saved training cluster results to: " << filename << "\n";
}

void MiniBatchKMeansDetector::SaveTestAbnormalClusterResult(const std::string& filename) {
    const auto& test_flows = *loader_->getTestData();
    std::ofstream ofs(filename);
    ofs << "Index,SrcIP,DstIP,Cluster,Distance,Label\n";

    Time start_time = to_time_point(test_flows.front().first.ts_start);
    auto graphExtractor = std::make_unique<GraphFeatureExtractor>(config_, start_time);
    auto flowExtractor = std::make_unique<FlowFeatureExtractor>();

    size_t index = 0;
    for (const auto& pair : test_flows) {
        const FlowRecord& flow = pair.first;
        size_t label = pair.second;

        arma::vec flowVec = flowExtractor->extract(flow);
        arma::vec graphVec = graphExtractor->extract(flow);
        arma::vec feat = arma::join_vert(flowVec, graphVec);
        auto [cluster, dist] = mbk_.Predict(feat);

        ofs << index++ << ","
            << flow.src_ip << "," << flow.dst_ip << ","
            << cluster << "," << dist << "," << label << "\n";
    }

    std::cout << "✅ Saved test cluster results to: " << filename << "\n";
}