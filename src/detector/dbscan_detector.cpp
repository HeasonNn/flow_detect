// dbscan.cpp

#include "detector.hpp"

namespace fs = filesystem;

void DBscanDetector::addSample(const arma::vec &sample) {
    sample_vecs_.push_back(sample);
}

void DBscanDetector::train() {
    if (sample_vecs_.empty()) {
        std::cerr << "No samples to train DBSCAN." << std::endl;
        return;
    }

    arma::mat data(sample_vecs_[0].n_elem, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i) {
        data.col(i) = sample_vecs_[i];
    }

    scaler_.Fit(data);
    scaler_.Transform(data, norm_train_data_);

    arma::mat centroids;
    dbscan_.Cluster(norm_train_data_, cluster_labels_, centroids);

    trained_ = true;
}

int DBscanDetector::predict(const arma::vec& sample) {
    if (!trained_) {
        std::cerr << "Model not trained!" << std::endl;
        return -1;
    }

    // 将输入样本标准化
    arma::mat sample_mat = sample;
    arma::mat norm_query;
    scaler_.Transform(sample_mat, norm_query);

    // 使用 mlpack 要求的输出格式
    std::vector<std::vector<size_t>> neighbors;
    std::vector<std::vector<double>> distances;

    // 初始化 RangeSearch 并执行搜索
    mlpack::RangeSearch<> range_search(norm_train_data_);
    range_search.Search(norm_query, epsilon_, neighbors, distances);  // epsilon_ 是 DBSCAN 的邻域半径

    // 检查是否有邻居
    if (!neighbors.empty() && !neighbors[0].empty()) {
        size_t neighbor_index = neighbors[0][0];  // 取第一个邻居
        return static_cast<int>(cluster_labels_[neighbor_index]);
    }

    return -1;  // 无邻居或异常情况
}

void DBscanDetector::run_detection(void) {
    const auto data_path = loader_->getDataPath();
    const auto test_flows = *loader_->getTestData();
    size_t TP = 0, TN = 0, FP = 0, FN = 0;

    // 输出路径准备
    fs::create_directory("result");
    string current_time = get_current_time_str();
    string base = fs::path(data_path).stem().string();
    string output_file = "result/" + base + "_dbscan_pred_" + current_time + ".csv";
    string metric_file = "result/" + base + "_dbscan_metrics_" + current_time + ".txt";

    ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Pred,Label\n";

    cout << "\n🔎 Running DBSCAN Detection:\n";

    for (const auto& pair : test_flows) {
        const FlowRecord& flow = pair.first;
        arma::vec flowVec = flowExtractor_->extract(flow);
        arma::vec graphVec = graphExtractor_->extract(flow.src_ip, flow.dst_ip);

        if (flowVec.n_elem == 0 || graphVec.n_elem == 0) continue;

        arma::vec feat = arma::join_vert(flowVec, graphVec);

        size_t pred_raw = predict(feat);
        size_t pred = (pred_raw == -1) ? 1 : 0;  // -1 表示异常，映射为 1
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

    // 保存评估指标
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

void DBscanDetector::printFeatures(void) const noexcept {
    std::cout << "Number of samples: " << sample_vecs_.size() << std::endl;
}

void DBscanDetector::run(void) {
        const auto train_flows = *loader_->getTrainData();

    size_t total = train_flows.size();
    size_t count = 0;
    size_t print_interval = 1000;

    for (const auto &[flow, label] : train_flows)
    {
        graphExtractor_->updateGraph(flow.src_ip, flow.dst_ip);
        arma::vec flowVec = flowExtractor_->extract(flow);
        arma::vec graphVec = graphExtractor_->extract(flow.src_ip, flow.dst_ip);

        if (flowVec.is_empty() || graphVec.is_empty()) continue;
        addSample(arma::join_vert(flowVec, graphVec));

        if (++count % print_interval == 0 || count == total) {
            cout << "\rProcessed " << count << " / " << total << " samples." << flush;
        }
    }
    cout << endl; 

    cout << " 🔄 Start train: " << "\n";
    train();

    run_detection();
}
