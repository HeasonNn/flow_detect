// dbscan.cpp

#include "detector.hpp"

namespace fs = filesystem;


void DBscanDetector::AddSample(const arma::vec &sample) {
    sample_vecs_.push_back(sample);
}


void DBscanDetector::Detect(const vector<pair<FlowRecord, size_t>>& flows) {
    if (flows.empty()) {
        cerr << "No flows to Detect." << endl;
        return;
    }

    // === Step 1: 构造特征矩阵 ===
    arma::mat data(sample_vecs_[0].n_elem, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i)
        data.col(i) = sample_vecs_[i];

    scaler_.Fit(data);
    arma::mat norm_data;
    scaler_.Transform(data, norm_data);

    // === Step 2: DBSCAN 聚类并获取中心 ===
    arma::Row<size_t> assignments;
    arma::mat centroids;
    mlpack::DBSCAN<> dbscan(epsilon_, minPoints_);
    dbscan.Cluster(norm_data, assignments, centroids);

    size_t num_clusters = centroids.n_cols;
    cout << "✅ Clustered into " << num_clusters << " clusters." << endl;

    // === Step 3: 计算每个点到最近簇心的距离 ===
    unordered_map<size_t, double> flowid_to_dist;
    vector<pair<size_t, double>> flowid_dist_vec(norm_data.n_cols);
    for (size_t i = 0; i < norm_data.n_cols; ++i) {
        const arma::vec& xi = norm_data.col(i);
        for (size_t c = 0; c < centroids.n_cols; ++c) {
            double dist = arma::norm(xi - centroids.col(c), 2);
            flowid_to_dist[i] = dist;
            flowid_dist_vec.emplace_back(i, dist);
        }
    }

    // === Step 4: 取 top-1% 作为异常 ===
    std::sort(flowid_dist_vec.begin(), flowid_dist_vec.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second;});
    size_t top_k = std::max((size_t)1, static_cast<size_t>(0.1 * flowid_dist_vec.size()));
    unordered_set<size_t> outlier_ids;
    for (size_t i = 0; i < top_k; ++i) {
        outlier_ids.insert(flowid_dist_vec[i].first);  // flow ID
    }

    // === Step 5: 输出与评估 ===
    fs::create_directory("result");
    string base = fs::path(loader_->getDataPath()).stem().string();
    string timestamp = get_current_time_str();
    string output_file = "result/" + base + "_predict_center_dbscan_" + timestamp + ".csv";

    ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Dist,Pred,Label\n";

    size_t TP = 0, TN = 0, FP = 0, FN = 0;
    for (size_t i = 0; i < flows.size(); ++i) {
        const auto& [flow, label] = flows[i];
        double dist = flowid_to_dist[i];
        size_t pred = outlier_ids.count(i) ? 1 : 0;

        if (pred == 1 && label == 1) TP++;
        else if (pred == 0 && label == 0) TN++;
        else if (pred == 1 && label == 0) FP++;
        else if (pred == 0 && label == 1) FN++;

        ofs << flow.src_ip << "," << flow.dst_ip << "," << dist << "," << pred << "," << label << "\n";
    }

    ofs.close();

    size_t total = TP + TN + FP + FN;
    double accuracy = (double)(TP + TN) / total;
    double precision = (TP + FP) ? (double)TP / (TP + FP) : 0.0;
    double recall = (TP + FN) ? (double)TP / (TP + FN) : 0.0;
    double f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;

    cout << "📊 Final Evaluation (Center Distance + DBSCAN):\n";
    cout << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    cout << "🎯 Precision : " << precision * 100 << "%\n";
    cout << "📥 Recall    : " << recall * 100 << "%\n";
    cout << "📈 F1-Score  : " << f1 * 100 << "%\n";
    cout << "📁 Results written to: " << output_file << "\n";


    mlpack::PCA pca;
    arma::mat reduced;
    pca.Apply(norm_data, reduced, 2);  // 降至2维

    // 保存到 CSV
    std::ofstream fout("result/dbscan_pca_result.csv");
    fout << "x,y,assignments,label\n";
    for (size_t i = 0; i < reduced.n_cols; ++i) {
        const size_t label = flows[i].second;

        int assignment = static_cast<int>(assignments[i]);
        if (assignment == static_cast<int>(SIZE_MAX)){
            assignment = -1;
        }
        
        fout << reduced(0, i) << "," << reduced(1, i) << "," << assignment << "," << label <<"\n";
    }
    fout.close();
}


void DBscanDetector::run(void) {
    const auto& all_flows = *loader_->getAllData();
    size_t total = all_flows.size();
    size_t count = 0;
    size_t print_interval = 1000;

    for (const auto &[flow, label] : all_flows) {
        graphExtractor_->updateGraph(flow.src_ip, flow.dst_ip);
        arma::vec flowVec = flowExtractor_->extract(flow);
        arma::vec graphVec = graphExtractor_->extract(flow.src_ip, flow.dst_ip);

        if (flowVec.is_empty() || graphVec.is_empty()) continue;
        AddSample(arma::join_vert(flowVec, graphVec));

        if (++count % print_interval == 0 || count == total) {
            cout << "\rProcessed " << count << " / " << total << " samples." << flush;
        }
    }
    cout << endl; 

    cout << "🔄 Start train: " << "\n";

    Detect(all_flows);
}