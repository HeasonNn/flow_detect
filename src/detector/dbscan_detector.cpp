// dbscan.cpp

#include "detector.hpp"

#include <iomanip>


namespace fs = filesystem;


void DBscanDetector::addSample(const arma::vec &sample) {
    sample_vecs_.push_back(sample);
}


void DBscanDetector::printSamples() const {
    if (sample_vecs_.empty()) {
        std::cout << "[printSamples] No sample vectors available.\n";
        return;
    }

    size_t dim = sample_vecs_[0].n_elem;
    size_t n_samples = sample_vecs_.size();

    arma::mat mat(dim, n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        mat.col(i) = sample_vecs_[i];
    }

    arma::rowvec mean = arma::mean(mat, 1).t();
    arma::rowvec max = arma::max(mat, 1).t();
    arma::rowvec median = arma::median(mat, 1).t();

    std::cout << "[printSamples] Sample Stats (per feature dimension):\n";
    std::cout << std::left
            << std::setw(8) << "Feature"
            << std::setw(15) << "Mean"
            << std::setw(15) << "Max"
            << std::setw(15) << "Median"
            << std::setw(15) << "Mode"
            << "\n";

    for (size_t i = 0; i < dim; ++i) {
        arma::vec values = mat.row(i).t();

        // === 众数计算 ===
        std::unordered_map<double, size_t> freq;
        for (size_t j = 0; j < values.n_elem; ++j) {
            double v = values[j];
            freq[v]++;
        }

        double mode = values[0];
        size_t max_count = 0;
        for (const auto& [val, count] : freq) {
            if (count > max_count) {
                max_count = count;
                mode = val;
            }
        }

    std::cout << std::left
              << std::setw(8) << i
              << std::setw(15) << std::fixed << std::setprecision(4) << mean[i]
              << std::setw(15) << std::fixed << std::setprecision(4) << max[i]
              << std::setw(15) << std::fixed << std::setprecision(4) << median[i]
              << std::setw(15) << std::fixed << std::setprecision(4) << mode
              << "\n";
    }
}


void DBscanDetector::detect(const vector<pair<FlowRecord, size_t>>& flows) {
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
    mlpack::DBSCAN<> dbscan(epsilon_, min_points_);
    dbscan.Cluster(norm_data, assignments, centroids);

    size_t num_clusters = centroids.n_cols;
    cout << "✅ Clustered into " << num_clusters << " clusters." << endl;

    // === Step 3: 计算每个点到其分配簇心的距离 ===
    unordered_map<size_t, vector<pair<size_t, double>>> cluster_points;  // cluster_id -> vector of (point_id, dist)
    unordered_map<size_t, double> point_to_dist;
    unordered_map<size_t, size_t> point_to_cluster;

    for (size_t i = 0; i < norm_data.n_cols; ++i) {
        size_t cid = assignments[i];
        if (cid == static_cast<size_t>(-1) || cid == SIZE_MAX) continue;  // ignore noise

        const arma::vec& xi = norm_data.col(i);
        double dist = arma::norm(xi - centroids.col(cid), 2);

        cluster_points[cid].emplace_back(i, dist);
        point_to_dist[i] = dist;
        point_to_cluster[i] = cid;
    }

    // === Step 4.1: 每个簇中最远的点 ===
    unordered_set<size_t> outlier_ids;
    for (const auto& [cid, points] : cluster_points) {
        auto max_it = std::max_element(
            points.begin(), points.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            }
        );
        outlier_ids.insert(max_it->first);  // point index
    }

    // === Step 4.2: 找到最“离散”的簇（簇内平均距离最大）===
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

    // === Step 4.3: 把这个簇所有点标记为异常 ===
    if (cluster_points.count(extreme_cluster)) {
        for (const auto& [pid, _] : cluster_points[extreme_cluster]) {
            outlier_ids.insert(pid);
        }
    }

    // === Step 5: 输出与评估 ===
    fs::create_directory("result");
    string base = fs::path(loader_->getDataPath()).stem().string();
    string timestamp = get_current_time_str();
    // string output_file = "result/" + base + "_predict_center_dbscan_" + timestamp + ".csv";
    string output_file = "result/" + base + "_predict_center_dbscan.csv";

    ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Dist,Pred,Label\n";

    size_t TP = 0, TN = 0, FP = 0, FN = 0;
    for (size_t i = 0; i < flows.size(); ++i) {
        const auto& [flow, label] = flows[i];
        double dist = point_to_dist.count(i) ? point_to_dist[i] : -1.0;
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
    fout << "x,y,assignments,label,is_outlier\n";
    for (size_t i = 0; i < reduced.n_cols; ++i) {
        const size_t label = flows[i].second;

        int assignment = static_cast<int>(assignments[i]);
        if (assignment == static_cast<int>(SIZE_MAX)){
            assignment = -1;
        }
        bool is_outlier = outlier_ids.count(i) > 0;
        fout << reduced(0, i) << "," 
             << reduced(1, i) << "," 
             << assignment << "," 
             << label << "," 
             << is_outlier << "\n";
    }
    fout.close();
}

void DBscanDetector::aggreagte(void) {
    auto all_flows = loader_->getAllData();
    auto edge_constructor = make_shared<EdgeConstructor>(all_flows);

    edge_constructor->ClassifyFlow();
    edge_constructor->AggregateFlow();
}


void DBscanDetector::run(void) {
    const auto& all_flows = *loader_->getAllData();
    size_t total = all_flows.size();
    size_t count = 0;
    size_t print_interval = 1000;

    // aggreagte();

    for (const auto &[flow, label] : all_flows) {
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

    printSamples();

    cout << "🔄 Start train: " << "\n";

    detect(all_flows);
}