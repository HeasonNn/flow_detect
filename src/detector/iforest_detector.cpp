#include "detector.hpp"

namespace fs = filesystem;


void IForestDetector::Train(){
    if (sample_vecs_.empty()) {
        std::cerr << "⚠️  No valid samples collected for training!\n";
        return;
    }

    arma::mat data(sample_vecs_[0].n_elem, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i)
        data.col(i) = sample_vecs_[i];

    scaler_.Fit(data);
    arma::mat norm_data;
    scaler_.Transform(data, norm_data);

    iforest_->Fit(norm_data);
}


double IForestDetector::Detect(arma::vec& sample_vec) {
    arma::vec norm;
    scaler_.Transform(sample_vec, norm);
    double score = iforest_->AnomalyScore(norm);
    return score;
}


void IForestDetector::run(){
    const auto& train_flows = *loader_->getTrainData();
    size_t total = train_flows.size();
    size_t count = 0;
    size_t print_interval = 1000;

    for (const auto &[flow, label] : train_flows) {
        if(label) continue;

        graphExtractor_->advance_time(GET_DOUBLE_TS(flow.ts_start));
        graphExtractor_->updateGraph(flow);
        arma::vec graphVec = graphExtractor_->extract(flow);

        if (graphVec.is_empty()) continue;
        addSample(graphVec);

        if (++count % print_interval == 0 || count == total) {
            cout << "\rProcessed " << count << " / " << total << " samples." << flush;
        }
    }
    cout << endl; 

    printSamples();

    cout << "🔄 Start train: " << "\n";
    Train();


    cout << "🔄 Start detect: " << "\n";
    const auto& test_flows = *loader_->getTestData();

    fs::create_directory("result");
    string base = fs::path(loader_->getDataPath()).stem().string();
    string timestamp = get_current_time_str();
    // string output_file = "result/" + base + "_predict_iforest_" + timestamp + ".csv";
    string output_file = "result/" + base + "_predict_iforest.csv";

    ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Score,Pred,Label\n";

    arma::mat all_test_vecs;
    vector<size_t> all_labels;
    vector<size_t> all_preds;

    size_t TP = 0, FP = 0, FN = 0, TN = 0;
    size_t count_detect = 0;
    size_t total_detect = test_flows.size();
    for (const auto& [flow, label] : test_flows) {
        graphExtractor_->advance_time(GET_DOUBLE_TS(flow.ts_start));
        graphExtractor_->updateGraph(flow);

        arma::vec sample = graphExtractor_->extract(flow);
        if (sample.is_empty()) continue;
        arma::vec norm_data;
        scaler_.Transform(sample, norm_data);

        double score = iforest_->AnomalyScore(norm_data);
        bool is_anomaly = score > outline_threshold_;
        size_t pred = is_anomaly ? 1 : 0;

        ofs << flow.src_ip << "," << flow.dst_ip << ","
            << score << "," << pred << "," << label << "\n";

        all_test_vecs.insert_cols(all_test_vecs.n_cols, norm_data);
        all_labels.push_back(label);
        all_preds.push_back(is_anomaly);

        if (label == 1 && pred == 1) TP++;
        else if (label == 0 && pred == 0) TN++;
        else if (label == 0 && pred == 1) FP++;
        else if (label == 1 && pred == 0) FN++;

        if (++count_detect % print_interval == 0 || count_detect == total_detect) {
            cout << "\rDetected " << count_detect << " / " << total_detect << " samples." << flush;
        }
    }
    cout << endl;
    ofs.close();

    size_t sum       = TP + TN + FP + FN;
    double accuracy  = (double)(TP + TN) / sum;
    double precision = (TP + FP) ? (double)TP / (TP + FP) : 0.0;
    double recall    = (TP + FN) ? (double)TP / (TP + FN) : 0.0;
    double fpr       = (FP + TN) ? (double)FP / (FP + TN) : 0.0;
    double f1        = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0.0;

    cout << "📊 Final Evaluation (Isolation Forest):\n";
    cout << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    cout << "🎯 Precision : " << precision * 100 << "%\n";
    cout << "📥 Recall    : " << recall * 100 << "%\n";
    cout << "🚨 FPR       : " << fpr * 100 << "%\n";
    cout << "📈 F1-Score  : " << f1 * 100 << "%\n";
    cout << "📁 Results written to: " << output_file << "\n";

    if (all_test_vecs.n_cols > 0) {
        mlpack::PCA pca;
        arma::mat reduced;
        pca.Apply(all_test_vecs, reduced, 2);

        std::ofstream fout("result/iforest_pca_result.csv");
        fout << "x,y,pred,label\n";
        for (size_t i = 0; i < reduced.n_cols; ++i) {
            fout << reduced(0, i) << "," 
                 << reduced(1, i) << ","
                 << all_preds[i]  << "," 
                 << all_labels[i] << "\n";
        }
        fout.close();
    }
}