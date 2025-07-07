// rfdetecor.cpp
#include "detector.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <ctime>

namespace fs = filesystem;

void RFDetector::addSample(const arma::vec& x, size_t label) {
    sample_vecs_.push_back(x);
    labels_.push_back(label);
}


void RFDetector::train() {
    if (trained_ || sample_vecs_.empty()) return;

    const size_t dim = sample_vecs_[0].n_elem;
    arma::mat X(dim, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i)
        X.col(i) = sample_vecs_[i];

    arma::Row<size_t> Y(labels_);
    rf_.Train(X, Y, 2);
    trained_ = true;
}


size_t RFDetector::predict(const arma::vec& x) {
    arma::Row<size_t> pred;
    rf_.Classify(x, pred);
    return pred(0);
}


void RFDetector::printFeatures(void) const noexcept
{
    if (sample_vecs_.empty()) {
        cout << "No features to print." << endl;
        return;
    }

    cout << "Feature Vectors:" << endl;
    cout << setw(8) << "Sample";

    for (size_t i = 0; i < sample_vecs_[0].n_elem; ++i) {
        cout << setw(12) << "Field " << i + 1;
    }
    cout << endl;

    for (size_t i = 0; i < sample_vecs_.size(); ++i)
    {
        cout << setw(8) << "Sample " << i + 1;
        for (size_t j = 0; j < sample_vecs_[i].n_elem; ++j) {
            cout << setw(12) << sample_vecs_[i][j];
        }
        cout << endl;
    }
}


void RFDetector::run_detection(void)
{
    const auto data_path = loader_->getDataPath();
    const auto test_flows = *loader_->getTestData();
    size_t TP = 0, TN = 0, FP = 0, FN = 0;

    // === 输出路径处理 ===
    fs::create_directory("result");
    string current_time = get_current_time_str();
    string base = fs::path(data_path).stem().string();  // 不含扩展名
    string output_file = "result/" + base + "_pred_" + current_time + ".csv";
    string metric_file = "result/" + base + "_metrics_" + current_time + ".txt";
    ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Pred,Label\n";

    cout << "\n🔎 Predicting:\n";

    for (const auto& pair : test_flows) {
        const FlowRecord& flow = pair.first;
        arma::vec flowVec = flowExtractor_->extract(flow);
        arma::vec graphVec = graphExtractor_->extract(flow.src_ip, flow.dst_ip);

        if (flowVec.n_elem == 0 || graphVec.n_elem == 0) continue;
        arma::vec feat = arma::join_vert(flowVec, graphVec);
        size_t pred = predict(feat);
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

    // 📄 保存评估指标到 TXT
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


void RFDetector::run(void) 
{
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
        addSample(arma::join_vert(flowVec, graphVec), label);

        if (++count % print_interval == 0 || count == total) {
            cout << "\rProcessed " << count << " / " << total << " samples." << flush;
        }
    }
    cout << endl; 

    cout << " 🔄 Start train: " << "\n";
    
    train();

    run_detection();
}