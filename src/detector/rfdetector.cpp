// rfdetecor.cpp
#include "detector.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <ctime>

namespace fs = std::filesystem;

void RfDetector::addSample(const arma::vec& x, size_t label) {
    sample_vecs_.push_back(x);
    labels_.push_back(label);
}


void RfDetector::train() {
    if (trained_ || sample_vecs_.empty()) return;

    const size_t dim = sample_vecs_[0].n_elem;
    arma::mat X(dim, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i)
        X.col(i) = sample_vecs_[i];

    arma::Row<size_t> Y(labels_);
    rf_.Train(X, Y, 2);
    trained_ = true;
}


size_t RfDetector::predict(const arma::vec& x) {
    arma::Row<size_t> pred;
    rf_.Classify(x, pred);
    return pred(0);
}


void RfDetector::printFeatures(void) const
{
    if (sample_vecs_.empty()) {
        std::cout << "No features to print." << std::endl;
        return;
    }

    std::cout << "Feature Vectors:" << std::endl;
    std::cout << std::setw(8) << "Sample";

    for (size_t i = 0; i < sample_vecs_[0].n_elem; ++i) {
        std::cout << std::setw(12) << "Field " << i + 1;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < sample_vecs_.size(); ++i)
    {
        std::cout << std::setw(8) << "Sample " << i + 1;
        for (size_t j = 0; j < sample_vecs_[i].n_elem; ++j) {
            std::cout << std::setw(12) << sample_vecs_[i][j];
        }
        std::cout << std::endl;
    }
}

std::string get_current_time_str(void) {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", std::localtime(&now_time_t));
    
    return std::string(buf);
}


void RfDetector::run_detection(void)
{
    const auto data_path = loader_->getDataPath();
    const auto test_flows = *loader_->getTestData();

    size_t TP = 0, TN = 0, FP = 0, FN = 0;

    std::string current_time = get_current_time_str();
    // === 输出路径处理 ===
    fs::create_directory("result");
    std::string base = fs::path(data_path).stem().string();  // 不含扩展名
    std::string output_file = "result/" + base + "_pred_" + current_time + ".csv";
    std::string metric_file = "result/" + base + "_metrics_" + current_time + ".txt";
    std::ofstream ofs(output_file);
    ofs << "SrcIP,DstIP,Pred,Label\n";

    std::cout << "\n🔎 Predicting:\n";

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

    std::cout << "\n📊 Evaluation Metrics:\n";
    std::cout << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    std::cout << "🎯 Precision : " << precision * 100 << "%\n";
    std::cout << "📥 Recall    : " << recall * 100 << "%\n";
    std::cout << "📈 F1-Score  : " << f1 * 100 << "%\n";
    std::cout << "📈 F2-Score  : " << f2 * 100 << "%\n";

    std::cout << "\n📊 Confusion Matrix:\n";
    std::cout << "             Predicted\n";
    std::cout << "            0        1\n";
    std::cout << "Actual 0 | " << std::setw(6) << TN << "  | " << std::setw(6) << FP << "\n";
    std::cout << "Actual 1 | " << std::setw(6) << FN << "  | " << std::setw(6) << TP << "\n";

    std::cout << "\n📁 Results written to: " << output_file << "\n";

    // 📄 保存评估指标到 TXT
    std::ofstream mfs(metric_file);
    mfs << std::fixed << std::setprecision(4);

    mfs << "📊 Evaluation Metrics:\n";
    mfs << "✅ Accuracy  : " << accuracy * 100 << "%\n";
    mfs << "🎯 Precision : " << precision * 100 << "%\n";
    mfs << "📥 Recall    : " << recall * 100 << "%\n";
    mfs << "📈 F1-Score  : " << f1 * 100 << "%\n";
    mfs << "📈 F2-Score  : " << f2 * 100 << "%\n\n";

    mfs << "📊 Confusion Matrix:\n";
    mfs << "             Predicted\n";
    mfs << "            0        1\n";
    mfs << "Actual 0 | " << std::setw(6) << TN << "  | " << std::setw(6) << FP << "\n";
    mfs << "Actual 1 | " << std::setw(6) << FN << "  | " << std::setw(6) << TP << "\n";

    mfs.close();

    std::cout << "📄 Metrics written to: " << metric_file << "\n";
}


void RfDetector::run(void) 
{
    const auto train_flows = *loader_->getTrainData();

    // ✅ 训练阶段
    size_t total = train_flows.size();
    size_t count = 0;
    size_t print_interval = 1000; // 每1000个样本打印一次

    for (const auto &[flow, label] : train_flows)
    {
        graphExtractor_->updateGraph(flow.src_ip, flow.dst_ip);
        arma::vec flowVec = flowExtractor_->extract(flow);
        arma::vec graphVec = graphExtractor_->extract(flow.src_ip, flow.dst_ip);

        if (flowVec.is_empty() || graphVec.is_empty())
            continue;

        addSample(arma::join_vert(flowVec, graphVec), label);

        if (++count % print_interval == 0 || count == total)
        {
            std::cout << "\rProcessed " << count << " / " << total << " samples." << std::flush;
        }
    }
    std::cout << std::endl; 

    cout << " 🔄 Start train: " << "\n";
    
    train();

    run_detection();
}