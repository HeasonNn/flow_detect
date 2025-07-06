#include "detector.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <ctime>

namespace fs = std::filesystem;

std::string get_current_time_str() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", std::localtime(&now_time_t));
    
    return std::string(buf);
}

Detector::Detector(AlgorithmType algoType)
    : algorithmType(algoType)
{
    setAlgorithm(algoType);
}

void Detector::setAlgorithm(AlgorithmType newAlgoType)
{
    algorithmType = newAlgoType;
    switch (algorithmType)
    {
        case AlgorithmType::RandomForest:
            rf = std::make_unique<RandomForestAlgorithm>();
            break;
        case AlgorithmType::DBSCAN:
            // dbscan = std::make_unique<DBSCAN>();
            break;
        case AlgorithmType::IsolationForest:
            // iforest = std::make_unique<IsolationForest>();
            break;
    }
}


void Detector::run_detection(const std::string& data_path,
                              const std::vector<std::pair<FlowRecord, size_t>>& test_flows)
{
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
        arma::vec flowVec = flowExtractor.extract(flow);
        arma::vec graphVec = graphExtractor.extract(flow.src_ip, flow.dst_ip);

        if (flowVec.n_elem == 0 || graphVec.n_elem == 0) continue;
        arma::vec feat = arma::join_vert(flowVec, graphVec);
        
        size_t pred = 0;
        switch (algorithmType)
        {
            case AlgorithmType::RandomForest:
                pred = rf->predict(feat);
                break;
            case AlgorithmType::DBSCAN:
                // pred = dbscan->predict(feat);
                break;
            case AlgorithmType::IsolationForest:
                // pred = iforest->predict(feat);
                break;
        }
        size_t label = pair.second;

        if (pred == 1 && label == 1) TP++;
        else if (pred == 0 && label == 0) TN++;
        else if (pred == 1 && label == 0) FP++;
        else if (pred == 0 && label == 1) FN++;

        // std::cout << flow.src_ip << " -> " << flow.dst_ip
        //           << (pred == 1 ? " 🔴 Anomaly" : " ✅ Normal")
        //           << " (expected: " << (label == 1 ? "Anomaly" : "Normal") << ")\n";

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
