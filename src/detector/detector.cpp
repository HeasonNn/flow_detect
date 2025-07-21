#include "detector.hpp"


string get_current_time_str(void) {
    auto now = chrono::system_clock::now();
    auto now_time_t = chrono::system_clock::to_time_t(now);
    
    char buf[100];
    strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", localtime(&now_time_t));
    
    return string(buf);
}


shared_ptr<Detector> createDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader,
                                    const json& config_j)
{
    const json& detector_config = config_j["detector"];
    const string& algorithm     = detector_config.value("algorithm", "Unknown");

    if (algorithm == "RF") {
        return make_shared<RFDetector>(flowExtractor, graphExtractor, loader);
    }
    else if (algorithm == "DBSCAN") {
        return make_shared<DBscanDetector>(flowExtractor, graphExtractor, loader, detector_config);
    }
    else if (algorithm == "Mini_Batch_KMeans") {
        return make_shared<MiniBatchKMeansDetector>(flowExtractor, graphExtractor, loader);
    }
    else if (algorithm == "IForest") {
        return make_shared<IForestDetector>(flowExtractor, graphExtractor, loader, detector_config);
    }
    else {
        throw std::invalid_argument("Unknown algorithm type: " + algorithm);
    }
    
    return nullptr;
}


void Detector::pcaAnalyze() {
    if (sample_vecs_.empty()) {
        std::cout << "[printSamples] No sample vectors available.\n";
        return;
    }

    arma::mat data(sample_vecs_[0].n_elem, sample_vecs_.size());
    for (size_t i = 0; i < sample_vecs_.size(); ++i)
        data.col(i) = sample_vecs_[i];
    
    mlpack::data::MinMaxScaler scaler;
    scaler.Fit(data);
    arma::mat norm_data;
    scaler.Transform(data, norm_data);

    mlpack::PCA pca;
    arma::mat reduced;
    pca.Apply(norm_data, reduced, 2);

    const auto& all_data       = *loader_->getAllData();
    const auto& data_file_name = loader_->getDataFileBaseName();

    std::string output_filename = "result/" + data_file_name + "_pca_analyze.csv";
    std::ofstream fout(output_filename);
    fout << "x,y,label\n";
    for (size_t i = 0; i < reduced.n_cols; ++i) {
        fout << reduced(0, i)      << "," 
             << reduced(1, i)      << ","
             << all_data[i].second << "\n";
    }
    fout.close();
}


void Detector::printSamples() const {
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
              << std::setw(8)  << "Feature"
              << std::setw(15) << "Mean"
              << std::setw(15) << "Max"
              << std::setw(15) << "Median"
              << std::setw(15) << "Mode" << "\n";

    for (size_t i = 0; i < dim; ++i) {
        arma::vec values = mat.row(i).t();

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
              << std::setw(15) << std::fixed << std::setprecision(4) << mode << "\n";
    }
}