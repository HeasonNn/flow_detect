//detector.hpp
#pragma once

#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/methods/pca/pca.hpp>

#include "../algorithm/mini_batch_kmeans.hpp"
#include "../feature/flow_feature.hpp"
#include "../feature/graph_features.hpp"
#include "../dataloader/data_loader.hpp"

using namespace std;


class Detector
{
protected:
    shared_ptr<FlowFeatureExtractor> flowExtractor_;
    shared_ptr<GraphFeatureExtractor> graphExtractor_;
    shared_ptr<DataLoader> loader_; 

public:

    explicit Detector(std::shared_ptr<FlowFeatureExtractor> flowExtractor, 
                      std::shared_ptr<GraphFeatureExtractor> graphExtractor,
                      std::shared_ptr<DataLoader> loader)
        : flowExtractor_(std::move(flowExtractor)), 
          graphExtractor_(std::move(graphExtractor)), 
          loader_(std::move(loader))
    {
        if (loader_) loader_->load();
    }

    virtual void run() = 0;
    virtual ~Detector() = default;
};


class RFDetector : public Detector {
private:
    vector<arma::vec> sample_vecs_;
    vector<size_t> labels_;
    mlpack::RandomForest<> rf_;

    bool trained_ = false;

    void addSample(const arma::vec &x, size_t label) ;
    void train(void);
    size_t predict(const arma::vec &x);

    void run_detection(void);
    void printFeatures(void) const noexcept;

public:
    explicit RFDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                        shared_ptr<GraphFeatureExtractor> graphExtractor,
                        shared_ptr<DataLoader> loader)
        : Detector(flowExtractor, graphExtractor, loader) {};

    void run(void) override;
};

struct DBSCANModel {
    arma::mat norm_data;
    arma::Row<size_t> cluster_labels;
};

class DBscanDetector : public Detector {
private:
    
    double epsilon_;
    size_t minPoints_;

    std::vector<DBSCANModel> models_;
    mlpack::data::MinMaxScaler scaler_;
    vector<arma::vec> sample_vecs_;

    bool trained_ = false;

    void AddSample(const arma::vec &x);
    void Detect(const vector<pair<FlowRecord, size_t>>& flows);

public:
    explicit DBscanDetector(std::shared_ptr<FlowFeatureExtractor> flowExtractor, 
                            std::shared_ptr<GraphFeatureExtractor> graphExtractor,
                            std::shared_ptr<DataLoader> loader,
                            double epsilon, size_t minPoints)
        : Detector(flowExtractor, graphExtractor, loader), epsilon_(epsilon), minPoints_(minPoints) {};

    void run(void) override;
};


class MiniBatchKMeansDetector : public Detector {
private:
    MiniBatchKMeansConfig model_config_;
    MiniBatchKMeans mbk_;
    vector<arma::vec> sample_vecs_;
    
    size_t global_threshold_ = 3;
    bool trained_ = false;

    std::vector<size_t> cluster_counts_;
    std::unordered_map<size_t, std::vector<double>> cluster_dists_;
    std::unordered_map<size_t, double> cluster_mean_;
    std::unordered_map<size_t, double> cluster_stddev_;
    std::unordered_map<size_t, double> cluster_thresholds_;

    std::vector<arma::vec> normalized_sample_vecs_;

    void AddSample(const arma::vec &x);
    void Train(void);
    size_t Predict(const arma::vec &x);

    void PerformPCAVisualization();

    void PrintClusteDetail();

    void SaveTrainClusterResult(const std::string& filename);
    void SaveTestAbnormalClusterResult(const std::string& filename);

public:
    explicit MiniBatchKMeansDetector(std::shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                     std::shared_ptr<GraphFeatureExtractor> graphExtractor,
                                     std::shared_ptr<DataLoader> loader)
        : Detector(flowExtractor, graphExtractor, loader), model_config_(), mbk_(model_config_){};

    void run_detection(void);
    void run(void) override;
};


shared_ptr<Detector> createDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader,
                                    const json& config_j);

string get_current_time_str(void);