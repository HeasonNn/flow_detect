//detector.hpp
#pragma once

#include <armadillo>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>

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

    explicit Detector(
        shared_ptr<FlowFeatureExtractor> flowExtractor, 
        shared_ptr<GraphFeatureExtractor> graphExtractor,
        shared_ptr<DataLoader> loader
    ):  flowExtractor_(flowExtractor), graphExtractor_(graphExtractor), loader_(loader) 
    { 
        loader_->load(); 
    };

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
    explicit RFDetector(
        shared_ptr<FlowFeatureExtractor> flowExtractor, 
        shared_ptr<GraphFeatureExtractor> graphExtractor,
        shared_ptr<DataLoader> loader
    ):  Detector(flowExtractor, graphExtractor, loader) {};

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

    void addSample(const arma::vec &x);
    void train(void);
    int predict(const arma::vec &x);

    void run_detection(void);
    void printFeatures(void) const noexcept;

public:
    explicit DBscanDetector(
        std::shared_ptr<FlowFeatureExtractor> flowExtractor, 
        std::shared_ptr<GraphFeatureExtractor> graphExtractor,
        std::shared_ptr<DataLoader> loader,
        double epsilon = 1.0, size_t minPoints = 5):  
        Detector(flowExtractor, graphExtractor, loader), epsilon_(epsilon), minPoints_(minPoints) {};

    void run(void) override;
};


class MiniBatchKMeansDetector : public Detector {
private:
    MiniBatchKMeans mbk_;
    bool trained_ = false;
    vector<arma::vec> sample_vecs_;
    size_t outlier_cluster_;

    void addSample(const arma::vec &x);
    void train(void);
    size_t predict(const arma::vec &x);

public:
    explicit MiniBatchKMeansDetector(
        std::shared_ptr<FlowFeatureExtractor> flowExtractor, 
        std::shared_ptr<GraphFeatureExtractor> graphExtractor,
        std::shared_ptr<DataLoader> loader):  
        Detector(flowExtractor, graphExtractor, loader),
        mbk_(MiniBatchKMeans(10, 64, 200, 0.8, true)) {};

    void run_detection(void);
    void run(void) override;
};


shared_ptr<Detector> createDetector(
    const std::string& algorithm, 
    shared_ptr<FlowFeatureExtractor> flowExtractor, 
    shared_ptr<GraphFeatureExtractor> graphExtractor,
    shared_ptr<DataLoader> loader
);

string get_current_time_str(void);