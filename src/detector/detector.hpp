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
#include "../graph_construct/edge_constructor.hpp"
#include "../algorithm/isolation_forest.hpp"

using namespace std;


class Detector
{
protected:
    shared_ptr<FlowFeatureExtractor> flowExtractor_;
    shared_ptr<GraphFeatureExtractor> graphExtractor_;
    shared_ptr<DataLoader> loader_;

    vector<arma::vec> sample_vecs_;

    virtual void addSample(const arma::vec &x) {sample_vecs_.push_back(x);}
    virtual void pcaAnalyze();
    
    virtual void printSamples() const;


public:

    explicit Detector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                      shared_ptr<GraphFeatureExtractor> graphExtractor,
                      shared_ptr<DataLoader> loader)
        : flowExtractor_(move(flowExtractor)), 
          graphExtractor_(move(graphExtractor)), 
          loader_(move(loader)) 
        {
            loader_->Load();
        }

    virtual ~Detector() = default;

    virtual void run() = 0;

};


class RFDetector : public Detector {
private:
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


class MiniBatchKMeansDetector : public Detector {
private:
    MiniBatchKMeansConfig model_config_;
    MiniBatchKMeans mbk_;
    
    size_t global_threshold_ = 3;
    bool trained_ = false;

    vector<size_t> cluster_counts_;
    unordered_map<size_t, vector<double>> cluster_dists_;
    unordered_map<size_t, double> cluster_mean_;
    unordered_map<size_t, double> cluster_stddev_;
    unordered_map<size_t, double> cluster_thresholds_;

    vector<arma::vec> normalized_sample_vecs_;

    void Train(void);
    size_t Predict(const arma::vec &x);

    void PerformPCAVisualization();
    void PrintClusteDetail();

    void SaveTrainClusterResult(const string& filename);
    void SaveTestAbnormalClusterResult(const string& filename);

public:
    explicit MiniBatchKMeansDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                     shared_ptr<GraphFeatureExtractor> graphExtractor,
                                     shared_ptr<DataLoader> loader)
        : Detector(flowExtractor, graphExtractor, loader), 
          model_config_(), 
          mbk_(model_config_){};

    void run_detection(void);
    void run(void) override;
};


struct DBSCANModel {
    arma::mat norm_data;
    arma::Row<size_t> cluster_labels;
};

class DBscanDetector : public Detector {
private:
    const json& config_;    

    double epsilon_;
    size_t min_points_;
    double outline_threshold_;

    vector<DBSCANModel> models_;
    mlpack::data::MinMaxScaler scaler_;

    bool trained_ = false;

    void aggregate(void);
    void detect(const vector<pair<FlowRecord, size_t>>& flows);

public:
    explicit DBscanDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                            shared_ptr<GraphFeatureExtractor> graphExtractor,
                            shared_ptr<DataLoader> loader,
                            const json& config)
        : Detector(flowExtractor, graphExtractor, loader), 
          config_(config)
    {
        const json& dbscan_config_ = config_["dbscan_config"];
        epsilon_ = dbscan_config_.value("epsilon", 0.1);
        min_points_ = dbscan_config_.value("min_points", 10);
        outline_threshold_ = dbscan_config_.value("outline_threshold", 0.65);

        cout << "epsilon_: "           << epsilon_ << ", "
             << "minPoints_: "         << min_points_ << ", "
             << "outline_threshold_: " << outline_threshold_ << "\n";
    };

    void run(void) override;
};


class IForestDetector : public Detector{
private:
    const json& config_;
    
    size_t n_trees_;
    size_t sample_size_;
    size_t max_depth_;

    mlpack::data::MinMaxScaler scaler_;
    unique_ptr<IsolationForest> iforest_;
    double outline_threshold_;
public:
    explicit IForestDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                             shared_ptr<GraphFeatureExtractor> graphExtractor,
                             shared_ptr<DataLoader> loader,
                             const json& config)
        : Detector(flowExtractor, graphExtractor, loader),
          config_(config)
        {
            const json& iforest_config_ = config_["iforest_config"];

            n_trees_           = iforest_config_.value("n_trees", 100);
            sample_size_       = iforest_config_.value("sample_size", 64);
            max_depth_         = iforest_config_.value("max_depth", 10);
            outline_threshold_ = iforest_config_.value("outline_threshold", 0.1);
            
            iforest_ = make_unique<IsolationForest>(n_trees_, sample_size_, max_depth_);

            cout << "n_trees_: "           << n_trees_     << ", "
                 << "minPoints_: "         << sample_size_ << ", "
                 << "outline_threshold_: " << max_depth_   << ", "
                 << "minPoints_: "         << sample_size_ << "\n";
        };

    void Train();
    double Detect(arma::vec& sample_vec);

    void run(void) override;
};


shared_ptr<Detector> createDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader,
                                    const json& config_j);

string get_current_time_str(void);