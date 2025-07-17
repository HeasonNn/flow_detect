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

using namespace std;


class Detector
{
protected:
    shared_ptr<FlowFeatureExtractor> flowExtractor_;
    shared_ptr<GraphFeatureExtractor> graphExtractor_;
    shared_ptr<DataLoader> loader_;
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
    void Start();
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
    const json& config_;    

    double epsilon_;
    size_t min_points_;
    double outline_threshold_;

    vector<DBSCANModel> models_;
    mlpack::data::MinMaxScaler scaler_;
    vector<arma::vec> sample_vecs_;

    bool trained_ = false;


    void aggreagte(void);
    void addSample(const arma::vec &x);
    void detect(const vector<pair<FlowRecord, size_t>>& flows);

    void printSamples() const;
    
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
        outline_threshold_ = dbscan_config_.value("outline_threshold", 0.1);

        cout << "epsilon_: "          << epsilon_ << " "
             << "minPoints_: "        << min_points_ << " "
             << "outline_threshold_: "<< outline_threshold_ << "\n";
    };

    void run(void) override;
};


class MiniBatchKMeansDetector : public Detector {
private:
    MiniBatchKMeansConfig model_config_;
    MiniBatchKMeans mbk_;
    vector<arma::vec> sample_vecs_;
    
    size_t global_threshold_ = 3;
    bool trained_ = false;

    vector<size_t> cluster_counts_;
    unordered_map<size_t, vector<double>> cluster_dists_;
    unordered_map<size_t, double> cluster_mean_;
    unordered_map<size_t, double> cluster_stddev_;
    unordered_map<size_t, double> cluster_thresholds_;

    vector<arma::vec> normalized_sample_vecs_;

    void AddSample(const arma::vec &x);
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


shared_ptr<Detector> createDetector(shared_ptr<FlowFeatureExtractor> flowExtractor, 
                                    shared_ptr<GraphFeatureExtractor> graphExtractor,
                                    shared_ptr<DataLoader> loader,
                                    const json& config_j);

string get_current_time_str(void);