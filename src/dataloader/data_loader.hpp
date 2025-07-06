#pragma once

#include "../feature/flow_feature.hpp"
#include "../flow_construct/flow_define.hpp"
#include "../common.hpp"

using namespace std;

class DataLoader
{
protected:
    shared_ptr<vector<pair<FlowRecord, size_t>>> train_data_ptr_;
    shared_ptr<vector<pair<FlowRecord, size_t>>> test_data_ptr_;
    string data_path_, label_path_;
    double train_ratio_;

public:
    explicit DataLoader()
        : train_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          test_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          train_ratio_(0.8) {}

    explicit DataLoader(const string &data_path)
        : train_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          test_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          data_path_(data_path),
          train_ratio_(0.8) {}

    explicit DataLoader(const string &data_path, const string &label_path, double train_ratio = 0.8)
        : train_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          test_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          data_path_(data_path),
          label_path_(label_path),
          train_ratio_(train_ratio) {}

    virtual ~DataLoader() = default;

    virtual void load() = 0;

    shared_ptr<const vector<pair<FlowRecord, size_t>>> getTrainData() const { return train_data_ptr_; }
    shared_ptr<const vector<pair<FlowRecord, size_t>>> getTestData() const { return test_data_ptr_; }

    const string &getDataPath() const { return data_path_; }
    void setDataPath(const string &path) { data_path_ = path; }
};