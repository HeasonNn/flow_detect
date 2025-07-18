#pragma once

#include "../feature/flow_feature.hpp"
#include "../flow_construct/flow_define.hpp"
#include "../common.hpp"

using namespace std;

class DataLoader
{
protected:
    shared_ptr<vector<pair<FlowRecord, size_t>>> all_data_ptr_;
    shared_ptr<vector<pair<FlowRecord, size_t>>> train_data_ptr_;
    shared_ptr<vector<pair<FlowRecord, size_t>>> test_data_ptr_;
    string data_path_, label_path_;
    double train_ratio_;
    int data_size_;

public:
    explicit DataLoader(const string &data_path, const string &label_path = "", double train_ratio = 0.8, int data_size = -1)
        : all_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          train_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          test_data_ptr_(make_shared<vector<pair<FlowRecord, size_t>>>()),
          data_size_(data_size), data_path_(data_path), label_path_(label_path), train_ratio_(train_ratio) {}

    virtual ~DataLoader() = default;
    virtual void Load() = 0;
    
    const auto& getAllData()   const { return all_data_ptr_; }
    const auto& getTrainData() const { return train_data_ptr_; }
    const auto& getTestData()  const { return test_data_ptr_; }
    const auto& getDataPath()  const { return data_path_; }

    const auto getDataFileBaseName() const -> decltype(data_path_);

    void setDataPath(const string &path) { data_path_ = path; }
};

class CICIDSLoader : public DataLoader
{
public:
    using DataLoader::DataLoader;

    double parse_timestamp(const string& ts_str);
    void Load() override;
};

using binary_label_t = vector<bool>;

class HyperVisonLoader : public DataLoader
{
private:
    shared_ptr<vector<shared_ptr<basic_packet>>> parse_result_;
    shared_ptr<vector<shared_ptr<tuple5_flow4>>> flow4_;
    shared_ptr<vector<shared_ptr<tuple5_flow6>>> flow6_;
    shared_ptr<binary_label_t> label_;

public:
    using DataLoader::DataLoader;
    bool import_dataset();
    void Load() override;
};


class PcapLoader : public DataLoader
{
private:
    shared_ptr<vector<shared_ptr<basic_packet>>> parse_result_;
    shared_ptr<vector<shared_ptr<tuple5_flow4>>> flow4_;
    shared_ptr<vector<shared_ptr<tuple5_flow6>>> flow6_;
    shared_ptr<binary_label_t> label_;

public:
    using DataLoader::DataLoader;
    void ParsePacket();
    void Load() override;
};


shared_ptr<DataLoader> createDataLoader(const json& config_j);