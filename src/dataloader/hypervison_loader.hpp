#pragma once

#include "data_loader.hpp"
#include "../parser/packet.hpp"

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

    void load() override;
    bool import_dataset();
};