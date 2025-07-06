#pragma once

#include "data_loader.hpp"

class CICIDSLoader : public DataLoader
{
public:
    using DataLoader::DataLoader;

    double parse_timestamp(const string& ts_str);
    void load() override;
};