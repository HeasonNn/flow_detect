#pragma once

#include <string>
#include <armadillo>
#include <memory>

struct FlowRecord {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t proto;

    timespec ts_start;
    timespec ts_end;
    size_t packets;
    size_t bytes;

    double get_duration() const
    {
        long long start_time_ms = ts_start.tv_sec * 1000 + ts_start.tv_nsec / 1000000;
        long long end_time_ms = ts_end.tv_sec * 1000 + ts_end.tv_nsec / 1000000;
        return static_cast<double>(end_time_ms - start_time_ms);
    }
};

class FlowFeatureExtractor {
public:
    arma::vec extract(const FlowRecord &flow);
};
