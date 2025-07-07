#include "flow_feature.hpp"

arma::vec FlowFeatureExtractor::extract(const FlowRecord& flow) {
    double avg_pkt = (flow.packets > 0) ? static_cast<double>(flow.bytes) / flow.packets : 0.0;
    
    arma::vec v(6);
    v(0) = flow.src_port;
    v(1) = flow.dst_port;
    v(2) = flow.proto;
    v(3) = flow.get_duration();
    v(4) = flow.packets;
    v(5) = avg_pkt;

    return v;
}
