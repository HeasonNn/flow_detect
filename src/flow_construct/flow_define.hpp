#pragma once

#include "../common.hpp"
#include "../parser/packet.hpp"

using flow_time_t = double_t;
using binary_label_t = std::vector<bool>;

class basic_flow {
protected:
    flow_time_t str = std::numeric_limits<flow_time_t>::max();
    flow_time_t end = std::numeric_limits<flow_time_t>::min();
    pkt_code_t code = 0;
    std::shared_ptr<std::vector<std::shared_ptr<basic_packet>>> p_packet_p_seq = std::make_shared<std::vector<std::shared_ptr<basic_packet>>>();
    std::shared_ptr<std::vector<size_t>> p_reverse_index = std::make_shared<std::vector<size_t>>();

public:
    basic_flow() = default;

    basic_flow(flow_time_t str, flow_time_t end, pkt_code_t code, 
               std::shared_ptr<std::vector<std::shared_ptr<basic_packet>>> p_packet_p_seq, 
               std::shared_ptr<std::vector<size_t>> p_reverse_index)
        : str(str), end(end), code(code), p_packet_p_seq(p_packet_p_seq), p_reverse_index(p_reverse_index) {}

    basic_flow(std::shared_ptr<std::vector<std::shared_ptr<basic_packet>>> p_packet_p_seq, 
               std::shared_ptr<std::vector<size_t>> p_reverse_index)
        : p_packet_p_seq(p_packet_p_seq), p_reverse_index(p_reverse_index) {
        // Combine initialization of str, end, and code in a loop
        for (const auto& p : *p_packet_p_seq) {
            auto ts = GET_DOUBLE_TS(p->ts);
            str = std::min(str, ts);
            end = std::max(end, ts);
            code |= p->tp;
        }
    }

    virtual ~basic_flow() = default;

    basic_flow(const basic_flow&) = default;
    basic_flow& operator=(const basic_flow&) = default;

    bool emplace_packet(const std::shared_ptr<basic_packet> p, size_t rid) {
        if (typeid(*p) == typeid(basic_packet_bad)) {
            return false;
        }
        
        auto ts = GET_DOUBLE_TS(p->ts);
        str = std::min(str, ts);
        end = std::max(end, ts);
        code |= p->tp;
        p_packet_p_seq->push_back(p);
        p_reverse_index->push_back(rid);
        return true;
    }

    bool get_flow_label(const std::shared_ptr<binary_label_t>& p_label) {
        return std::any_of(p_reverse_index->begin(), p_reverse_index->end(),
                    [&](size_t idx) { return (*p_label)[idx]; });
    }

    size_t get_flow_length() const {
        size_t total_bytes = 0;

        for (const auto& pkt_ptr : *p_packet_p_seq) {
            total_bytes += pkt_ptr->len;
        }
        return total_bytes;
    }

    inline flow_time_t get_str_time() const { return str; }
    inline flow_time_t get_end_time() const { return end; }
    inline flow_time_t get_fct() const { return end - str; }
    inline pkt_code_t get_pkt_code() const { return code; }
    inline std::shared_ptr<std::vector<size_t>> get_p_reverse_id() const { return p_reverse_index; }
    inline std::shared_ptr<std::vector<std::shared_ptr<basic_packet>>> get_p_packet_p_seq() const { return p_packet_p_seq; }
};

template <typename FlowIdType>
class tuple5_flow : public basic_flow {
public:
    FlowIdType flow_id;

    tuple5_flow(const FlowIdType& flow_id)
        : flow_id(flow_id) {}

    tuple5_flow(const FlowIdType& flow_id, flow_time_t str, flow_time_t end, pkt_code_t code,
                std::shared_ptr<std::vector<std::shared_ptr<basic_packet>>> p_packet_p_seq, 
                std::shared_ptr<std::vector<size_t>> p_reverse_index)
        : flow_id(flow_id), basic_flow(str, end, code, p_packet_p_seq, p_reverse_index) {}

    tuple5_flow(const FlowIdType& flow_id, 
                std::shared_ptr<std::vector<std::shared_ptr<basic_packet>>> p_packet_p_seq, 
                std::shared_ptr<std::vector<size_t>> p_reverse_index)
        : flow_id(flow_id), basic_flow(p_packet_p_seq, p_reverse_index) {}

    tuple5_flow(const tuple5_flow&) = default;
    tuple5_flow& operator=(const tuple5_flow&) = default;
    virtual ~tuple5_flow() = default;
};


using tuple5_flow4 = tuple5_flow<tuple5_conn4>;
using tuple5_flow6 = tuple5_flow<tuple5_conn6>;