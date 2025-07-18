#include "data_loader.hpp"
#include "../parser/packet_parser.hpp"
#include "../flow_construct/explicit_constructor.hpp"

void PcapLoader::ParsePacket() {
    unique_ptr<PacketParser> packet_parser =
        label_path_.empty()
            ? make_unique<PacketParser>(data_path_)
            : make_unique<PacketParser>(data_path_, label_path_);

    packet_parser->ParseAll(data_size_);
    parse_result_ = packet_parser->getParseResult();
}

void PcapLoader::Load() 
{
    ParsePacket();
    auto flow_constructor = make_shared<explicit_flow_constructor>(parse_result_);
    flow_constructor->construct_flow();
    flow4_ = flow_constructor->get_constructed_flow().first;

    vector<pair<FlowRecord, size_t>> all;
    all.reserve(flow4_->size());

    for (const auto& flow : *flow4_) {
        const auto& [src_ip, dst_ip, src_port, dst_port, stack_code] = flow->flow_id;

        FlowRecord fr;
        fr.src_ip   = src_ip;
        fr.dst_ip   = dst_ip;
        fr.src_port = src_port;
        fr.dst_port = dst_port;
        fr.proto = stack_code_to_type(stack_code);

        fr.ts_start = get_time_spec(flow->get_str_time());
        fr.ts_end   = get_time_spec(flow->get_end_time());
        fr.packets  = flow->get_p_packet_p_seq()->size();
        fr.bytes    = flow->get_flow_length();

        size_t label = 0;
        all.emplace_back(fr, label);
    }

    size_t train_size = static_cast<size_t>(train_ratio_ * all.size());
    all_data_ptr_->assign(all.begin(), all.end());
    train_data_ptr_->assign(all.begin(), all.begin() + train_size);
    test_data_ptr_->assign(all.begin() + train_size, all.end());

    size_t benign_count = 0, attack_count = 0;
    for (const auto& [_, label] : all) {
        (label == 0) ? ++benign_count : ++attack_count;
    }

    LOGF("📦 Loaded total: %zu | Train: %zu | Test: %zu | BENIGN: %zu | ATTACK: %zu", 
        all.size(), train_data_ptr_->size(), test_data_ptr_->size(), benign_count, attack_count
    );
}