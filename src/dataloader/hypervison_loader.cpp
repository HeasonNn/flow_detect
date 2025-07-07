#include "data_loader.hpp"
#include "../flow_construct/explicit_constructor.hpp"

bool HyperVisonLoader::import_dataset() 
{
    ifstream data_file(data_path_);
    if (!data_file) {
        cerr << "❌ Failed to open data file: " << data_path_ << "\n";
        return false;
    }

    __START_FTIMMER__

    vector<string> packet_lines;
    string line;
    while (getline(data_file, line)) {
        if (!line.empty()) {
            packet_lines.emplace_back(std::move(line));
        }
    }

    if (packet_lines.empty()) {
        cerr << "❌ No packet lines loaded.\n";
        return false;
    }

    const size_t total_packets = packet_lines.size();
    parse_result_ = make_shared<decltype(parse_result_)::element_type>(total_packets);


    constexpr size_t kThreadCount = 24;
    const size_t chunk_size = (total_packets + kThreadCount - 1) / kThreadCount;

    auto parse_range = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const auto& packet = packet_lines[i];
            if (packet.empty()) {
                parse_result_->at(i) = make_shared<basic_packet_bad>();
            } else {
                switch (packet[0]) {
                    case '4': parse_result_->at(i) = make_shared<basic_packet4>(packet); break;
                    case '6': parse_result_->at(i) = make_shared<basic_packet6>(packet); break;
                    default:  parse_result_->at(i) = make_shared<basic_packet_bad>(); break;
                }
            }
        }
    };

    vector<thread> threads;
    for (size_t i = 0; i < kThreadCount; ++i) {
        size_t begin = i * chunk_size;
        size_t end = min(begin + chunk_size, total_packets);
        if (begin < end)
            threads.emplace_back(parse_range, begin, end);
    }

    for (auto& t : threads)
        if (t.joinable()) t.join();

    ifstream label_file(label_path_);
    if (!label_file) {
        cerr << "❌ Failed to open label file: " << label_path_ << "\n";
        return false;
    }

    label_ = make_shared<vector<bool>>();
    if (getline(label_file, line)) {
        label_->reserve(line.size());
        for (char c : line) {
            if (c == '0' || c == '1')
                label_->emplace_back(c == '1');
        }
    }

    if (label_->size() != parse_result_->size()) {
        cerr << "❌ Label count (" << label_->size()
             << ") does not match packet count (" << parse_result_->size() << ")\n";
        return false;
    }

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    return true;
}


void HyperVisonLoader::load() 
{
    if (!import_dataset()) {
        cerr << "❌ Failed to load dataset: " << data_path_ << "\n";
        return;
    }

    auto flow_constructor = make_shared<explicit_flow_constructor>(parse_result_);
    flow_constructor->construct_flow();
    flow4_ = flow_constructor->get_constructed_flow().first;

    vector<pair<FlowRecord, size_t>> all;
    all.reserve(flow4_->size());

    for (const auto& flow : *flow4_) {
        const auto& [src_ip, dst_ip, src_port, dst_port, stack_code] = flow->flow_id;

        FlowRecord fr;
        fr.src_ip   = get_str_addr(src_ip);
        fr.dst_ip   = get_str_addr(dst_ip);
        fr.src_port = src_port;
        fr.dst_port = dst_port;
        fr.proto = stack_code_to_type(stack_code);

        fr.ts_start = get_time_spec(flow->get_str_time());
        fr.ts_start = get_time_spec(flow->get_end_time());
        fr.packets  = flow->get_p_packet_p_seq()->size();
        fr.bytes    = flow->get_flow_length();

        size_t label = flow->get_flow_label(label_);
        all.emplace_back(fr, label);
    }

    size_t train_size = static_cast<size_t>(train_ratio_ * all.size());
    train_data_ptr_->assign(all.begin(), all.begin() + train_size);
    test_data_ptr_->assign(all.begin() + train_size, all.end());

    size_t benign_count = 0, attack_count = 0;
    for (const auto& [_, label] : all) {
        (label == 0) ? ++benign_count : ++attack_count;
    }

    LOGF("📦 Loaded total: %zu | Train: %zu | Test: %zu | BENIGN: %zu | ATTACK: %zu", 
        all.size(), 
        train_data_ptr_->size(), 
        test_data_ptr_->size(), 
        benign_count, 
        attack_count
    );
}