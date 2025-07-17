#include "data_loader.hpp"

double CICIDSLoader::parse_timestamp(const string& timestamp_str) {
    struct tm tm = {};
    istringstream ss(timestamp_str);
    ss >> get_time(&tm, "%m/%d/%Y %H:%M");

    if (ss.fail()) {
        FATAL_ERROR("Failed to parse timestamp: " + timestamp_str);
        return 0.0;
    }

    auto time_point = chrono::system_clock::from_time_t(mktime(&tm));
    auto timestamp = chrono::duration<double, milli>(time_point.time_since_epoch()).count();
    return timestamp;
}

void CICIDSLoader::Load() {

    __START_FTIMMER__

    ifstream file(data_path_);
    if (!file.is_open()) {
        FATAL_ERROR("Failed to open file: " + data_path_);
        return;
    }

    string line;
    getline(file, line);

    vector<pair<FlowRecord, size_t>> all;

    if (data_size_ == -1) {
        data_size_ = numeric_limits<int>::max();
    }
    
    size_t cnt = 0;
    while (getline(file, line) && cnt < data_size_) {
        vector<string> fields;
        string_view line_view(line);

        size_t start = 0;
        size_t end = 0;
        while ((end = line_view.find(',', start)) != string_view::npos) {
            fields.push_back(string(line_view.substr(start, end - start)));
            start = end + 1;
        }
        fields.push_back(string(line_view.substr(start)));

        if (fields.size() < 85) continue;

        FlowRecord fr;

        try {
            fr.src_ip = convert_str_addr4(fields[1]);
            fr.dst_ip = convert_str_addr4(fields[3]);

            fr.src_port = static_cast<uint16_t>(stoi(fields[2])); 
            fr.dst_port = static_cast<uint16_t>(stoi(fields[4])); 
            fr.proto = static_cast<uint16_t>(stoi(fields[5]));

            double start_timestamp = parse_timestamp(fields[6]);
            fr.ts_start.tv_sec = static_cast<time_t>(start_timestamp / 1000);
            fr.ts_start.tv_nsec = static_cast<long>((start_timestamp - fr.ts_start.tv_sec * 1000) * 1000000);

            long end_time_ms = start_timestamp + stoi(fields[7]);
            fr.ts_end.tv_sec = end_time_ms / 1000;
            fr.ts_end.tv_nsec = (end_time_ms % 1000) * 1000000;

            fr.packets = static_cast<size_t>(stoi(fields[8]));
            fr.bytes = static_cast<size_t>(stoi(fields[10]));
        } catch (const invalid_argument& e) {
            WARNF("Invalid argument encountered while parsing line: %s", line.c_str());
            continue;
        } catch (const out_of_range& e) {
            WARNF("Out of range error while parsing line: %s", line.c_str());
            continue;
        }

        string label = fields.back();
        label.erase(remove_if(label.begin(), label.end(), ::isspace), label.end());
        size_t y = (label == "BENIGN" ? 0 : 1);

        all.emplace_back(move(fr), y);
        ++cnt;
    }

    size_t train_size = static_cast<size_t>(train_ratio_ * all.size());
    all_data_ptr_->assign(all.begin(), all.end());
    train_data_ptr_->assign(all.begin(), all.begin() + train_size);
    test_data_ptr_->assign(all.begin() + train_size, all.end());

    size_t benign_count = 0, attack_count = 0;
    for (const auto& p : all) {
        p.second == 0 ? benign_count++ : attack_count++;
    }

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    LOGF("📦 Loaded total: %zu | Train: %zu | Test: %zu | BENIGN: %zu | ATTACK: %zu", 
        all.size(), 
        train_data_ptr_->size(), 
        test_data_ptr_->size(), 
        benign_count, 
        attack_count
    );
}