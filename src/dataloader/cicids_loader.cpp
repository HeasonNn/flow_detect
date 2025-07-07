#include "data_loader.hpp"

double CICIDSLoader::parse_timestamp(const std::string& timestamp_str) {
    struct tm tm = {};
    std::istringstream ss(timestamp_str);
    ss >> std::get_time(&tm, "%m/%d/%Y %H:%M");

    if (ss.fail()) {
        FATAL_ERROR("Failed to parse timestamp: " + timestamp_str);
        return 0.0;
    }

    auto time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    auto timestamp = std::chrono::duration<double, std::milli>(time_point.time_since_epoch()).count();
    return timestamp;
}

void CICIDSLoader::load() {

    __START_FTIMMER__

    std::ifstream file(data_path_);
    if (!file.is_open()) {
        FATAL_ERROR("Failed to open file: " + data_path_);
        return;
    }

    std::string line;
    std::getline(file, line);

    std::vector<std::pair<FlowRecord, size_t>> all;

    while (std::getline(file, line)) {
        std::vector<std::string> fields;
        std::string_view line_view(line);

        size_t start = 0;
        size_t end = 0;
        while ((end = line_view.find(',', start)) != std::string_view::npos) {
            fields.push_back(std::string(line_view.substr(start, end - start)));
            start = end + 1;
        }
        fields.push_back(std::string(line_view.substr(start)));

        if (fields.size() < 85) continue;

        FlowRecord fr;
        fr.src_ip = fields[1];
        fr.dst_ip = fields[3];
        fr.src_port = 0;
        fr.dst_port = 0;
        fr.proto = 0;

        try {
            fr.src_port = static_cast<uint16_t>(std::stoi(fields[2])); 
            fr.dst_port = static_cast<uint16_t>(std::stoi(fields[4])); 
            fr.proto = static_cast<uint16_t>(std::stoi(fields[5]));

            double start_timestamp = parse_timestamp(fields[6]);
            fr.ts_start.tv_sec = static_cast<time_t>(start_timestamp / 1000);
            fr.ts_start.tv_nsec = static_cast<long>((start_timestamp - fr.ts_start.tv_sec * 1000) * 1000000);

            long end_time_ms = start_timestamp + std::stoi(fields[7]);
            fr.ts_end.tv_sec = end_time_ms / 1000;
            fr.ts_end.tv_nsec = (end_time_ms % 1000) * 1000000;

            fr.packets = std::stoi(fields[8]);
            fr.bytes = std::stoi(fields[10]);
        } catch (const std::invalid_argument& e) {
            WARNF("Invalid argument encountered while parsing line: %s", line.c_str());
            continue;
        } catch (const std::out_of_range& e) {
            WARNF("Out of range error while parsing line: %s", line.c_str());
            continue;
        }

        std::string label = fields.back();
        label.erase(std::remove_if(label.begin(), label.end(), ::isspace), label.end());
        size_t y = (label == "BENIGN" ? 0 : 1);

        all.emplace_back(std::move(fr), y);
    }

    size_t train_size = static_cast<size_t>(train_ratio_ * all.size());
    train_data_ptr_->assign(all.begin(), all.begin() + train_size);
    test_data_ptr_->assign(all.begin() + train_size, all.end());

    size_t count0 = 0, count1 = 0;
    for (const auto& p : all) {
        p.second == 0 ? count0++ : count1++;
    }

    __STOP_FTIMER__
    __PRINTF_EXE_TIME__

    LOGF("Loaded total: %zu | Train: %zu | Test: %zu | BENIGN: %zu | ATTACK: %zu", 
        all.size(), 
        train_data_ptr_->size(), 
        test_data_ptr_->size(), 
        count0, 
        count1
    );
}