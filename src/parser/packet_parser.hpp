#pragma once

#include "packet.hpp"


using namespace std;


class PacketParser
{
private:
    shared_ptr<vector<shared_ptr<basic_packet>>> packet_vec_ptr;

    const string pcap_file_path_;
    const optional<string> label_path_;

    shared_ptr<basic_packet> ParsePacket(pcpp::RawPacket& raw_pkt);

public:
    explicit PacketParser(const string& pcap_file_path, optional<string> label_path = nullopt)
        : packet_vec_ptr(nullptr),
          pcap_file_path_(pcap_file_path), label_path_(std::move(label_path)) {}

    PacketParser(const PacketParser&) = delete;
    PacketParser & operator=(const PacketParser&) = delete;

    ~PacketParser() = default;

    size_t ParseAll(int data_size, size_t multiplex = 16);
    
    auto inline getParseResult() -> const decltype(packet_vec_ptr) { return packet_vec_ptr;};
};