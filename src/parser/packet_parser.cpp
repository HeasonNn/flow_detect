#include "packet_parser.hpp"


shared_ptr<basic_packet> PacketParser::ParsePacket(pcpp::RawPacket& raw_pkt) {
    auto parsed_packet_ptr = make_shared<pcpp::Packet>(&raw_pkt, false, pcpp::IP, pcpp::OsiModelNetworkLayer);

    pkt_ts_t packet_time = raw_pkt.getPacketTimeStamp();
    pkt_code_t packet_code = 0;
    pkt_port_t s_port = 0, d_port = 0;
    pkt_len_t packet_length = 0;
    pkt_addr4_t s4, d4;
    pkt_addr6_t s6, d6;

    auto _f_parse_udp = [&]() -> bool {
        auto* udp = parsed_packet_ptr->getLayerOfType<pcpp::UdpLayer>();
        if (!udp) return false;
        s_port = htons(udp->getUdpHeader()->portSrc);
        d_port = htons(udp->getUdpHeader()->portDst);
        set_pkt_code(packet_code, pkt_type_t::UDP);
        return true;
    };

    auto _f_parse_tcp = [&]() -> bool {
        auto* tcp = parsed_packet_ptr->getLayerOfType<pcpp::TcpLayer>();
        if (!tcp) return false;
        s_port = htons(tcp->getTcpHeader()->portSrc);
        d_port = htons(tcp->getTcpHeader()->portDst);
        if (tcp->getTcpHeader()->synFlag) set_pkt_code(packet_code, pkt_type_t::TCP_SYN);
        if (tcp->getTcpHeader()->finFlag) set_pkt_code(packet_code, pkt_type_t::TCP_FIN);
        if (tcp->getTcpHeader()->rstFlag) set_pkt_code(packet_code, pkt_type_t::TCP_RST);
        if (tcp->getTcpHeader()->ackFlag) set_pkt_code(packet_code, pkt_type_t::TCP_ACK);
        return true;
    };

    auto _f_load_ipv6_addr_byte = [](const pcpp::IPv6Address& addr6) -> pkt_addr6_t {
        __pkt_addr6 __t;
        memcpy(__t.byte_rep, addr6.toBytes(), sizeof(__t));
        return __t.num_rep;
    };

    shared_ptr<basic_packet> res_ptr = nullptr;
    pcpp::ProtocolType type_next;

    if (parsed_packet_ptr->isPacketOfType(pcpp::IPv4)) {
        auto* ip = parsed_packet_ptr->getLayerOfType<pcpp::IPv4Layer>();
        set_pkt_code(packet_code, pkt_type_t::IPv4);
        s4 = ip->getSrcIPv4Address().toInt();
        d4 = ip->getDstIPv4Address().toInt();
        packet_length = htons(ip->getIPv4Header()->totalLength);
        ip->parseNextLayer();
        type_next = ip->getNextLayer() ? ip->getNextLayer()->getProtocol() : pcpp::UnknownProtocol;
    } else if (parsed_packet_ptr->isPacketOfType(pcpp::IPv6)) {
        auto* ip = parsed_packet_ptr->getLayerOfType<pcpp::IPv6Layer>();
        set_pkt_code(packet_code, pkt_type_t::IPv6);
        s6 = _f_load_ipv6_addr_byte(ip->getSrcIPv6Address());
        d6 = _f_load_ipv6_addr_byte(ip->getDstIPv6Address());
        packet_length = htons(ip->getIPv6Header()->payloadLength);
        ip->parseNextLayer();
        type_next = ip->getNextLayer() ? ip->getNextLayer()->getProtocol() : pcpp::UnknownProtocol;
    } else {
        return make_shared<basic_packet_bad>(packet_time);
    }

    switch (type_next) {
        case pcpp::TCP:
            if (!_f_parse_tcp()) return make_shared<basic_packet_bad>(packet_time);
            break;
        case pcpp::UDP:
            if (!_f_parse_udp()) return make_shared<basic_packet_bad>(packet_time);;
            break;
        case pcpp::ICMP:
            set_pkt_code(packet_code, pkt_type_t::ICMP);
            break;
        case pcpp::IGMP:
            set_pkt_code(packet_code, pkt_type_t::IGMP);
            break;
        default:
            set_pkt_code(packet_code, pkt_type_t::UNKNOWN);
            break;
    }

    if (test_pkt_code(packet_code, pkt_type_t::IPv4)) {
        res_ptr = make_shared<basic_packet4>(s4, d4, s_port, d_port, packet_time, packet_code, packet_length);
    } else if (test_pkt_code(packet_code, pkt_type_t::IPv6)) {
        res_ptr = make_shared<basic_packet6>(s6, d6, s_port, d_port, packet_time, packet_code, packet_length);
    } else {
        return make_shared<basic_packet_bad>(packet_time);
    }

    return res_ptr;
}


size_t PacketParser::ParseAll(int data_size, size_t multiplex) {
    if (data_size == -1) {
        data_size = numeric_limits<int>::max();
    }
    
    size_t total_packet_count = 0;
    {
        pcpp::PcapFileReaderDevice reader(pcap_file_path_);
        reader.open();
        pcpp::RawPacket tmp;
        while (reader.getNextPacket(tmp) && total_packet_count < data_size) total_packet_count++;
        reader.close();
    }

    packet_vec_ptr = make_shared<vector<shared_ptr<basic_packet>>>(total_packet_count);
    size_t part_size = (total_packet_count + multiplex - 1) / multiplex;
    vector<pair<size_t, size_t>> thread_ranges;
    for (size_t core = 0, idx = 0; core < multiplex; ++core, idx += part_size) {
        thread_ranges.emplace_back(idx, min(idx + part_size, total_packet_count));
    }

    vector<thread> vt;
    for (size_t core = 0; core < multiplex; ++core) {
        vt.emplace_back([this, &thread_ranges, core]() {
            const size_t start_idx = thread_ranges[core].first;
            const size_t end_idx   = thread_ranges[core].second;
            
            size_t curr_idx = 0;
            pcpp::RawPacket raw_pkt;
            pcpp::PcapFileReaderDevice local_reader(pcap_file_path_);
            local_reader.open();
            while (curr_idx < start_idx && local_reader.getNextPacket(raw_pkt)) curr_idx++;
            while (curr_idx < end_idx && local_reader.getNextPacket(raw_pkt)) {
                packet_vec_ptr->at(curr_idx) = ParsePacket(raw_pkt);
                ++curr_idx;
            }
            local_reader.close();
        });
    }
    for (auto& t : vt) t.join();
    
    return packet_vec_ptr->size();
}