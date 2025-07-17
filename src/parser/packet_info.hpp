#pragma once

#include <netinet/in.h>
#include <pcapplusplus/Packet.h>
#include <pcapplusplus/IPv4Layer.h>
#include <pcapplusplus/IPv6Layer.h>
#include <pcapplusplus/TcpLayer.h>
#include <pcapplusplus/UdpLayer.h>
#include <pcapplusplus/PcapFileDevice.h>

#include <array>
#include <tuple>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#include "../common.hpp"

using namespace std;

using pkt_addr4_t = uint32_t;
using pkt_addr6_t = __uint128_t;
using pkt_len_t = uint16_t;
using pkt_port_t = uint16_t;
using pkt_ts_t = timespec;
using pkt_code_t = uint16_t;
using stack_code_t = uint16_t;

union __pkt_addr6 {
    __uint128_t num_rep;
    uint8_t byte_rep[16];
};

enum pkt_type_t : uint8_t {
    IPv4,
    IPv6,
    ICMP,
    IGMP,
    TCP_SYN,
    TCP_ACK,
    TCP_FIN,
    TCP_RST,
    UDP,
    UNKNOWN
};

static const vector<const char*> pkt_type_names = {
    "IPv4", 
    "IPv6", 
    "ICMP", 
    "IGMP",
    "TCP_SYN", 
    "TCP_ACK", 
    "TCP_FIN", 
    "TCP_RST",
    "UDP", 
    "UNKNOWN"
};

inline constexpr pkt_code_t to_pkt_code(pkt_type_t t) {
    return static_cast<pkt_code_t>(1 << static_cast<uint8_t>(t));
}

inline void set_pkt_code(pkt_code_t& code, pkt_type_t t) {
    code |= to_pkt_code(t);
}

inline bool test_pkt_code(pkt_code_t code, pkt_type_t t) {
    return (code & to_pkt_code(t)) != 0;
}

enum stack_type_t : uint8_t {
    F_ICMP,
    F_IGMP,
    F_TCP,
    F_UDP,
    F_UNKNOWN
};

static const vector<const char*> stack_type_names = {
    "ICMP", "IGMP", "TCP", "UDP", "UNKNOWN"
};

inline constexpr stack_code_t to_stack_code(stack_type_t t) {
    return static_cast<stack_code_t>(1 << static_cast<uint8_t>(t));
}

inline stack_code_t convert_pkt_code_to_stack(pkt_code_t code) {
    if (test_pkt_code(code, pkt_type_t::ICMP)) return to_stack_code(stack_type_t::F_ICMP);
    if (test_pkt_code(code, pkt_type_t::IGMP)) return to_stack_code(stack_type_t::F_IGMP);
    if (test_pkt_code(code, pkt_type_t::UDP))  return to_stack_code(stack_type_t::F_UDP);
    if (test_pkt_code(code, pkt_type_t::UNKNOWN)) return to_stack_code(stack_type_t::F_UNKNOWN);
    return to_stack_code(stack_type_t::F_TCP); // default: TCP
}

inline stack_type_t stack_code_to_type(stack_code_t code) {
    if (code == 0) return stack_type_t::F_UNKNOWN;
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<stack_type_t>(__builtin_ctz(code));
#else
    for (uint8_t i = 0; i <= static_cast<uint8_t>(stack_type_t::F_UNKNOWN); ++i) {
        if (code == (1 << i)) return static_cast<stack_type_t>(i);
    }
    return stack_type_t::F_UNKNOWN;
#endif
}

using tuple2_conn4 = tuple<pkt_addr4_t, pkt_addr4_t>;
using tuple2_conn6 = tuple<pkt_addr6_t, pkt_addr6_t>;
using tuple4_conn4 = tuple<pkt_addr4_t, pkt_addr4_t, pkt_port_t, pkt_port_t>;
using tuple4_conn6 = tuple<pkt_addr6_t, pkt_addr6_t, pkt_port_t, pkt_port_t>;
using tuple5_conn4 = tuple<pkt_addr4_t, pkt_addr4_t, pkt_port_t, pkt_port_t, stack_code_t>;
using tuple5_conn6 = tuple<pkt_addr6_t, pkt_addr6_t, pkt_port_t, pkt_port_t, stack_code_t>;

template <typename Tuple>
inline auto tuple_get_src_addr(const Tuple& cn) { return get<0>(cn); }

template <typename Tuple>
inline auto tuple_get_dst_addr(const Tuple& cn) { return get<1>(cn); }

template <typename Tuple>
inline auto tuple_get_src_port(const Tuple& cn) { return get<2>(cn); }

template <typename Tuple>
inline auto tuple_get_dst_port(const Tuple& cn) { return get<3>(cn); }

template <typename Tuple>
inline auto tuple_get_stack(const Tuple& cn) { return get<4>(cn); }

template <typename Tuple>
inline auto tuple_is_stack(const Tuple& cn, stack_type_t tp) {
    return get<4>(cn) & to_stack_code(tp);
}

inline tuple5_conn4 reverse_tuple(const tuple5_conn4& cn) {
    return {get<1>(cn), get<0>(cn), get<3>(cn), get<2>(cn), get<4>(cn)};
}

inline tuple5_conn6 reverse_tuple(const tuple5_conn6& cn) {
    return {get<1>(cn), get<0>(cn), get<3>(cn), get<2>(cn), get<4>(cn)};
}

inline tuple5_conn4 extend_tuple(const tuple4_conn4& cn, stack_code_t code) {
    return {get<0>(cn), get<1>(cn), get<2>(cn), get<3>(cn), code};
}
inline tuple5_conn6 extend_tuple(const tuple4_conn6& cn, stack_code_t code) {
    return {get<0>(cn), get<1>(cn), get<2>(cn), get<3>(cn), code};
}

inline string get_str_addr(pkt_addr4_t addr) {
    return pcpp::IPv4Address(addr).toString();
}

inline string get_str_addr(pkt_addr6_t addr) {
    __pkt_addr6 __t;
    __t.num_rep = addr;
    return pcpp::IPv6Address(__t.byte_rep).toString();
}

inline pkt_addr4_t convert_str_addr4(const string& str) {
    pcpp::IPv4Address ip(str);
    if (!ip.isValid()) throw invalid_argument("Invalid IPv4 address: " + str);
    return ip.toInt();
}

inline pkt_addr6_t convert_str_addr6(const string& str) {
    pcpp::IPv6Address ip(str);
    if (!ip.isValid()) throw invalid_argument("Invalid IPv6 address: " + str);
    __pkt_addr6 __t;
    memcpy(__t.byte_rep, ip.toBytes(), sizeof(__t));
    return __t.num_rep;
}

inline __uint128_t string_2_uint128(const string& str) {
    __uint128_t result = 0;
    for (char ch : str) {
        if (ch < '0' || ch > '9') throw invalid_argument("Invalid digit in input");
        result = result * 10 + (ch - '0');
    }
    return result;
}

inline string uint128_2_string(__uint128_t value) {
    if (value == 0) return "0";
    string result;
    while (value > 0) {
        result += '0' + (value % 10);
        value /= 10;
    }
    reverse(result.begin(), result.end());
    return result;
}
