#pragma once

#include <vector>
#include <cstdint>
#include <functional>
#include <random>
#include <limits>

// 支持 uint32_t/uint64_t 的高效 Bloom Filter
class BloomFilter {
public:
    BloomFilter(size_t num_bits = 1<<18, size_t num_hashes = 8)
        : bits_((num_bits+63)/64, 0), bitmask_(num_bits-1), num_hashes_(num_hashes)
    {
        // 使用 std::hash+盐生成多个哈希
        std::seed_seq seed{2025};
        std::mt19937_64 rng(seed);
        for (size_t i = 0; i < num_hashes_; ++i)
            salts_.push_back(rng());
    }

    // 支持 uint32_t 输入
    void insert(uint32_t v) { insert64(static_cast<uint64_t>(v)); }
    void insert(uint64_t v) { insert64(v); }
    
    // 支持 uint64_t 输入
    bool contains(uint64_t v) const { return contains64(v); }
    bool contains(uint32_t v) const { return contains64(static_cast<uint64_t>(v)); }

    // 可选：支持任意二进制 buffer
    void insert(const void* data, size_t len) {
        uint64_t hv = simple_hash(data, len, 0);
        insert64(hv);
    }
    
    bool contains(const void* data, size_t len) const {
        uint64_t hv = simple_hash(data, len, 0);
        return contains64(hv);
    }

    void clear() { std::fill(bits_.begin(), bits_.end(), 0); }

private:
    std::vector<uint64_t> bits_;
    size_t bitmask_; // 用于模运算
    size_t num_hashes_;
    std::vector<uint64_t> salts_;

    // 主要插入/查询逻辑
    void insert64(uint64_t v) {
        for (size_t i = 0; i < num_hashes_; ++i) {
            size_t idx = hash_i(v, i) & bitmask_;
            bits_[idx/64] |= (uint64_t(1) << (idx%64));
        }
    }
    bool contains64(uint64_t v) const {
        for (size_t i = 0; i < num_hashes_; ++i) {
            size_t idx = hash_i(v, i) & bitmask_;
            if (!(bits_[idx/64] & (uint64_t(1) << (idx%64))))
                return false;
        }
        return true;
    }

    // 多哈希：FNV1a+salt（快速/无冲突）
    size_t hash_i(uint64_t x, size_t i) const {
        uint64_t h = x ^ salts_[i];
        h ^= (h >> 33);
        h *= 0xff51afd7ed558ccdULL;
        h ^= (h >> 33);
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= (h >> 33);
        return static_cast<size_t>(h);
    }

    // 简易通用哈希（可用于 buffer 数据）
    static uint64_t simple_hash(const void* data, size_t len, uint64_t salt) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        uint64_t h = 0xcbf29ce484222325ULL ^ salt;
        for (size_t i = 0; i < len; ++i)
            h = (h ^ p[i]) * 0x100000001b3ULL;
        return h;
    }
};