#pragma once
#include <cstdint>
#include <cstring>
#include "robin_hood.hpp"   // 单头文件即可

class FixedRollingCounter
{
public:
    explicit FixedRollingCounter(uint64_t window_ms, size_t capacity = 65536)
        : window_(window_ms), cap_(capacity)
    {
        slots_   = new Slot[cap_];
        bitmap_  = new uint8_t[(cap_ + 7) / 8];
        std::memset(slots_, 0, cap_ * sizeof(Slot));
        std::memset(bitmap_, 0, (cap_ + 7) / 8);
    }

    ~FixedRollingCounter()
    {
        delete[] slots_;
        delete[] bitmap_;
    }

    FixedRollingCounter(const FixedRollingCounter &) = delete;
    FixedRollingCounter &operator=(const FixedRollingCounter &) = delete;

    /* 加入记录；若数组满则丢弃最旧 */
    void add(uint64_t ts, uint32_t ip)
    {
        /* 先清过期 */
        while (head_ != tail_ && (ts - slots_[head_].ts > window_))
        {
            uint32_t old_ip = slots_[head_].ip;
            if (--cnt_[old_ip] == 0)
                --unique_;
            clear_bit(head_);
            head_ = (head_ + 1) % cap_;
        }

        /* 若满，强制丢弃最旧 */
        if (((tail_ + 1) % cap_) == head_)
        {
            uint32_t old_ip = slots_[head_].ip;
            if (--cnt_[old_ip] == 0)
                --unique_;
            clear_bit(head_);
            head_ = (head_ + 1) % cap_;
        }

        /* 插入新记录 */
        size_t idx = tail_;
        slots_[idx] = {ts, ip};
        set_bit(idx);
        tail_ = (tail_ + 1) % cap_;
        if (cnt_[ip]++ == 0)
            ++unique_;
    }

    size_t unique() const { return unique_; }

private:
    struct Slot
    {
        uint64_t ts;
        uint32_t ip;
    };

    uint64_t window_;
    size_t   cap_;
    Slot    *slots_;
    uint8_t *bitmap_;
    size_t   head_ = 0;
    size_t   tail_ = 0;

    /* 位图操作 */
    inline void set_bit(size_t i)   { bitmap_[i >> 3] |=  (1u << (i & 7)); }
    inline void clear_bit(size_t i) { bitmap_[i >> 3] &= ~(1u << (i & 7)); }

    /* 哈希表：去重计数 */
    robin_hood::unordered_flat_map<uint32_t, uint32_t> cnt_;
    uint32_t unique_ = 0;
};