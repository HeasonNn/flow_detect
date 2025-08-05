#pragma once

#include <chrono>
#include <unordered_map>
#include <list>
#include <vector>
#include <armadillo>
#include <condition_variable>
#include <queue>

#include "../common.hpp"
#include "flow_feature.hpp"

using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using Dur = std::chrono::milliseconds;


inline Time to_time_point(const timespec& ts) {
    return Time(std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec));
}

inline double to_double_seconds(const Time& tp) {
    auto duration = tp.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

/* ---------- 元数据 ---------- */
struct EdgeMeta {
    size_t count = 0;
    Time last_seen; // 记录该边最后一次出现的真实时间
};

struct NodeMeta {
    size_t out_deg = 0;
    size_t in_deg = 0;
    Time last_seen; // 记录该节点最后一次出现的真实时间
};

#define TIME_NEAR_SHIFT 8
#define TIME_NEAR (1 << TIME_NEAR_SHIFT)
#define TIME_LEVEL_SHIFT 6
#define TIME_LEVEL (1 << TIME_LEVEL_SHIFT)
#define TIME_NEAR_MASK (TIME_NEAR - 1)
#define TIME_LEVEL_MASK (TIME_LEVEL - 1)

template <class K>
class TimingWheel {
private:
    struct TimerNode {
        K key;
        uint32_t expire;
        typename std::list<TimerNode>::iterator iter_in_list;
    };

    struct LinkList {
        std::list<TimerNode> nodes;
        typename std::list<TimerNode>::iterator tail_iter;

        LinkList() : tail_iter(nodes.end()) {};

        void clear() noexcept {
            nodes.clear();
            tail_iter = nodes.end();
        }

        void push_back(TimerNode&& node) {
            nodes.emplace_back(std::move(node));
            tail_iter = std::prev(nodes.end());
        }

        [[nodiscard]] bool empty() const noexcept { return nodes.empty(); }
    };

    // 主轮：TIME_NEAR 个槽
    // 次轮：4 层，每层 TIME_LEVEL 个槽
    std::array<LinkList, TIME_NEAR> near;
    std::array<std::array<LinkList, TIME_LEVEL>, 4> t;

    // 当前时间指针 (tick)
    uint32_t time;

    // 起始时间点，用于将 Time 转换为 tick
    Time start_time_point;
    // 每个 tick 的持续时间
    Dur tick_duration;

    // key 到节点的映射，用于 O(1) 取消
    std::unordered_map<K, std::unique_ptr<TimerNode>> key_index;

    // 将 Time 转换为 tick
    [[nodiscard]] uint32_t time_to_tick(Time tp) const {
        if (tp <= start_time_point) return 0;
        auto diff = std::chrono::duration_cast<Dur>(tp - start_time_point);
        return static_cast<uint32_t>(diff.count() / tick_duration.count());
    }

    // 将 tick 转换为 Time
    [[nodiscard]] Time tick_to_time(uint32_t tick) const {
        return start_time_point + tick_duration * static_cast<int64_t>(tick);
    }

    // 清空链表并返回其拥有的节点
    [[nodiscard]] std::list<TimerNode> link_clear(LinkList& list) {
        std::list<TimerNode> ret;
        ret.swap(list.nodes);
        list.tail_iter = ret.end();
        return ret;
    }

    // 添加任务节点到定时器
    void add_node(TimerNode* node) {
        uint32_t expire_tick = node->expire;
        uint32_t current_tick = time;
        uint32_t remaining_ticks = expire_tick - current_tick; // 剩余的 tick 数

        // 通过移位和掩码计算层级和槽位，避免了昂贵的除法和取模
        if (remaining_ticks < TIME_NEAR) {
            // 放入 near 轮
            near[expire_tick & TIME_NEAR_MASK].push_back(std::move(*node));
        } 
        else {
            // 计算需要多少个 level_shift
            // 从 t[0] 开始检查
            uint32_t shift = TIME_NEAR_SHIFT;
            for (int level = 0; level < 4; ++level) {
                // 计算该层级的掩码
                uint32_t level_mask = (1U << (shift + TIME_LEVEL_SHIFT)) - 1;
                if (remaining_ticks < level_mask) {
                    t[level][(expire_tick >> shift) & TIME_LEVEL_MASK].push_back(std::move(*node));
                    return;
                }
                shift += TIME_LEVEL_SHIFT;
            }
            // 如果剩余时间太长，放入最外层 t[3]
            t[3][(expire_tick >> shift) & TIME_LEVEL_MASK].push_back(std::move(*node));
        }
    }

    // 移动链表：将 level 层 idx 槽的链表重新添加到定时器
    void move_list(int level, int idx) {
        auto current_list = link_clear(t[level][idx]);
        for (auto& node : current_list) {
            add_node(&node);
        }
    }

    // 时间推进核心：处理进位和重新映射
    void timer_shift() {
        uint32_t ct = ++time;
        if (ct == 0) {
            // 时间溢出，重置并移动 t[3][0]
            move_list(3, 0);
        } else {
            uint32_t time_val = ct >> TIME_NEAR_SHIFT;
            int i = 0;
            int mask = TIME_NEAR;

            while ((ct & (mask - 1)) == 0) {
                int idx = time_val & TIME_LEVEL_MASK;
                if (idx != 0) {
                    move_list(i, idx);
                    break;
                }
                mask <<= TIME_LEVEL_SHIFT;
                time_val >>= TIME_LEVEL_SHIFT;
                ++i;
            }
        }
    }

    // 执行当前 near 轮中过期的任务
    void timer_execute(const std::function<void(const K&)>& on_expire) {
        int idx = time & TIME_NEAR_MASK;
        auto current_list = link_clear(near[idx]);

        for (auto& node : current_list) {
            key_index.erase(node.key); // 从索引中移除
            if (on_expire) {
                on_expire(node.key);
            }
        }
    }

public:
    explicit TimingWheel(Dur tick_duration, Time start_time)
        : tick_duration(tick_duration), start_time_point(start_time), time(0) 
    {
        if (tick_duration.count() <= 0) {
            throw std::invalid_argument("tick_duration must be positive");
        }
    }

    ~TimingWheel() {
        try {
            clear([](const K&){});
        } 
        catch (...) {}
    }

    void insert(const K& key, Time expire_time) {
        uint32_t expire_tick = time_to_tick(expire_time);

        if (expire_tick <= time) return;

        // 创建节点
        auto node = std::make_unique<TimerNode>();
        node->key = key;
        node->expire = expire_tick;

        // 将节点添加到定时器
        add_node(node.get());

        // 将唯一所有权存入 key_index，用于取消
        key_index[key] = std::move(node);
    }

    [[nodiscard]] bool cancel(const K& key) {
        return key_index.erase(key) > 0; // unique_ptr 自动释放
    }

    void advance(Time now, std::function<void(const K&)> on_expire) {
        uint32_t now_tick = time_to_tick(now);

        while (time < now_tick) {
            timer_execute(on_expire);  // 执行当前 tick 的任务
            timer_shift();             // 推进 tick 并处理重新映射
            timer_execute(on_expire);  // 再次执行，处理因重新映射而进入 near 轮的任务
        }
    }

    void clear(std::function<void(const K&)> on_expire) {
        advance(tick_to_time(time + 1000000), on_expire);
        key_index.clear();
    }

    void print_levels() const {
        std::cout << "Level near: slots=" << TIME_NEAR << ", span=" 
                  << (TIME_NEAR * tick_duration.count()) << "ms (" 
                  << (TIME_NEAR * tick_duration.count() / 1000.0) << "s)" 
                  << ", current_idx=" << (time & TIME_NEAR_MASK) << std::endl;
        for (int i = 0; i < 4; ++i) {
            std::cout << "Level t" << i << ": slots=" << TIME_LEVEL << ", span=" 
                      << ((1 << ((i+1)*TIME_LEVEL_SHIFT)) * tick_duration.count()) 
                      << "ms" << std::endl;
        }
    }

    void print_slot_sizes(const std::string& prefix = "") const {
        std::string label = prefix.empty() ? "Slot Sizes" : ("Slot Sizes [" + prefix + "]");
        std::cout << label << ": ";
        for (int i = 0; i < TIME_NEAR; ++i) {
            std::cout << near[i].nodes.size() << " ";
        }
        std::cout << " (tick=" << tick_duration.count() << "ms, current_idx=" << (time & TIME_NEAR_MASK) << ")" << std::endl;

        for (int level = 0; level < 4; ++level) {
            std::cout << "Slot Sizes [t" << level << "]: ";
            for (int i = 0; i < TIME_LEVEL; ++i) {
                std::cout << t[level][i].nodes.size() << " ";
            }
            std::cout << std::endl;
        }
    }

    struct Stats {
        size_t total_keys = 0;
        size_t level_count = 0;
        std::vector<size_t> keys_per_level;
    };

    [[nodiscard]] Stats get_stats() const {
        Stats stats;
        stats.total_keys = key_index.size();
        for (int i = 0; i < TIME_NEAR; ++i) {
            stats.keys_per_level.push_back(near[i].nodes.size());
        }
        for (int l = 0; l < 4; ++l) {
            for (int i = 0; i < TIME_LEVEL; ++i) {
                stats.keys_per_level.push_back(t[l][i].nodes.size());
            }
        }
        stats.level_count = 5;
        return stats;
    }
};

/* ---------- GraphMaintainer ---------- */
class GraphMaintainer {
private:
    std::unordered_map<std::pair<uint32_t, uint32_t>, EdgeMeta> edge_map_;
    std::unordered_map<uint32_t, NodeMeta> node_map_;

    // 时间轮：只存储 key 和 expire_time
    TimingWheel<std::pair<uint32_t, uint32_t>> edge_wheel_;
    TimingWheel<uint32_t> node_wheel_;

    // 索引结构
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> out_edges_;
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> in_edges_;

    Dur ttl_; // 所有条目的 TTL
    Dur prune_interval_; // 用于惰性清理的间隔

    // 最后一次推进时间轮的时间
    Time last_prune_time_;

    // 保护共享数据的互斥锁
    mutable std::mutex mutex_;

public:
    // 构造函数
    GraphMaintainer(Dur ttl, Dur wheel_granularity, Time start_time)
        : ttl_(ttl), 
          edge_wheel_(wheel_granularity, start_time),
          node_wheel_(wheel_granularity, start_time),
          last_prune_time_(Time()) {}

    // 析构函数
    ~GraphMaintainer() {
        std::vector<std::pair<uint32_t, uint32_t>> expired_edges;
        std::vector<uint32_t> expired_nodes;

        edge_wheel_.clear([&](const auto& edge_key) {
            expired_edges.push_back(edge_key);
        });

        node_wheel_.clear([&](uint32_t ip) {
            expired_nodes.push_back(ip);
        });

        for (const auto& edge_key : expired_edges) {
            auto map_it = edge_map_.find(edge_key);
            if (map_it != edge_map_.end()) {
                handle_edge_removal(edge_key, map_it->second);
                edge_map_.erase(map_it);
            }
        }

        for (uint32_t ip : expired_nodes) {
            auto map_it = node_map_.find(ip);
            if (map_it != node_map_.end()) {
                const auto& meta = map_it->second;
                if (meta.in_deg == 0 && meta.out_deg == 0) {
                    handle_node_removal(ip, meta);
                    node_map_.erase(map_it);
                }
            }
        }
    }

    void update(uint32_t src, uint32_t dst, Time ts_start, Time ts_end) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto edge_key = std::make_pair(src, dst);
        auto& em = edge_map_[edge_key];
        const bool fresh_edge = (em.count == 0);
        em.count++;
        em.last_seen = ts_start;

        auto& nm_src = node_map_[src];
        auto& nm_dst = node_map_[dst];

        if (fresh_edge) {
            nm_src.out_deg++;
            nm_dst.in_deg++;
            out_edges_[src].insert(dst);
            in_edges_[dst].insert(src);
        }

        nm_src.last_seen = nm_dst.last_seen = ts_start;

        // 计算过期时间
        Time expire_time = ts_end + ttl_;

        // 将 key 和 expire_time 插入时间轮
        edge_wheel_.insert(edge_key, expire_time);
        node_wheel_.insert(src, expire_time);
        node_wheel_.insert(dst, expire_time);
    }

    // prune 函数：使用流时间推进时间轮
    void prune(Time now) {
        std::lock_guard<std::mutex> lock(mutex_);

        // if (last_prune_time_ == Time() || (now - last_prune_time_) >= prune_interval_) 
        {
            // 推进边的时间轮
            // edge_wheel_.print_slot_sizes("[edge_wheel] Before Advance");
            // node_wheel_.print_slot_sizes("[node_wheel] Before Advance");
            
            // === 1. 先收集过期 key ===
            std::vector<std::pair<uint32_t, uint32_t>> expired_edges;
            std::vector<uint32_t> expired_nodes;

            edge_wheel_.advance(now, [&](const auto& edge_key) {
                expired_edges.push_back(edge_key);
            });

            node_wheel_.advance(now, [&](uint32_t ip) {
                expired_nodes.push_back(ip);
            });

            // === 2. 回调结束后，再统一处理删除 ===
            for (const auto& edge_key : expired_edges) {
                auto map_it = edge_map_.find(edge_key);
                if (map_it != edge_map_.end()) {
                    handle_edge_removal(edge_key, map_it->second);
                    edge_map_.erase(map_it);
                }
            }

            for (uint32_t ip : expired_nodes) {
                auto map_it = node_map_.find(ip);
                if (map_it != node_map_.end()) {
                    const auto& meta = map_it->second;
                    // 只有孤立节点才可删
                    if (meta.in_deg == 0 && meta.out_deg == 0) {
                        handle_node_removal(ip, meta);
                        node_map_.erase(map_it);
                    }
                }
            }

            // edge_wheel_.print_slot_sizes("[edge_wheel] After Advance");
            // node_wheel_.print_slot_sizes("[node_wheel] After Advance");

            last_prune_time_ = now;
        }
    }

    const NodeMeta* node(uint32_t ip, Time now) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = node_map_.find(ip);
        if (it != node_map_.end()) {
            // 检查是否过期
            if (now - it->second.last_seen <= ttl_) {
                return &it->second;
            }
        }
        return nullptr;
    }

    const EdgeMeta* edge(uint32_t s, uint32_t d, Time now) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = std::make_pair(s, d);
        auto it = edge_map_.find(key);
        if (it != edge_map_.end()) {
            if (now - it->second.last_seen <= ttl_) {
                return &it->second;
            }
        }
        return nullptr;
    }

    size_t active_nodes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return node_map_.size();
    }

    size_t active_edges() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return edge_map_.size();
    }

private:
    void handle_node_removal(uint32_t ip, const NodeMeta& meta) {
        auto out_it = out_edges_.find(ip);
        if (out_it != out_edges_.end()) {
            for (auto dst : out_it->second) {
                if (auto nit = node_map_.find(dst); nit != node_map_.end() && nit->second.in_deg > 0) {
                    nit->second.in_deg--;
                }
            }
            out_edges_.erase(out_it);
        }

        auto in_it = in_edges_.find(ip);
        if (in_it != in_edges_.end()) {
            for (auto src : in_it->second) {
                if (auto nit = node_map_.find(src); nit != node_map_.end() && nit->second.out_deg > 0) {
                    nit->second.out_deg--;
                }
            }
            in_edges_.erase(in_it);
        }
    }

    void handle_edge_removal(const std::pair<uint32_t, uint32_t>& edge, const EdgeMeta& meta) {
        uint32_t src = edge.first;
        uint32_t dst = edge.second;
        if (auto node_it = node_map_.find(src); node_it != node_map_.end()) {
            if (node_it->second.out_deg > 0) {
                node_it->second.out_deg--;
            }
        }
        if (auto node_it = node_map_.find(dst); node_it != node_map_.end()) {
            if (node_it->second.in_deg > 0) {
                node_it->second.in_deg--;
            }
        }
        auto out_it = out_edges_.find(src);
        if (out_it != out_edges_.end()) {
            out_it->second.erase(dst);
            if (out_it->second.empty()) {
                out_edges_.erase(out_it);
            }
        }
        auto in_it = in_edges_.find(dst);
        if (in_it != in_edges_.end()) {
            in_it->second.erase(src);
            if (in_it->second.empty()) {
                in_edges_.erase(in_it);
            }
        }
    }
};

/* ---------- 特征提取器 ---------- */
class GraphFeatureExtractor {
public:
    explicit GraphFeatureExtractor(const nlohmann::json& cfg, Time start_time);

    // 传入 flow 时间戳，不再依赖系统时间
    void updateGraph(const FlowRecord& f);
    arma::vec extract(const FlowRecord& f) const;

private:
    const json& cfg_;

    void maybe_prune(Time now); // 显式传入当前时间点

    std::unique_ptr<GraphMaintainer> g_;
    Dur prune_interval_;
    Time last_prune_ts_; // 但这不再是系统时间，而是上一次处理的流时间
};