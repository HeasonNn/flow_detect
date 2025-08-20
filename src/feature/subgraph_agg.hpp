#pragma once
#include <armadillo>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <iostream>
#include <iomanip>
#include <memory>

// ======================= Feature indices (from GraphFeatureExtractor) =======================
namespace feat {
    // Node & Edge Local
    constexpr std::size_t SRC_OUT_DEG_LOG = 0;
    constexpr std::size_t SRC_IN_DEG_LOG  = 1;
    constexpr std::size_t DST_IN_DEG_LOG  = 2;
    constexpr std::size_t DST_OUT_DEG_LOG = 3;
    constexpr std::size_t ROLE_IMB_SRCABS = 4;  // |out-in| @src
    constexpr std::size_t ROLE_IMB_DSTABS = 5;  // |out-in| @dst
    constexpr std::size_t EDGE_FREQ_LOG   = 6;

    // traffic basics
    constexpr std::size_t BYTES_LOG       = 7;  // log1p(bytes)
    constexpr std::size_t PKTS_LOG        = 8;  // log1p(pkts)
    constexpr std::size_t DUR_LOG         = 9;  // log1p(duration_ms)
    constexpr std::size_t BW_LOG          = 10; // log1p(bytes/dur)
    constexpr std::size_t PPS_LOG         = 11; // log1p(pkts/dur)
    constexpr std::size_t PKT_SIZE        = 12; // bytes/packets
    constexpr std::size_t PROTO           = 13; // protocol id

    // Node & Edge Global
    constexpr std::size_t NODE_CNT_LOG    = 14; // active node
    constexpr std::size_t EDGE_CNT_LOG    = 15; // active edge
    constexpr std::size_t DENSITY         = 16; // edges/nodes

    // scan / rarity
    constexpr std::size_t SCAN_SCORE      = 18; // [0,1]
    constexpr std::size_t RARITY_SCORE    = 19; // >=0
}

// ======================= 1D Histogram =======================
struct Histogram1D {
    Histogram1D(std::size_t bins=32, double lo=0.0, double hi=1.0) noexcept
        : bins_(bins), lo_(lo), hi_(hi), counts_(bins, 0.0), total_(0.0) {}

    void add(double x) noexcept {
        if (!std::isfinite(x)) return;
        if (x < lo_) x = lo_;
        if (x > hi_) x = hi_;
        std::size_t b = (hi_ > lo_)
            ? std::min<std::size_t>(bins_ - 1, static_cast<std::size_t>((x - lo_) / (hi_ - lo_) * bins_))
            : 0u;
        counts_[b] += 1.0;
        total_     += 1.0;
    }

    arma::vec pdf(double eps=1e-12) const {
        arma::vec p(bins_, arma::fill::zeros);
        if (total_ <= 0.0) return p;
        for (std::size_t i = 0; i < bins_; ++i) p(i) = counts_[i] / total_;
        p += eps; p /= arma::accu(p);
        return p;
    }

    arma::vec counts() const {
        arma::vec v(bins_, arma::fill::zeros);
        for (std::size_t i = 0; i < bins_; ++i) v(i) = counts_[i];
        return v;
    }

    std::size_t bins() const noexcept { return bins_; }
    double lo() const noexcept { return lo_; }
    double hi() const noexcept { return hi_; }

    void print(std::ostream& os = std::cout) const {
        os << "Histogram1D(bins=" << bins_
           << ", lo=" << lo_
           << ", hi=" << hi_
           << ", total=" << total_
           << ")\n";
        double width = (hi_ - lo_) / bins_;
        for (std::size_t i = 0; i < bins_; ++i) {
            double bin_lo = lo_ + i * width;
            double bin_hi = bin_lo + width;
            os << "  [" << std::setw(10) << bin_lo << ", "
               << std::setw(10) << bin_hi << "): "
               << counts_[i] << "\n";
        }
    }

private:
    std::size_t         bins_;
    double              lo_, hi_;
    std::vector<double> counts_;
    double              total_;
};

struct SubgraphProfileConfig {
    std::size_t bins = 32;

    double bw_lo   = 0.0,  bw_hi   = 14.0;    // v(10)
    double pps_lo  = 0.0,  pps_hi  = 14.0;    // v(11)
    double pkt_lo  = 0.0,  pkt_hi  = 2000.0;  // v(12)
    double rar_lo  = 0.0,  rar_hi  = 10.0;    // v(19)
    double scn_lo  = 0.0,  scn_hi  = 1.0;     // v(18)

    double deg_lo  = 0.0,  deg_hi  = 14.0;    // 对数度
    double imb_lo  = 0.0,  imb_hi  = 14.0;    // 角色不平衡
};

class SubgraphProfiler {
private:
    SubgraphProfileConfig cfg_;

    // 分布（当前窗口）
    Histogram1D h_bw_, h_pps_, h_pkt_, h_rar_, h_scn_;
    Histogram1D h_srcdeg_, h_dstdeg_, h_imb_;
    std::unordered_map<int,std::size_t> proto_counts_;

    // 流量稳健统计
    double bytes_log_mean_ = 0.0, pkts_log_mean_ = 0.0, pkt_size_mean_ = 0.0;
    double dur_med_ = 0.0;
    double bw_p50_ = 0.0, bw_p95_ = 0.0, bw_cv_ = 0.0;
    double pps_cv_ = 0.0;
    double pkt_cv_ = 0.0;

    // 结构
    double src_deg_mean_ = 0.0, src_deg_iqr_ = 0.0;
    double dst_deg_mean_ = 0.0, dst_deg_iqr_ = 0.0;
    double role_imb_p95_ = 0.0;
    double edge_freq_log_mean_ = 0.0, density_mean_ = 0.0;

    // 稀有/扫描
    double rarity_mean_ = 0.0, rarity_p95_ = 0.0;
    double scan_mean_ = 0.0, scan_hi_frac_ = 0.0;

    // 协议/度分布
    arma::vec P_proto_, P_srcdeg_, P_dstdeg_;
    double proto_entropy_ = 0.0, proto_hhi_ = 0.0;
    double srcdeg_entropy_ = 0.0, dstdeg_entropy_ = 0.0;

    // ---------- helpers ----------
    static void push(std::vector<double>& v, double x) {
        if (!std::isfinite(x)) x = 0.0;
        v.push_back(x);
    }

    static double mean_(const std::vector<double>& xs) {
        if (xs.empty()) return 0.0;
        double s = std::accumulate(xs.begin(), xs.end(), 0.0);
        return s / xs.size();
    }

    static double fracAbove_(const std::vector<double>& xs, double thr) {
        if (xs.empty()) return 0.0;
        std::size_t cnt = 0;
        for (double x : xs) {
            if (std::isfinite(x) && x >= thr) ++cnt;
        }
        return static_cast<double>(cnt) / static_cast<double>(xs.size());
    }

    static double percentile_(std::vector<double> xs, double q) {
        if (xs.empty()) return 0.0;
        q = std::clamp(q, 0.0, 1.0);
        std::size_t k = static_cast<std::size_t>(std::floor(q * (xs.size() - 1)));
        std::nth_element(xs.begin(), xs.begin() + k, xs.end());
        return xs[k];
    }

    static double iqr_(std::vector<double> xs) {
        if (xs.size() < 2) return 0.0;
        double q75 = percentile_(xs, 0.75);
        double q25 = percentile_(xs, 0.25);
        return std::max(0.0, q75 - q25);
    }

    static double cv_(const std::vector<double>& xs) {
        if (xs.size() < 2) return 0.0;
        double m = mean_(xs);
        if (m == 0.0) return 0.0;
        double s2 = 0.0;
        for (double x : xs) { double d = x - m; s2 += d*d; }
        double sd = std::sqrt(s2 / (xs.size() - 1));
        return sd / std::max(1e-12, std::abs(m));
    }

    static void align_size(arma::vec& a, arma::vec& b) {
        arma::uword n = std::max(a.n_elem, b.n_elem);
        if (a.n_elem < n) { arma::vec t(n, arma::fill::zeros); t.head(a.n_elem) = a; a = std::move(t); }
        if (b.n_elem < n) { arma::vec t(n, arma::fill::zeros); t.head(b.n_elem) = b; b = std::move(t); }
    }

    static arma::vec to_prob(arma::vec p, double eps=1e-12) {
        if (p.n_elem == 0) return arma::vec(1, arma::fill::ones);
        p += eps;
        double s = arma::accu(p);
        if (!(s > 0.0)) s = 1.0;
        return p / s;
    }
    static double entropy_(const arma::vec& p, double eps=1e-12) {
        if (p.n_elem == 0) return 0.0;
        arma::vec q = p + eps; q /= arma::accu(q);
        return -arma::accu(q % arma::log(q));
    }

    static arma::vec protoPdf_(const std::unordered_map<int,std::size_t>& cnt, double eps=1e-12) {
        if (cnt.empty()) return arma::vec(1, arma::fill::ones);
        int maxk = 0; for (auto& kv : cnt) maxk = std::max(maxk, kv.first);
        arma::vec p(maxk + 1, arma::fill::zeros);
        double tot = 0.0;
        for (auto& kv : cnt) { p(kv.first) = static_cast<double>(kv.second); tot += kv.second; }
        if (tot <= 0.0) { p.fill(1.0); }
        p += eps; p /= arma::accu(p);
        return p;
    }

    static double hhi_(const arma::vec& p) {
        if (p.n_elem == 0) return 0.0;
        arma::vec q = p / std::max(1e-12, arma::accu(p));
        return arma::accu(q % q); // Herfindahl-Hirschman Index
    }

public:
    explicit SubgraphProfiler(const SubgraphProfileConfig& cfg)
        : cfg_(cfg),
          h_bw_ (cfg.bins, cfg.bw_lo,  cfg.bw_hi),
          h_pps_(cfg.bins, cfg.pps_lo, cfg.pps_hi),
          h_pkt_(cfg.bins, cfg.pkt_lo, cfg.pkt_hi),
          h_rar_(cfg.bins, cfg.rar_lo, cfg.rar_hi),
          h_scn_(cfg.bins, cfg.scn_lo, cfg.scn_hi),
          h_srcdeg_(cfg.bins, cfg.deg_lo, cfg.deg_hi),
          h_dstdeg_(cfg.bins, cfg.deg_lo, cfg.deg_hi),
          h_imb_   (cfg.bins, cfg.imb_lo, cfg.imb_hi) {}

    // B 阶段：用子图内全部样本构建统计量（仅子图自身）
    void buildFromSamples(std::shared_ptr<std::vector<arma::vec>> samples) {
        if (!samples || samples->empty()) return;

        // 收集标量序列（用于稳健统计）
        std::vector<double> bytes_log, pkts_log, dur_ms, bw_log, pps_log, pkt_size;
        std::vector<double> src_outdeg_log, dst_indeg_log, edge_freq_log, density;
        std::vector<double> role_imb, rarity, scan;

        bytes_log.reserve(samples->size());
        pkts_log .reserve(samples->size());
        dur_ms   .reserve(samples->size());
        bw_log   .reserve(samples->size());
        pps_log  .reserve(samples->size());
        pkt_size .reserve(samples->size());
        src_outdeg_log.reserve(samples->size());
        dst_indeg_log.reserve(samples->size());
        edge_freq_log .reserve(samples->size());
        density       .reserve(samples->size());
        role_imb.reserve(samples->size());
        rarity.reserve(samples->size());
        scan.reserve(samples->size());

        for (const auto& v : *samples) {
            if (v.n_elem < 20) continue;

            // ---------- 取值 ----------
            const double bw   = v(feat::BW_LOG);
            const double pps  = v(feat::PPS_LOG);
            const double pkt  = v(feat::PKT_SIZE);
            const double rar  = v(feat::RARITY_SCORE);
            const double scn  = std::clamp((double)v(feat::SCAN_SCORE), 0.0, 1.0);
            const double sdeg = v(feat::SRC_OUT_DEG_LOG);
            const double ddeg = v(feat::DST_IN_DEG_LOG);
            const double efr  = v(feat::EDGE_FREQ_LOG);
            const double dens = v(feat::DENSITY);

            double imb = 0.0;
            if (v.n_elem > std::max(feat::ROLE_IMB_SRCABS, feat::ROLE_IMB_DSTABS)) {
                imb = 0.5 * (std::max(0.0, (double)v(feat::ROLE_IMB_SRCABS)) +
                             std::max(0.0, (double)v(feat::ROLE_IMB_DSTABS)));
            }

            // ---------- 直方图 ----------
            h_bw_.add(bw); 
            h_pps_.add(pps); 
            h_pkt_.add(pkt);
            h_rar_.add(rar); 
            h_scn_.add(scn);
            h_srcdeg_.add(sdeg); 
            h_dstdeg_.add(ddeg); 
            h_imb_.add(imb);

            // ---------- 统计用序列 ----------
            bytes_log.push_back(v(feat::BYTES_LOG));
            pkts_log .push_back(v(feat::PKTS_LOG));
            dur_ms   .push_back(std::max(0.0, std::exp(v(feat::DUR_LOG)) - 1.0));
            bw_log   .push_back(bw);
            pps_log  .push_back(pps);
            pkt_size .push_back(pkt);

            src_outdeg_log.push_back(sdeg);
            dst_indeg_log.push_back(ddeg);
            edge_freq_log .push_back(efr);
            density       .push_back(dens);

            role_imb.push_back(imb);
            rarity.push_back(rar);
            scan.push_back(scn);

            // 协议计数
            ++proto_counts_[(int)v(feat::PROTO)];
        }

        // ---------- 基本流量/结构稳健统计 ----------
        bytes_log_mean_ = mean_(bytes_log);
        pkts_log_mean_  = mean_(pkts_log);
        pkt_size_mean_  = mean_(pkt_size);

        dur_med_   = percentile_(dur_ms, 0.50);

        bw_p50_    = percentile_(bw_log,  0.50);
        bw_p95_    = percentile_(bw_log,  0.95);
        bw_cv_     = cv_(bw_log);

        // 保留 CV，去掉 p50/p95
        pps_cv_    = cv_(pps_log);
        pkt_cv_    = cv_(pkt_size);

        src_deg_mean_ = mean_(src_outdeg_log);
        src_deg_iqr_  = iqr_(src_outdeg_log);

        dst_deg_mean_ = mean_(dst_indeg_log);
        dst_deg_iqr_  = iqr_(dst_indeg_log);

        role_imb_p95_  = percentile_(role_imb, 0.95);

        edge_freq_log_mean_ = mean_(edge_freq_log);
        density_mean_       = mean_(density);

        rarity_mean_ = mean_(rarity);
        rarity_p95_  = percentile_(rarity, 0.95);

        scan_mean_   = mean_(scan);
        scan_hi_frac_= fracAbove_(scan, 0.80); // 高扫描占比

        // ---------- 分布与集中度 ----------
        P_proto_       = protoPdf_(proto_counts_);
        proto_entropy_ = std::max(0.0, entropy_(P_proto_));
        proto_hhi_     = hhi_(P_proto_);       // 集中度（越大越集中）

        // 度分布形状
        P_srcdeg_      = h_srcdeg_.pdf();
        P_dstdeg_      = h_dstdeg_.pdf();
        srcdeg_entropy_= std::max(0.0, entropy_(P_srcdeg_));
        dstdeg_entropy_= std::max(0.0, entropy_(P_dstdeg_));
    }

    // 输出：子图自身 24 维特征向量（顺序与 featureNames() 对应）
    arma::vec finalizeSignature() const {
        std::vector<double> sig; sig.reserve(24);

        // 流量稳健统计
        push(sig, bytes_log_mean_);                // F1 bytes_log_mean
        push(sig, pkts_log_mean_);                 // F2 pkts_log_mean
        push(sig, dur_med_);                       // F3 dur_median
        push(sig, bw_p50_);                        // F4 bw_p50
        push(sig, bw_p95_);                        // F5 bw_p95
        push(sig, bw_cv_);                         // F6 bw_cv
        push(sig, pps_cv_);                        // F9 pps_cv
        push(sig, pkt_size_mean_);                 // F10 pkt_size_mean
        push(sig, pkt_cv_);                        // F11 pkt_size_cv

        // 结构角色
        push(sig, src_deg_mean_);                  // G1 src_outdeg_mean
        push(sig, src_deg_iqr_);                   // G2 src_outdeg_iqr
        push(sig, dst_deg_mean_);                  // G3 dst_indeg_mean
        push(sig, dst_deg_iqr_);                   // G4 dst_indeg_iqr
        push(sig, role_imb_p95_);                  // G6 role_imb_p95
        push(sig, edge_freq_log_mean_);            // G7 edge_freq_log_mean
        push(sig, density_mean_);                  // G8 density_mean

        // 稀有/扫描
        push(sig, rarity_mean_);                   // R1 rarity_mean
        push(sig, rarity_p95_);                    // R2 rarity_p95
        push(sig, scan_mean_);                     // S1 scan_mean
        push(sig, scan_hi_frac_);                  // S3 scan_high_frac(>=0.8)

        // 协议/度分布
        push(sig, proto_entropy_);                 // P1 proto_entropy
        push(sig, proto_hhi_);                     // P2 proto_hhi
        push(sig, srcdeg_entropy_);                // D1 srcdeg_entropy
        push(sig, dstdeg_entropy_);                // D2 dstdeg_entropy

        return arma::vec(sig);
    }

    // 便于调试对齐：给出特征名列表（与 finalizeSignature 顺序一致）
    static std::vector<std::string> featureNames() {
        return {
            "F1", "F2", "F3", "F4", "F5", "F6",
            "F9", "F10", "F11", "G1", "G2", "G3",
            "G4", "G6", "G7", "G8", "R1", "R2", 
            "S1", "S3", "P1", "P2", "D1", "D2"
        };
    }

    arma::vec dist_bw()   const { return h_bw_.pdf();  }
    arma::vec dist_pps()  const { return h_pps_.pdf(); }
    arma::vec dist_pkt()  const { return h_pkt_.pdf(); }
    arma::vec dist_rar()  const { return h_rar_.pdf(); }
    arma::vec dist_scn()  const { return h_scn_.pdf(); }

    const arma::vec& dist_srcdeg() const { return P_srcdeg_; }
    const arma::vec& dist_dstdeg() const { return P_dstdeg_; }
    const arma::vec& dist_proto()  const { return P_proto_;  }

    static double kl_div(const arma::vec& P, const arma::vec& Q, double eps=1e-12) {
        arma::vec p = P, q = Q;
        align_size(p, q);
        p = to_prob(p, eps);
        q = to_prob(q, eps);
        arma::vec ratio = p / arma::clamp(q, eps, std::numeric_limits<double>::infinity());
        return arma::accu(p % arma::log(ratio));
    }

    // JS(P||Q) = 0.5*KL(P||M)+0.5*KL(Q||M), M=(P+Q)/2
    static double js_div(const arma::vec& P, const arma::vec& Q, double eps=1e-12) {
        arma::vec p = P, q = Q;
        align_size(p, q);
        p = to_prob(p, eps);
        q = to_prob(q, eps);
        arma::vec m = 0.5 * (p + q);
        return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps);
    }

    // metric 版本（可用于距离）：sqrt(JS)
    static double js_distance(const arma::vec& P, const arma::vec& Q, double eps=1e-12) {
        return std::sqrt(std::max(0.0, js_div(P, Q, eps)));
    }
};