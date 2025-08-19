#pragma once
#include <armadillo>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

// ======================= Feature indices (from GraphFeatureExtractor) =======================
namespace feat {
    // node / edge role
    constexpr std::size_t SRC_OUT_DEG_LOG = 0;
    constexpr std::size_t DST_IN_DEG_LOG  = 2;
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

    // global context
    constexpr std::size_t DENSITY         = 16; // edges/nodes

    // scan / rarity
    constexpr std::size_t SCAN_SCORE      = 18; // [0,1]
    constexpr std::size_t RARITY_SCORE    = 19; // >=0
}

// ======================= 1D Histogram for batch accumulation =======================
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
        // smoothing to avoid zeros
        p += eps; p /= arma::accu(p);
        return p;
    }

    arma::vec counts() const {
        arma::vec v(bins_, arma::fill::zeros);
        for (std::size_t i = 0; i < bins_; ++i) v(i) = counts_[i];
        return v;
    }

    std::size_t bins() const noexcept { return bins_; }

    // KL divergence D_KL(P || Q) with eps-smoothing & length align
    static double klDivergence(const arma::vec& p, const arma::vec& q, double eps=1e-12) {
        if (p.n_elem == 0 || q.n_elem == 0) return 0.0;
        arma::vec P = p, Q = q;
        if (P.n_elem != Q.n_elem) {
            auto m = std::min(P.n_elem, Q.n_elem);
            P = P.head(m); Q = Q.head(m);
        }
        P += eps; P /= arma::accu(P);
        Q += eps; Q /= arma::accu(Q);
        arma::vec ratio = P / Q;
        arma::vec term  = P % arma::log(ratio);
        double val = arma::accu(term);
        return std::isfinite(val) ? val : 0.0;
    }

private:
    std::size_t        bins_;
    double             lo_, hi_;
    std::vector<double> counts_;
    double             total_;
};

// ======================= Config =======================
struct SubgraphProfileConfig {
    std::size_t bins = 32;
    double bw_lo   = 0.0,  bw_hi   = 14.0;    // v(10)
    double pps_lo  = 0.0,  pps_hi  = 14.0;    // v(11)
    double pkt_lo  = 0.0,  pkt_hi  = 2000.0;  // v(12)
    double rar_lo  = 0.0,  rar_hi  = 10.0;    // v(19)
    double scn_lo  = 0.0,  scn_hi  = 1.0;     // v(18)
};

// ======================= Batch-only Subgraph Profiler (KL-based) =======================
class SubgraphProfiler {
public:
    explicit SubgraphProfiler(const SubgraphProfileConfig& cfg)
        : cfg_(cfg),
          h_bw_ (cfg.bins, cfg.bw_lo,  cfg.bw_hi),
          h_pps_(cfg.bins, cfg.pps_lo, cfg.pps_hi),
          h_pkt_(cfg.bins, cfg.pkt_lo, cfg.pkt_hi),
          h_rar_(cfg.bins, cfg.rar_lo, cfg.rar_hi),
          h_scn_(cfg.bins, cfg.scn_lo, cfg.scn_hi) {}

    // ---------- Stage-B: build stats from all flows in the subgraph ----------
    void buildFromSamples(std::shared_ptr<std::vector<arma::vec>> samples) {
        clearWindow_();
        size_t sample_size = samples->size();

        std::vector<double> bytes_log, pkts_log, dur_ms, bw_log, pps_log, pkt_size;
        std::vector<double> src_outdeg_log, dst_indeg_log, edge_freq_log, density;
        std::vector<double> role_imb, rarity, scan;

        bytes_log.reserve(sample_size); 
        pkts_log.reserve(sample_size); 
        dur_ms.reserve(sample_size);
        bw_log.reserve(sample_size);    
        pps_log.reserve(sample_size);  
        pkt_size.reserve(sample_size);
        src_outdeg_log.reserve(sample_size); 
        dst_indeg_log.reserve(sample_size);
        edge_freq_log.reserve(sample_size);  
        density.reserve(sample_size);
        role_imb.reserve(sample_size); 
        rarity.reserve(sample_size); 
        scan.reserve(sample_size);

        for (const auto& v : *samples) {
            if (v.n_elem < 20) continue;
            flows_count_++;

            // histograms
            const double bw  = v(feat::BW_LOG);
            const double pps = v(feat::PPS_LOG);
            const double pkt = v(feat::PKT_SIZE);
            const double rar = v(feat::RARITY_SCORE);
            const double scn = std::clamp((double)v(feat::SCAN_SCORE), 0.0, 1.0);
            h_bw_.add(bw); h_pps_.add(pps); h_pkt_.add(pkt); h_rar_.add(rar); h_scn_.add(scn);

            // proto counts
            ++proto_counts_[(int)v(feat::PROTO)];

            // scalar series
            bytes_log.push_back(v(feat::BYTES_LOG));
            pkts_log .push_back(v(feat::PKTS_LOG));
            dur_ms   .push_back(std::max(0.0, std::exp(v(feat::DUR_LOG)) - 1.0));
            bw_log   .push_back(bw);
            pps_log  .push_back(pps);
            pkt_size .push_back(pkt);

            src_outdeg_log.push_back(v(feat::SRC_OUT_DEG_LOG));
            dst_indeg_log.push_back(v(feat::DST_IN_DEG_LOG));
            edge_freq_log .push_back(v(feat::EDGE_FREQ_LOG));
            density       .push_back(v(feat::DENSITY));

            // role imbalance: avg of src/dst abs diffs if available
            double imb = 0.0;
            if (v.n_elem > std::max(feat::ROLE_IMB_SRCABS, feat::ROLE_IMB_DSTABS)) {
                imb = 0.5 * (std::max(0.0, (double)v(feat::ROLE_IMB_SRCABS)) +
                             std::max(0.0, (double)v(feat::ROLE_IMB_DSTABS)));
            }
            role_imb.push_back(imb);

            rarity.push_back(rar);
            scan.push_back(scn);
        }

        // write cached scalars
        bytes_log_mean_ = mean_(bytes_log);
        pkts_log_mean_  = mean_(pkts_log);
        pkt_size_mean_  = mean_(pkt_size);

        dur_med_        = percentile_(dur_ms, 0.50);
        bw_p95_         = percentile_(bw_log,  0.95);
        pps_p95_        = percentile_(pps_log, 0.95);

        src_outdeg_log_mean_ = mean_(src_outdeg_log);
        dst_indeg_log_mean_  = mean_(dst_indeg_log);
        role_imb_mean_       = mean_(role_imb);
        edge_freq_log_mean_  = mean_(edge_freq_log);
        density_mean_        = mean_(density);

        rarity_mean_ = mean_(rarity);
        scan_mean_   = mean_(scan);
    }

    // ---------- Optional: set baselines for KL_self (if not set → KL_self=0) ----------
    void setBaselineBw  (const arma::vec& pdf){ base_bw_=pdf;   has_base_bw_=true;   }
    void setBaselinePps (const arma::vec& pdf){ base_pps_=pdf;  has_base_pps_=true;  }
    void setBaselinePkt (const arma::vec& pdf){ base_pkt_=pdf;  has_base_pkt_=true;  }
    void setBaselineRar (const arma::vec& pdf){ base_rar_=pdf;  has_base_rar_=true;  }
    void setBaselineScn (const arma::vec& pdf){ base_scn_=pdf;  has_base_scn_=true;  }
    void setBaselineProto(const arma::vec& pdf){ base_proto_=pdf;has_base_proto_=true;}

    // ---------- Global references for KL_global ----------
    void setGlobalRefBw  (const arma::vec& pdf){ g_bw_=pdf;   }
    void setGlobalRefPps (const arma::vec& pdf){ g_pps_=pdf;  }
    void setGlobalRefPkt (const arma::vec& pdf){ g_pkt_=pdf;  }
    void setGlobalRefRar (const arma::vec& pdf){ g_rar_=pdf;  }
    void setGlobalRefScn (const arma::vec& pdf){ g_scn_=pdf;  }
    void setGlobalRefProto(const std::unordered_map<int,std::size_t>& cnt){ g_proto_counts_=cnt; }

    // ---------- Signature: 18 core + 6 KL_self + (optional) 6 KL_global ----------
    arma::vec finalizeSignature(bool include_global=true) const {
        // pdfs from histograms
        const arma::vec P_bw  = h_bw_.pdf();
        const arma::vec P_pps = h_pps_.pdf();
        const arma::vec P_pkt = h_pkt_.pdf();
        const arma::vec P_rar = h_rar_.pdf();
        const arma::vec P_scn = h_scn_.pdf();
        const arma::vec P_pr  = protoPdf_(proto_counts_);

        // helpers (KL with safety)
        auto kl_or0 = [](const arma::vec& P, const arma::vec& baseline, bool has)->double {
            return has ? Histogram1D::klDivergence(P, baseline) : 0.0;
        };
        auto kl_if_nonempty = [](const arma::vec& P, const arma::vec& Q)->double {
            return (P.n_elem && Q.n_elem) ? Histogram1D::klDivergence(P, Q) : 0.0;
        };

        // KL_self
        const double KL_bw_self   = kl_or0(P_bw , base_bw_ , has_base_bw_);
        const double KL_pps_self  = kl_or0(P_pps, base_pps_, has_base_pps_);
        const double KL_pkt_self  = kl_or0(P_pkt, base_pkt_, has_base_pkt_);
        const double KL_rar_self  = kl_or0(P_rar, base_rar_, has_base_rar_);
        const double KL_scn_self  = kl_or0(P_scn, base_scn_, has_base_scn_);
        const double KL_pr_self   = kl_or0(P_pr , base_proto_, has_base_proto_);

        // KL_global
        double KL_bw_g=0, KL_pps_g=0, KL_pkt_g=0, KL_rar_g=0, KL_scn_g=0, KL_pr_g=0;
        if (include_global) {
            KL_bw_g  = kl_if_nonempty(P_bw , g_bw_);
            KL_pps_g = kl_if_nonempty(P_pps, g_pps_);
            KL_pkt_g = kl_if_nonempty(P_pkt, g_pkt_);
            KL_rar_g = kl_if_nonempty(P_rar, g_rar_);
            KL_scn_g = kl_if_nonempty(P_scn, g_scn_);
            if (!g_proto_counts_.empty()) {
                KL_pr_g = Histogram1D::klDivergence(P_pr, protoPdf_(g_proto_counts_));
            }
        }

        // assemble signature (same layout as before; slots now hold KL instead of JS)
        std::vector<double> sig; sig.reserve(24 + (include_global?6:0));

        // F1-F7 scale & activity
        sig.push_back(static_cast<double>(flows_count_)); // F1
        sig.push_back(bytes_log_mean_);                   // F2
        sig.push_back(pkts_log_mean_);                    // F3
        sig.push_back(dur_med_);                          // F4
        sig.push_back(bw_p95_);                           // F5
        sig.push_back(pps_p95_);                          // F6
        sig.push_back(pkt_size_mean_);                    // F7

        // G1-G5 structural roles
        sig.push_back(src_outdeg_log_mean_);              // G1
        sig.push_back(dst_indeg_log_mean_);               // G2
        sig.push_back(role_imb_mean_);                    // G3
        sig.push_back(edge_freq_log_mean_);               // G4
        sig.push_back(density_mean_);                     // G5

        // R1-R2 rarity
        sig.push_back(rarity_mean_);                      // R1
        sig.push_back(q95FromHist_(h_rar_));              // R2

        // S1-S2 scanning
        sig.push_back(scan_mean_);                        // S1
        sig.push_back(q95FromHist_(h_scn_));              // S2

        // P1-P2 protocol & diversity
        sig.push_back(entropy_(P_pr));                    // P1
        sig.push_back(0.0);                               // P2 (coef of variation) – optional to fill

        // KL_self (5 continuous + proto)
        sig.push_back(KL_bw_self);
        sig.push_back(KL_pps_self);
        sig.push_back(KL_pkt_self);
        sig.push_back(KL_rar_self);
        sig.push_back(KL_scn_self);
        sig.push_back(KL_pr_self);

        // KL_global (optional)
        if (include_global) {
            sig.push_back(KL_bw_g);
            sig.push_back(KL_pps_g);
            sig.push_back(KL_pkt_g);
            sig.push_back(KL_rar_g);
            sig.push_back(KL_scn_g);
            sig.push_back(KL_pr_g);
        }

        return arma::vec(sig);
    }

private:
    // ---------- helpers ----------
    static double mean_(const std::vector<double>& xs) {
        if (xs.empty()) return 0.0;
        double s = std::accumulate(xs.begin(), xs.end(), 0.0);
        return s / xs.size();
    }

    static double percentile_(std::vector<double> xs, double q) {
        if (xs.empty()) return 0.0;
        q = std::clamp(q, 0.0, 1.0);
        std::size_t k = static_cast<std::size_t>(std::floor(q * (xs.size() - 1)));
        std::nth_element(xs.begin(), xs.begin() + k, xs.end());
        return xs[k];
    }

    static double entropy(const arma::vec& p, double eps=1e-12) {
        if (p.n_elem == 0) return 0.0;
        arma::vec q = p + eps; q /= arma::accu(q);
        return -arma::accu(q % arma::log(q));
    }
    static double entropy_(const arma::vec& p) { return entropy(p); }

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

    static double q95FromHist_(const Histogram1D& h) {
        arma::vec c = h.counts();
        double tot = arma::accu(c);
        if (tot <= 0.0) return 0.0;
        arma::vec cc = arma::cumsum(c) / tot;
        for (std::size_t i = 0; i < cc.n_elem; ++i) if (cc(i) >= 0.95) {
            // 返回 bin 索引（或你也可以按需要映射回原值范围）
            return static_cast<double>(i);
        }
        return static_cast<double>(cc.n_elem - 1);
    }

    void clearWindow_() {
        h_bw_  = Histogram1D(cfg_.bins, cfg_.bw_lo,  cfg_.bw_hi);
        h_pps_ = Histogram1D(cfg_.bins, cfg_.pps_lo, cfg_.pps_hi);
        h_pkt_ = Histogram1D(cfg_.bins, cfg_.pkt_lo, cfg_.pkt_hi);
        h_rar_ = Histogram1D(cfg_.bins, cfg_.rar_lo, cfg_.rar_hi);
        h_scn_ = Histogram1D(cfg_.bins, cfg_.scn_lo, cfg_.scn_hi);

        proto_counts_.clear();

        flows_count_ = 0;
        bytes_log_mean_ = pkts_log_mean_ = pkt_size_mean_ = 0.0;
        dur_med_ = bw_p95_ = pps_p95_ = 0.0;
        src_outdeg_log_mean_ = dst_indeg_log_mean_ = role_imb_mean_ = 0.0;
        edge_freq_log_mean_ = density_mean_ = 0.0;
        rarity_mean_ = scan_mean_ = 0.0;
    }

private:
    SubgraphProfileConfig cfg_;

    // distributions built from the current window
    Histogram1D h_bw_, h_pps_, h_pkt_, h_rar_, h_scn_;
    std::unordered_map<int,std::size_t> proto_counts_;

    // cached scalars (batch only)
    std::size_t flows_count_ = 0;
    double bytes_log_mean_ = 0.0, pkts_log_mean_ = 0.0, pkt_size_mean_ = 0.0;
    double dur_med_ = 0.0, bw_p95_ = 0.0, pps_p95_ = 0.0;
    double src_outdeg_log_mean_ = 0.0, dst_indeg_log_mean_ = 0.0, role_imb_mean_ = 0.0;
    double edge_freq_log_mean_ = 0.0, density_mean_ = 0.0;
    double rarity_mean_ = 0.0, scan_mean_ = 0.0;

    // baselines for KL_self (optional)
    arma::vec base_bw_, base_pps_, base_pkt_, base_rar_, base_scn_, base_proto_;
    bool has_base_bw_ = false, has_base_pps_ = false, has_base_pkt_ = false;
    bool has_base_rar_ = false, has_base_scn_ = false, has_base_proto_ = false;

    // global references for KL_global (optional)
    arma::vec g_bw_, g_pps_, g_pkt_, g_rar_, g_scn_;
    std::unordered_map<int,std::size_t> g_proto_counts_;
};