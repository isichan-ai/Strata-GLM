// src/core/verify_glm5.cpp - the verify window for glm5-next (GLM-5.3-Flash), 2026-10-08.
//
// The same window as record_window (verify.cpp): T tokens, one captured graph, the expert doorbell / GPU plan / CPU
// pool / combine machinery of the Qwen path unchanged.  What differs per layer (glm-notes/配線図_decode経路):
//   mHC read (attention side; the previous layer's FFN write folded in)       fused_mhc_read_multi
//   KDA layers: q|k|v Q8_0 -> conv+SiLU+L2 -> f_a/f_b, beta, g_a/g_b -> KDA recurrence + RMSNorm x gamma x
//               sigmoid(z) -> attn_output                                     gdn_*_multi (kda), kda_gate_beta_multi
//   DSA layers: q_a + q_a_norm -> q_b -> wk_b per head (absorb); kv_a_mqa + kv_a_norm -> the fp16 latent cache;
//               the k-pool indexer (append, scores, top-k); MLA attention over the selection; wv_b per head; wo
//   mHC read (FFN side; the attention write folded in)
//   layers 0-2: the dense SwiGLU FFN (no experts: the doorbell only rings); the others: the sigmoid router with
//               its bias and x2.5, the shared expert (no gate), the routed experts through the doorbell
//   the last layer: the plain mHC write, the mean of the 4 streams, output_norm, the head.
// The commit re-runs the KDA recurrence of the kept tokens and re-appends their indexer keys after restoring the
// indexer tail it snapshotted (as the Qwen commit does).
#include "strata/core/verify.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/remote_stage.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/glm5_kda_chunk.hpp"
#include "strata/kernels/glm5_misc.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

// the MLA decode's per-head absorb products (wk_b[h] . q[h], wv_b[h] . lat[h]) in one launch each (2026-10-10:
// 2 x 64 launches of ~4.9 us a DSA layer -> 2; bitwise the per-head calls).  STRATA_GLM5_ABSORB_BATCH=0: head by head
static const bool absorb_batch = [] { const char* v = std::getenv("STRATA_GLM5_ABSORB_BATCH"); return !(v && v[0] == '0'); }();
// the projections that share an input (KDA: q, k, v, f_a, g_a, beta; DSA: q_a, kv_a, indexer k, gate) in one launch
// each for a one-token window (2026-10-10; bitwise the separate launches).  STRATA_GLM5_PROJ_BATCH=0: separate
static const bool proj_batch = [] { const char* v = std::getenv("STRATA_GLM5_PROJ_BATCH"); return !(v && v[0] == '0'); }();

struct Verifier::Glm5 {
    // geometry (from the GGUF)
    int64_t N = 0, H = 0, S = 0, C = 0, DI = 0;            // n_embd, KDA heads, head dim, conv channels, H*S
    int64_t NH = 0, QLR = 0, KVR = 0, QK = 0, VD = 0;      // MLA heads, q_lora_rank, kv_lora_rank (latent), q/v head dims
    int64_t IH = 0, ID = 0, NFF = 0, NFFD = 0, NE = 0, K = 0, max_cells = 0, max_blocks = 0, cap = 0;
    float rms_eps = 1e-5f, ln_eps = 1e-6f, hc_eps = 1e-6f, lb = -5.f, wscale = 1.f, clamp_exp = 0.f, clamp_sh = 0.f;
    int iters = 20, n_slices = 0;
    std::vector<int64_t> kda_index, dsa_index;            // per layer: its ordinal among its kind, or -1
    // per KDA layer
    float* convw = nullptr;          // [nK][C][4]
    float* gate = nullptr;           // [nK][MT][H][S]
    // per DSA layer
    uint8_t* latent = nullptr;       // [nD][max_cells] rows of lat_row bytes: fp16 [KVR], or q4_0 (STRATA_GLM5_KV)
    bool lat_q4 = true;
    size_t lat_row = 0;
    float* tail = nullptr;           // [nD][3][2*ID]
    uint16_t* pooled = nullptr;      // [nD][max_cells/4 + 1][ID] BF16 (2026-10-09)
    float* tail_snap = nullptr;      // [nD][3][2*ID]
    float* raw_k = nullptr;          // [nD][MT][ID]  the window's indexer keys / gates, for the commit's re-append
    float* raw_g = nullptr;
    int32_t* page_table = nullptr;   // identity, [max_cells / 4]
    // scratch [MT][...]
    float *q = nullptr, *k = nullptr, *v = nullptr, *fa = nullptr, *f = nullptr, *ga = nullptr, *bl = nullptr;
    float *qr = nullptr, *qb = nullptr, *qT = nullptr, *qaT = nullptr, *qa = nullptr, *kv = nullptr, *iq = nullptr,
          *iw = nullptr, *lat = nullptr, *latT = nullptr, *vbT = nullptr, *vb = nullptr, *mla_scratch = nullptr;
    float *part = nullptr, *ssp = nullptr, *postA = nullptr, *combA = nullptr, *postF = nullptr, *combF = nullptr;
    float *dg = nullptr, *du = nullptr, *hmean = nullptr;
    uint8_t* q8 = nullptr;           // q8_1 activations, the widest use (per-head wv_b: 64 x MT columns of 512)
    std::vector<void*> allocs;
    template <typename T> T* dev(size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T) + 256) != cudaSuccess) { cudaGetLastError(); allocs.push_back(nullptr); return nullptr; }
        cudaMemset(p, 0, n * sizeof(T) + 256);
        allocs.push_back(p);
        return (T*) p;
    }
};

namespace {
const WeightRef* wfind(const WeightTable& wt, int64_t l, const char* suffix, std::string& err) {
    const std::string name = l < 0 ? std::string(suffix) : "blk." + std::to_string(l) + "." + suffix;
    const WeightRef* r = wt.find(name);
    if (r == nullptr && err.empty()) err = "verify glm5: " + name + " is missing";
    return r;
}
// one Q8_0/K-quant native matrix on `ncols` columns of x (q8_1-quantized into `q8` first)
void mmvq(const WeightRef* w, const float* x, int n_in, float* y, int n_out, int ncols, uint8_t* q8, void* cs) {
    strata::kernels::native_quantize_q8_1(x, q8, n_in, ncols, cs);
    strata::kernels::native_mmvq(w->native_type, w->native_data, q8, y, n_in, n_out, ncols, cs);
}
}  // namespace

void Verifier::free_glm5() {
    if (!glm_) return;
    for (void* p : glm_->allocs) cudaFree(p);
    delete glm_;
    glm_ = nullptr;
}

bool Verifier::init_glm5(std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    glm_ = new Glm5();
    Glm5& m = *glm_;
    const int64_t MT = max_t_;
    m.N = g.n_embd; m.H = g.ssm_v_heads; m.S = g.ssm_state_size; m.C = g.ssm_conv_channels; m.DI = g.ssm_value_dim;
    m.NH = g.n_head; m.QLR = g.q_lora_rank; m.KVR = g.kv_lora_rank; m.QK = g.mla_key_dim; m.VD = g.mla_value_dim;
    m.IH = g.idx_q_heads; m.ID = g.idx_key_dim; m.NFF = g.n_ff; m.NFFD = g.n_ff_dense; m.NE = g.n_expert; m.K = ss.k;
    m.rms_eps = g.rms_eps; m.ln_eps = g.norm_eps; m.hc_eps = g.hc_eps; m.iters = (int) g.hc_sinkhorn_iters;
    m.lb = g.kda_gate_lower_bound; m.wscale = g.expert_weights_scale;
    m.clamp_exp = g.swiglu_clamp_exp.empty() ? 0.f : g.swiglu_clamp_exp.back();
    m.clamp_sh = g.swiglu_clamp_shared.empty() ? 0.f : g.swiglu_clamp_shared.back();
    m.max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    m.max_blocks = m.max_cells / 4 + 2;
    m.cap = cap_;
    m.n_slices = fused_mhc_slices(m.N);
    if (!fused_mhc_supported(m.N, g.hc, (2 + g.hc) * g.hc) || m.S != 128 || m.NH % 16 != 0 || m.KVR != 512 ||
        m.ID != 128 || m.IH != 32 || lb_ < 0 || le_ > g.n_layers || lb_ >= le_) {
        err = "verify glm5: unsupported geometry or layer range";
        return false;
    }
    // the layers by kind, and their ordinals
    m.kda_index.assign((size_t) g.n_layers, -1);
    m.dsa_index.assign((size_t) g.n_layers, -1);
    int64_t nK = 0, nD = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) (g.is_sparse_layer(l) ? m.dsa_index : m.kda_index)[(size_t) l] = g.is_sparse_layer(l) ? nD++ : nK++;
    if (nK != g.n_gdn_layers() || nD != g.n_qsa_layers()) { err = "verify glm5: layer kinds disagree with the geometry"; return false; }
    // per-layer device state
    m.convw = m.dev<float>((size_t) nK * m.C * 4);
    m.gate = m.dev<float>((size_t) nK * MT * m.H * m.S);
    { const char* kvv = std::getenv("STRATA_GLM5_KV"); m.lat_q4 = !(kvv && std::string(kvv) == "f16"); }   // default q4 (2026-10-08)
    m.lat_row = m.lat_q4 ? (size_t) kMlaQ4RowBytes : (size_t) m.KVR * 2;
    m.latent = m.dev<uint8_t>((size_t) nD * m.max_cells * m.lat_row);
    m.tail = m.dev<float>((size_t) nD * 3 * 2 * m.ID);
    m.pooled = m.dev<uint16_t>((size_t) nD * (m.max_cells / 4 + 1) * m.ID);
    m.tail_snap = m.dev<float>((size_t) nD * 3 * 2 * m.ID);
    m.raw_k = m.dev<float>((size_t) nD * MT * m.ID);
    m.raw_g = m.dev<float>((size_t) nD * MT * m.ID);
    m.page_table = m.dev<int32_t>((size_t) (m.max_cells / 4 + 1));
    // the KDA conv weights: GGUF ssm_conv1d_{q,k,v} [4, 1, DI] each (channel-major, tap fastest) -> q|k|v [C][4]
    for (int64_t l = lb_; l < le_; ++l) {   // this verifier's layers (a stage engine loads only its own)
        const int64_t gi = m.kda_index[(size_t) l];
        if (gi < 0) continue;
        const char* names[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"};
        for (int r = 0; r < 3; ++r) {
            const WeightRef* w = wfind(wt, l, names[r], err);
            if (!w || !w->data) { if (err.empty()) err = "verify glm5: a conv weight is not in the arena"; return false; }
            cudaMemcpy(m.convw + ((size_t) gi * m.C + (size_t) r * m.DI) * 4, w->data, (size_t) m.DI * 4 * 4,
                       cudaMemcpyDeviceToDevice);
        }
    }
    {
        std::vector<int32_t> pt((size_t) (m.max_cells / 4 + 1));
        for (size_t i = 0; i < pt.size(); ++i) pt[i] = (int32_t) i;
        cudaMemcpy(m.page_table, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice);
    }
    // scratch
    const size_t T = (size_t) MT;
    m.q = m.dev<float>(T * m.DI); m.k = m.dev<float>(T * m.DI); m.v = m.dev<float>(T * m.DI);
    m.fa = m.dev<float>(T * 128); m.f = m.dev<float>(T * m.DI); m.ga = m.dev<float>(T * 128); m.bl = m.dev<float>(T * m.H);
    m.qr = m.dev<float>(T * m.QLR); m.qb = m.dev<float>(T * m.NH * m.QK); m.qT = m.dev<float>(T * m.NH * m.QK);
    m.qaT = m.dev<float>(T * m.NH * m.KVR); m.qa = m.dev<float>(T * m.NH * m.KVR); m.kv = m.dev<float>(T * m.KVR);
    m.iq = m.dev<float>(T * m.IH * m.ID); m.iw = m.dev<float>(T * m.IH);
    m.lat = m.dev<float>(T * m.NH * m.KVR); m.latT = m.dev<float>(T * m.NH * m.KVR);
    m.vbT = m.dev<float>(T * m.NH * m.VD); m.vb = m.dev<float>(T * m.NH * m.VD);
    m.mla_scratch = m.dev<float>(T * mla_decode_attn_scratch_floats(m.cap, m.NH));
    m.part = m.dev<float>(T * m.n_slices * 24); m.ssp = m.dev<float>(T * m.n_slices);
    m.postA = m.dev<float>(T * 4); m.combA = m.dev<float>(T * 16); m.postF = m.dev<float>(T * 4); m.combF = m.dev<float>(T * 16);
    m.dg = m.dev<float>(T * m.NFFD); m.du = m.dev<float>(T * m.NFFD); m.hmean = m.dev<float>(T * m.N);
    m.q8 = m.dev<uint8_t>((size_t) (m.NH * T) * (m.KVR / 32) * 36 + (size_t) T * (m.NFFD / 32) * 36);
    for (void* p : m.allocs) if (p == nullptr) { err = "verify glm5: a device buffer does not fit"; return false; }
    // the small tensors are read as F32 straight from the arena: refuse a pack that stored them narrower
    {   // one of each kind in this verifier's layers (a layer split: a stage holds only its own)
        int64_t lk = -1, ld = -1, lm = -1;
        for (int64_t l = lb_; l < le_; ++l) {
            if (lk < 0 && m.kda_index[(size_t) l] >= 0) lk = l;
            if (ld < 0 && m.dsa_index[(size_t) l] >= 0) ld = l;
            if (lm < 0 && l >= g.n_dense_layers) lm = l;
        }
        std::vector<std::string> names = {"blk." + std::to_string(lb_) + ".attn_norm.weight",
                                          "blk." + std::to_string(lb_) + ".hc_attn_scale.weight"};
        if (lk >= 0) names.push_back("blk." + std::to_string(lk) + ".ssm_dt.bias");
        if (ld >= 0) names.push_back("blk." + std::to_string(ld) + ".indexer.proj.weight");
        if (lm >= 0) names.push_back("blk." + std::to_string(lm) + ".ffn_gate_inp.weight");
        if (le_ == g.n_layers) names.push_back("output_norm.weight");
        for (const std::string& nm : names) {
            const WeightRef* r = wt.find(nm);
            if (r == nullptr || r->data == nullptr || r->bytes < (uint64_t) r->elements * 4) {
                err = "verify glm5: " + nm + " is not an F32 tensor in the arena";
                return false;
            }
        }
    }
    std::fprintf(stderr, "strata verify: glm5 geometry N %lld H %lld S %lld C %lld DI %lld NH %lld QLR %lld KVR %lld QK %lld "
                         "VD %lld IH %lld ID %lld NFF %lld NFFD %lld NE %lld K %lld cap %lld eps %g/%g/%g lb %g ws %g clamp %g/%g\n",
                 (long long) m.N, (long long) m.H, (long long) m.S, (long long) m.C, (long long) m.DI, (long long) m.NH,
                 (long long) m.QLR, (long long) m.KVR, (long long) m.QK, (long long) m.VD, (long long) m.IH, (long long) m.ID,
                 (long long) m.NFF, (long long) m.NFFD, (long long) m.NE, (long long) m.K, (long long) m.cap, m.rms_eps,
                 m.ln_eps, m.hc_eps, m.lb, m.wscale, m.clamp_exp, m.clamp_sh);
    std::fprintf(stderr, "strata verify: glm5-next window: %lld KDA + %lld DSA layers, %lld dense, %lld-cell latent cache "
                         "(%.1f MiB, %s)\n", (long long) nK, (long long) nD, (long long) g.n_dense_layers,
                 (long long) m.max_cells, (double) nD * m.max_cells * m.lat_row / 1048576.0, m.lat_q4 ? "q4_0" : "fp16");
    return true;
}

bool Verifier::record_window_glm5(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    Glm5& m = *glm_;
    const int64_t N = m.N, HC = g.hc, K = m.K, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) m.S * m.H * m.S + (uint64_t) m.C * (g.ssm_d_conv - 1);
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const QsaShapes s = [&] { QsaShapes x = qsa_real_shapes(); x.n_head = g.n_head; x.n_head_kv = g.n_head_kv;
                              x.head_dim = g.head_dim; x.idx_n_head = g.idx_q_heads; x.idx_dim = g.idx_key_dim; return x; }();
    groups_[T] = 1;
    const int tb = 0, te = T, n = T;

    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    // a layer split (2026-10-08): a stage after the first starts from the hand-off - the streams R before the stage
    // before folded its last FFN in, that FFN's output and its post / comb - and its first hyper-connection read
    // folds them in exactly as the whole trunk's would (so the split is bit for bit the one engine)
    const int64_t HF = handoff_floats(g);
    auto hand_rows = [&](float* dst_base, const float* src_base, bool out) {
        for (int t = 0; t < T; ++t) {
            float* h = (float*) (out ? dst_base : src_base) + (size_t) t * HF;
            float* parts[4] = {Rt(t), bo_ + (size_t) t * N, m.postF + (size_t) t * HC, m.combF + (size_t) t * HC * HC};
            const int64_t at[4] = {0, HC * N, HC * N + N, HC * N + N + HC}, cnt[4] = {HC * N, N, HC, HC * HC};
            for (int i = 0; i < 4; ++i)
                out ? copy_from_mapped(h + at[i], parts[i], cnt[i], cs) : copy_from_mapped(parts[i], h + at[i], cnt[i], cs);
        }
    };
    if (lb_ > 0) {
        if (hand_in_ == nullptr) { err = "verify glm5: a stage without its hand-off"; return false; }
        hand_rows(nullptr, hand_in_, false);
    } else if (const NativeEmbed* ne = native_embed()) {
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else { err = "verify glm5: needs the native token embedding"; return false; }

    auto mhc_read = [&](int64_t l, const char* which, bool apply, const float* post_prev, const float* comb_prev,
                        float* post_out, float* comb_out, const char* norm, float* y) -> bool {
        const std::string w = std::string("hc_") + which + "_";
        const WeightRef *fn = wfind(wt, l, (w + "fn.weight").c_str(), err), *sc = wfind(wt, l, (w + "scale.weight").c_str(), err),
                        *bs = wfind(wt, l, (w + "base.weight").c_str(), err), *wn = wfind(wt, l, norm, err);
        if (!fn || !sc || !bs || !wn || !fn->native_data) { if (err.empty()) err = "verify glm5: mHC weights"; return false; }
        MhcMulti a;
        a.n_tok = n; a.n_embd = (int) N; a.n_slices = m.n_slices; a.apply = apply; a.iters = m.iters;
        a.rms_eps = m.rms_eps; a.hc_eps = m.hc_eps;
        for (int t = 0; t < n; ++t) { a.R[t] = Rt(t); a.R_out[t] = Rt(t); a.out_prev[t] = bo_ + (size_t) t * N; a.y[t] = y + (size_t) t * N; }
        a.post_prev = post_prev; a.comb_prev = comb_prev;
        a.fn = fn->native_data; a.scale = (const float*) sc->data; a.base = (const float*) bs->data; a.w_norm = (const float*) wn->data;
        a.post_out = post_out; a.comb_out = comb_out; a.part = m.part; a.ss_part = m.ssp;
        fused_mhc_read_multi(a, cs);
        return true;
    };

    // ---- pre(l): the attention block and the FFN's input, router and doorbell
    auto pre = [&](int64_t l) -> bool {
        float* xm = mixed_;   // the block input of the attention, then of the FFN
        if (!mhc_read(l, "attn", l > 0, m.postF, m.combF, m.postA, m.combA, "attn_norm.weight", xm)) return false;
        const int64_t gi = m.kda_index[(size_t) l], qi = m.dsa_index[(size_t) l];
        if (gi >= 0) {
            // ======================= KDA =======================
            const WeightRef *wq = wfind(wt, l, "attn_q.weight", err), *wk = wfind(wt, l, "attn_k.weight", err),
                            *wv = wfind(wt, l, "attn_v.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                            *wfa = wfind(wt, l, "ssm_f_a.weight", err), *wfb = wfind(wt, l, "ssm_f_b.weight", err),
                            *wga = wfind(wt, l, "ssm_g_a.weight", err), *wgb = wfind(wt, l, "ssm_g_b.weight", err),
                            *wbe = wfind(wt, l, "ssm_beta.weight", err), *wdt = wfind(wt, l, "ssm_dt.bias", err),
                            *wsa = wfind(wt, l, "ssm_a", err), *wnm = wfind(wt, l, "ssm_norm.weight", err);
            if (!wq || !wk || !wv || !wo || !wfa || !wfb || !wga || !wgb || !wbe || !wdt || !wsa || !wnm) return false;
            float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
            float* conv = state + (uint64_t) m.S * m.H * m.S;
            float* qkv = qkv_L_ + (size_t) gi * MT * m.C;
            float* hb = h_L_ + (size_t) gi * MT * m.C;
            float* gate = m.gate + (size_t) gi * MT * m.H * m.S;
            float* beta = beta_L_ + (size_t) gi * MT * m.H;
            native_quantize_q8_1(xm, xq_, (int) N, n, cs);   // x once for the six projections from it
            {
                // one token: the six in one launch (2026-10-10; STRATA_GLM5_PROJ_BATCH=0: six launches)
                const WeightRef* ws[6] = {wq, wk, wv, wfa, wga, wbe};
                const void* wp[6];
                float* yp[6] = {m.q, m.k, m.v, m.fa, m.ga, m.bl};
                const int no[6] = {(int) m.DI, (int) m.DI, (int) m.DI, 128, 128, (int) m.H};
                bool q8 = n == 1 && proj_batch;
                for (int i = 0; i < 6; ++i) { q8 = q8 && ws[i]->native_type == 8; wp[i] = ws[i]->native_data; }
                if (!(q8 && native_mmvq_table_q8_0(wp, no, 6, xq_, yp, (int) N, cs)))
                    for (int i = 0; i < 6; ++i)
                        native_mmvq(ws[i]->native_type, ws[i]->native_data, xq_, yp[i], (int) N, no[i], n, cs);
            }
            glm5_pack3(m.q, m.k, m.v, qkv, (int) m.DI, n, cs);
            gdn_conv_l2_multi(conv, qkv, m.convw + (size_t) gi * m.C * 4, hb, (int) m.C, (int) (2 * m.H), 1e-6f, n, cs, tb);
            mmvq(wfb, m.fa, 128, m.f, (int) m.DI, n, m.q8, cs);
            mmvq(wgb, m.ga, 128, z_, (int) m.DI, n, m.q8, cs);
            kda_gate_beta_multi(m.f, (const float*) wdt->data, (const float*) wsa->data, m.lb, m.bl, gate, beta,
                                (int) m.H, n, cs);
            gdn_step_norm_multi(state, hb, (int) m.C, gate, beta, z_, (const float*) wnm->data, m.rms_eps, y_, (int) m.H,
                                (int) m.H, te, nullptr, cs, tb, nullptr, true);
            mmvq(wo, y_, (int) m.DI, bo_, (int) N, n, m.q8, cs);
        } else {
            // ======================= DSA (MLA + k-pool indexer) =======================
            const WeightRef *wqa = wfind(wt, l, "attn_q_a.weight", err), *wqan = wfind(wt, l, "attn_q_a_norm.weight", err),
                            *wqb = wfind(wt, l, "attn_q_b.weight", err), *wkb = wfind(wt, l, "attn_k_b.weight", err),
                            *wvb = wfind(wt, l, "attn_v_b.weight", err), *wkv = wfind(wt, l, "attn_kv_a_mqa.weight", err),
                            *wkvn = wfind(wt, l, "attn_kv_a_norm.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                            *wik = wfind(wt, l, "indexer.attn_k.weight", err), *wiqb = wfind(wt, l, "indexer.attn_q_b.weight", err),
                            *wikn = wfind(wt, l, "indexer.k_norm.weight", err), *wikb = wfind(wt, l, "indexer.k_norm.bias", err),
                            *wip = wfind(wt, l, "indexer.proj.weight", err), *wape = wfind(wt, l, "indexer_compressor_ape.weight", err),
                            *wig = wfind(wt, l, "indexer_compressor_gate.weight", err);
            if (!wqa || !wqan || !wqb || !wkb || !wvb || !wkv || !wkvn || !wo || !wik || !wiqb || !wikn || !wikb || !wip ||
                !wape || !wig) return false;
            const int64_t QLR = m.QLR, NH = m.NH, QK = m.QK, KVR = m.KVR, VD = m.VD, ID = m.ID, IH = m.IH;
            uint8_t* lat_pool = m.latent + (size_t) qi * m.max_cells * m.lat_row;
            float* tail = m.tail + (size_t) qi * 3 * 2 * ID;
            uint16_t* pooled = m.pooled + (size_t) qi * (m.max_cells / 4 + 1) * ID;
            float* rk = m.raw_k + (size_t) qi * MT * ID;
            float* rg = m.raw_g + (size_t) qi * MT * ID;
            native_quantize_q8_1(xm, xq_, (int) N, n, cs);
            {
                // one token: the four projections of x in one launch (2026-10-10; STRATA_GLM5_PROJ_BATCH=0: four)
                const WeightRef* ws[4] = {wqa, wkv, wik, wig};
                const void* wp[4];
                float* yp[4] = {m.qr, m.kv, rk, rg};
                const int no[4] = {(int) QLR, (int) KVR, (int) ID, (int) ID};
                bool q8 = n == 1 && proj_batch;
                for (int i = 0; i < 4; ++i) { q8 = q8 && ws[i]->native_type == 8; wp[i] = ws[i]->native_data; }
                if (!(q8 && native_mmvq_table_q8_0(wp, no, 4, xq_, yp, (int) N, cs)))
                    for (int i = 0; i < 4; ++i)
                        native_mmvq(ws[i]->native_type, ws[i]->native_data, xq_, yp[i], (int) N, no[i], n, cs);
            }
            native_qsa_rms_norm_weighted(m.qr, (const float*) wqan->data, m.qr, (int) QLR, n, m.rms_eps, cs);
            native_qsa_rms_norm_weighted(m.kv, (const float*) wkvn->data, m.kv, (int) KVR, n, m.rms_eps, cs);
            if (m.lat_q4) glm5_latent_store_q4(m.kv, lat_pool, m.page_table, step_, kStepCount, 4, (int) KVR, n, cs);
            else glm5_latent_store(m.kv, (uint16_t*) lat_pool, m.page_table, step_, kStepCount, 4, (int) KVR, n, cs);
            // q: q_b, then the absorbed query q_abs[h] = wk_b[h] . q[h] (256 -> 512), head by head
            mmvq(wqb, m.qr, (int) QLR, m.qb, (int) (NH * QK), n, m.q8, cs);
            glm5_transpose_abd(m.qb, m.qT, n, (int) NH, (int) QK, cs);                      // [h][t][QK]
            native_quantize_q8_1(m.qT, m.q8, (int) (QK * NH * n), 1, cs);   // NH*n contiguous columns as one vector
            {
                const size_t row = (size_t) (QK / 32) * 34, qcol = (size_t) (QK / 32) * 36;
                // one token: the heads in one launch (2026-10-10; STRATA_GLM5_ABSORB_BATCH=0: head by head)
                if (!(n == 1 && absorb_batch && wkb->native_type == 8 &&
                      native_mmvq_batched_q8_0(wkb->native_data, (size_t) KVR * row, m.q8, qcol, m.qaT, (size_t) KVR,
                                               (int) QK, (int) KVR, (int) NH, cs)))
                    for (int64_t h = 0; h < NH; ++h)
                        native_mmvq(wkb->native_type, (const uint8_t*) wkb->native_data + (size_t) h * KVR * row,
                                    m.q8 + (size_t) h * n * qcol, m.qaT + (size_t) h * n * KVR, (int) QK, (int) KVR, n, cs);
            }
            glm5_transpose_abd(m.qaT, m.qa, (int) NH, n, (int) KVR, cs);                    // [t][h][KVR]
            // the indexer: keys and gates of these tokens into the pool (tail snapshot first, for the commit)
            copy_from_mapped(m.tail_snap + (size_t) qi * 3 * 2 * ID, tail, 3 * 2 * ID, cs);
            const Glm5KpoolBuffers kb{tail, pooled};
            for (int t = 0; t < n; ++t)
                glm5_kpool_append(rk + (size_t) t * ID, rg + (size_t) t * ID, step_ + (size_t) t * kStepCount + kStepPos,
                                  (const float*) wikn->data, (const float*) wikb->data, (const float*) wape->data, m.ln_eps,
                                  kb, m.max_cells, cs);
            mmvq(wiqb, m.qr, (int) QLR, m.iq, (int) (IH * ID), n, m.q8, cs);
            glm5_f32_gemv((const float*) wip->data, xm, m.iw, (int) N, (int) IH, n, 1.0f / std::sqrt((float) (ID * IH)), cs);
            glm5_kpool_scores(m.iq, m.iw, pooled, step_, n, max_blocks_, scores_, cs);
            qsa_block_topk(scores_, step_, n, max_blocks_, cap_, s, sel_, cs);
            // MLA over the selection (fp16 latents; scale 1/sqrt(q head dim) as llama.cpp), wv_b per head, then wo
            if (m.lat_q4) mla_decode_attn_batch_q4(m.qa, lat_pool, m.page_table, 4, sel_, step_, cap_, (int) NH, 1.0f / std::sqrt((float) QK), m.mla_scratch, m.lat, n, cs);
            else mla_decode_attn_batch(m.qa, (const uint16_t*) lat_pool, m.page_table, 4, sel_, step_, cap_, (int) NH, 1.0f / std::sqrt((float) QK),
                                  m.mla_scratch, m.lat, n, cs);
            glm5_transpose_abd(m.lat, m.latT, n, (int) NH, (int) KVR, cs);                  // [h][t][KVR]
            native_quantize_q8_1(m.latT, m.q8, (int) (KVR * NH * n), 1, cs);
            {
                const size_t row = (size_t) (KVR / 32) * 34, qcol = (size_t) (KVR / 32) * 36;
                if (!(n == 1 && absorb_batch && wvb->native_type == 8 &&
                      native_mmvq_batched_q8_0(wvb->native_data, (size_t) VD * row, m.q8, qcol, m.vbT, (size_t) VD,
                                               (int) KVR, (int) VD, (int) NH, cs)))
                    for (int64_t h = 0; h < NH; ++h)
                        native_mmvq(wvb->native_type, (const uint8_t*) wvb->native_data + (size_t) h * VD * row,
                                    m.q8 + (size_t) h * n * qcol, m.vbT + (size_t) h * n * VD, (int) KVR, (int) VD, n, cs);
            }
            glm5_transpose_abd(m.vbT, m.vb, (int) NH, n, (int) VD, cs);                     // [t][h][VD]
            mmvq(wo, m.vb, (int) (NH * VD), bo_, (int) N, n, m.q8, cs);
        }
        // ---- the FFN side
        if (!mhc_read(l, "ffn", true, m.postA, m.combA, m.postF, m.combF, "ffn_norm.weight", xm)) return false;
        if (l < g.n_dense_layers) {
            const WeightRef *wg = wfind(wt, l, "ffn_gate.weight", err), *wu = wfind(wt, l, "ffn_up.weight", err),
                            *wd = wfind(wt, l, "ffn_down.weight", err);
            if (!wg || !wu || !wd) return false;
            NativeSharedWeights nw;
            nw.gate_type = wg->native_type; nw.gate_data = wg->native_data; nw.up_type = wu->native_type; nw.up_data = wu->native_data;
            nw.down_type = wd->native_type; nw.down_data = wd->native_data; nw.q8_1 = m.q8;
            shared_expert_multi(n, xm, nullptr, nw, nullptr, m.dg, m.du, sh_g_, bo_, N, m.NFFD, cs, nullptr, 0, m.clamp_sh);
            doorbell_ring(m_seq_, cs);   // no experts: the host skips this layer's pool and raises its flags
            return true;
        }
        const WeightRef *wr = wfind(wt, l, "ffn_gate_inp.weight", err), *wb = wfind(wt, l, "exp_probs_b.bias", err),
                        *wsg = wfind(wt, l, "ffn_gate_shexp.weight", err), *wsu = wfind(wt, l, "ffn_up_shexp.weight", err),
                        *wsd = wfind(wt, l, "ffn_down_shexp.weight", err);
        if (!wr || !wb || !wsg || !wsu || !wsd) return false;
        glm5_f32_gemv((const float*) wr->data, xm, logits_, (int) N, (int) m.NE, n, 1.0f, cs);
        router_sigmoid_topk(logits_, (const float*) wb->data, n, (int) m.NE, (int) K, m.wscale, ids_, w_, cs);
        doorbell_publish(xm, ids_, w_, (int64_t) n * N, (int64_t) n * K, m_x_, m_ids_, m_w_, m_seq_, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data; nsw.up_type = wsu->native_type;
        nsw.up_data = wsu->native_data; nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data; nsw.q8_1 = m.q8;
        shared_expert_multi(n, xm, nullptr, nsw, nullptr, sh_gate_, sh_up_, sh_g_, shared_, N, m.NFF, cs, nullptr, 0, m.clamp_sh);
        quantize_q8_1_rows(xm, n, N, nat_xq_, cs);
        return true;
    };

    // ---- post(l): the routed experts (the Qwen path's machinery), the combine into bo_
    auto post = [&](int64_t l) -> bool {
        if (l < g.n_dense_layers) return true;   // the dense FFN's output is already in bo_
        const uint32_t ring = (uint32_t) (l - lb_ + 1);   // the host counts this stage's layers from 1 (run())
        const int64_t cap = (int64_t) n * K, capx = (int64_t) MT * K;
        int32_t* pl = plan_;
        wait_flag_ge(m_flagA_, ring, cs);
        copy_i32_from_mapped(pl, m_plan_, plan_i32_, cs);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const auto& f = lay.fmt[(size_t) l];
        NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        L.swiglu_limit = m.clamp_exp;
        native_expert_grouped(L, p_ptr, p_start, p_counts, p_dst, p_tok, cap, cap, nat_xq_, hit_scratch_, hit_out_, cs, 0);
        wait_flag_ge(m_flagB_, ring, cs);
        if (sink_.pcie_mode == 2) {
            fetch_blobs(p_ptr2, p_counts + 2, staging_, (int64_t) lay.blob_bytes(l), (int) kStagingBlobs, cs);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, staging_, (int64_t) lay.blob_bytes(l), cs);
        }
        native_expert_grouped(L, p_ptr2, p_start2, p_counts + 2, p_dst, p_tok, cap, cap, nat_xq_, hit_scratch_, hit_out_, cs,
                              kPcieGroupRows);
        wait_flag_ge(m_flag_, ring, cs);
        // STRATA_GLM5_SKIP_RENORM (expert_source.cpp): the host rewrote the window's weights in m_w_ (skipped 0, the rest
        // scaled) - the combine takes them back (a bit copy; unchanged rows copy as they were)
        static const bool car_renorm = [] { const char* v = std::getenv("STRATA_GLM5_SKIP_RENORM"); return v && std::atof(v) > 0.0; }();
        if (car_renorm) copy_i32_from_mapped((int32_t*) w_, (const int32_t*) m_w_, (int64_t) n * K, cs);
        copy_rows_from_mapped(parts_, m_ymiss_, (int64_t) n * K, N, p_dst, p_counts + 1, cs);
        moe_hit_add(parts_, hit_out_, p_dst, p_counts + 1, cap, N, cs);
        native_moe_combine_multi(parts_, w_, shared_, bo_, N, K, n, cs);
        return true;
    };

    try {
        for (int64_t l = lb_; l < le_; ++l) {
            if (!pre(l)) return false;
            if (!post(l)) return false;
        }
        if (le_ < g.n_layers) {   // an earlier stage: R, the last FFN's output and its post / comb go on, no head
            if (hand_out_ == nullptr) { err = "verify glm5: a stage without its hand-off"; return false; }
            hand_rows(hand_out_, nullptr, true);
            return true;
        }
        // the last layer's FFN write (nothing follows to fold it into)
        MhcMulti w;
        w.n_tok = n; w.n_embd = (int) N; w.apply = true; w.post_prev = m.postF; w.comb_prev = m.combF;
        for (int t = 0; t < n; ++t) { w.R[t] = Rt(t); w.R_out[t] = Rt(t); w.out_prev[t] = bo_ + (size_t) t * N; w.y[t] = head_mixed_ + (size_t) t * N; }
        fused_mhc_write_multi(w, cs);
        // the mean of the streams, output_norm, the head
        fused_mhc_exit_mean(w, cs);
        const WeightRef* on = wfind(wt, -1, "output_norm.weight", err);
        if (!on) return false;
        native_qsa_rms_norm_weighted(head_mixed_, (const float*) on->data, head_mixed_, (int) N, n, m.rms_eps, cs);
        if (head_ == nullptr || !head_->loaded()) { err = "verify glm5: needs the native head (--native)"; return false; }
        native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
        native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
    } catch (const std::exception& e) {
        err = std::string("verify glm5: ") + e.what();
        return false;
    }
    return true;
}

bool Verifier::capture_commit_glm5(std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    Glm5& m = *glm_;
    const int64_t MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) m.S * m.H * m.S + (uint64_t) m.C * (g.ssm_d_conv - 1);
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify glm5: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        for (int64_t l = lb_; l < le_ && ok; ++l) {   // this stage's layers
            const int64_t gi = m.kda_index[(size_t) l], qi = m.dsa_index[(size_t) l];
            if (gi >= 0) {
                const WeightRef* wnm = wfind(*wt_, l, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) m.S * m.H * m.S;
                gdn_conv_commit(conv, qkv_L_ + (size_t) gi * MT * m.C, (int) m.C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gi * MT * m.C, (int) m.C, m.gate + (size_t) gi * MT * m.H * m.S,
                                    beta_L_ + (size_t) gi * MT * m.H, z_, (const float*) wnm->data, m.rms_eps, y_dummy_,
                                    (int) m.H, (int) m.H, (int) MT, commit_, cs_, 0, nullptr, true);
            } else {
                const WeightRef *wikn = wfind(*wt_, l, "indexer.k_norm.weight", err),
                                *wikb = wfind(*wt_, l, "indexer.k_norm.bias", err),
                                *wape = wfind(*wt_, l, "indexer_compressor_ape.weight", err);
                if (!wikn || !wikb || !wape) { ok = false; break; }
                float* tail = m.tail + (size_t) qi * 3 * 2 * m.ID;
                copy_from_mapped(tail, m.tail_snap + (size_t) qi * 3 * 2 * m.ID, 3 * 2 * m.ID, cs_);
                const Glm5KpoolBuffers kb{tail, m.pooled + (size_t) qi * (m.max_cells / 4 + 1) * m.ID};
                // commit_ = [n_keep, n_keep - 1, pos_0 .. pos_{MT-1}]: rows past n_keep carry -1 and are ignored
                for (int64_t t = 0; t < MT; ++t)
                    glm5_kpool_append(m.raw_k + ((size_t) qi * MT + t) * m.ID, m.raw_g + ((size_t) qi * MT + t) * m.ID,
                                      commit_ + 2 + t, (const float*) wikn->data, (const float*) wikb->data,
                                      (const float*) wape->data, m.ln_eps, kb, m.max_cells, cs_);
            }
        }
    } catch (const std::exception& e) {
        err = std::string("verify glm5 commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify glm5: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

// ================================ SERVE: THE SEQUENCE STATE (2026-10-08) ================================
namespace {
struct StateSpans {
    float* gdn = nullptr; size_t gdn_n = 0;     // ss.gdn_state for the KDA layers (state + conv history each)
    float* tail = nullptr; size_t tail_n = 0;   // the indexer tails of the DSA layers
};
}  // namespace

static StateSpans glm5_spans(const ModelGeometry& g, SessionState& ss, float* tail, int64_t n_dsa, int64_t id) {
    StateSpans s;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    s.gdn = ss.gdn_state;
    s.gdn_n = (size_t) ss.gdn_alloc * gdn_floats;   // the session's own rows (a stage engine carves its layers only)
    s.tail = tail;
    s.tail_n = (size_t) n_dsa * 3 * 2 * id;
    return s;
}

bool Verifier::glm5_state_save(std::vector<float>& host, std::string& err) {
    if (glm_ == nullptr) { err = "glm5 state: no glm5 window"; return false; }
    const OnDevice on_device(device_);
    const StateSpans s = glm5_spans(*g_, *ss_, glm_->tail, g_->n_qsa_layers(), glm_->ID);
    host.resize(s.gdn_n + s.tail_n);
    cudaStreamSynchronize(cs_);
    if (cudaMemcpy(host.data(), s.gdn, s.gdn_n * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host.data() + s.gdn_n, s.tail, s.tail_n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "glm5 state: save failed";
        return false;
    }
    return true;
}

bool Verifier::glm5_state_restore(const std::vector<float>& host, std::string& err) {
    if (glm_ == nullptr) { err = "glm5 state: no glm5 window"; return false; }
    const OnDevice on_device(device_);
    const StateSpans s = glm5_spans(*g_, *ss_, glm_->tail, g_->n_qsa_layers(), glm_->ID);
    if (host.size() != s.gdn_n + s.tail_n) { err = "glm5 state: a snapshot of another size"; return false; }
    // 2026-10-13: on cs_ and waited for.  A cudaMemcpy from pageable memory may return before its DMA lands, and cs_
    // (non-blocking) does not wait for the legacy stream - the next prompt could read the previous conversation's
    // state (Project Maya 4f1f4d7 saw a reply to another conversation's question)
    cudaStreamSynchronize(cs_);
    if (cudaMemcpyAsync(s.gdn, host.data(), s.gdn_n * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess ||
        cudaMemcpyAsync(s.tail, host.data() + s.gdn_n, s.tail_n * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "glm5 state: restore failed";
        return false;
    }
    return true;
}

bool Verifier::glm5_state_zero(std::string& err) {
    // a layer split (2026-10-08): every stage starts empty - a stage in another engine through its link's Reset
    if (remote_ != nullptr) return remote_->reset(err) && (next_ == nullptr || next_->glm5_state_zero(err));
    if (glm_ == nullptr) { err = "glm5 state: no glm5 window"; return false; }
    const OnDevice on_device(device_);
    const StateSpans s = glm5_spans(*g_, *ss_, glm_->tail, g_->n_qsa_layers(), glm_->ID);
    cudaStreamSynchronize(cs_);
    // on cs_ and waited for (2026-10-13): a cudaMemset on the legacy stream can still be running when cs_ starts the
    // next prompt, which then reads the previous conversation's state
    if (cudaMemsetAsync(s.gdn, 0, s.gdn_n * 4, cs_) != cudaSuccess || cudaMemsetAsync(s.tail, 0, s.tail_n * 4, cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "glm5 state: zero failed";
        return false;
    }
    return next_ == nullptr || next_->glm5_state_zero(err);
}

// ================================ SERVE: SLOTS ON DISK (2026-10-09) ================================
// file = SlotHeader | tokens i32[n_cells] | KDA states + conv histories f32[gdn_n] | indexer tails f32[tail_n]
//        | per DSA layer: latents [n_cells][lat_row] | per DSA layer: indexer blocks f32[n_cells/4 + 1][ID]
// The latents and completed indexer blocks of cells below n_cells are never rewritten while the sequence grows, so
// cells [0, n_cells) plus the recurrent state at n_cells are the whole sequence state there.
namespace {
struct SlotHeader {
    char magic[8];
    uint64_t fingerprint;
    int64_t n_cells, n_dsa, lat_row, id, gdn_n, tail_n;
};
constexpr char kSlotMagic[8] = {'G', 'L', 'M', '5', 'S', 'L', 'T', '1'};
}  // namespace

bool Verifier::glm5_slot_write(const std::string& path, const int32_t* toks, int64_t n_cells, uint64_t fingerprint,
                               std::string& err) {
    if (glm_ == nullptr || remote_ != nullptr) { err = "glm5 slot: no local glm5 window"; return false; }
    const Glm5& m = *glm_;
    if (n_cells < 1 || n_cells > m.max_cells) { err = "glm5 slot: cells out of range"; return false; }
    const OnDevice on_device(device_);
    const int64_t nD = g_->n_qsa_layers();
    const StateSpans s = glm5_spans(*g_, *ss_, m.tail, nD, m.ID);
    SlotHeader h{};
    std::memcpy(h.magic, kSlotMagic, 8);
    h.fingerprint = fingerprint;
    h.n_cells = n_cells; h.n_dsa = nD; h.lat_row = (int64_t) m.lat_row; h.id = m.ID;
    h.gdn_n = (int64_t) s.gdn_n; h.tail_n = (int64_t) s.tail_n;
    const size_t lat_bytes = (size_t) n_cells * m.lat_row, pool_rows = (size_t) n_cells / 4 + 1;
    std::vector<uint8_t> buf(std::max<size_t>({s.gdn_n * 4, s.tail_n * 4, (size_t) nD * lat_bytes,
                                               (size_t) nD * pool_rows * m.ID * 2}));
    cudaStreamSynchronize(cs_);
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) { err = "glm5 slot: cannot write " + tmp; return false; }
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(toks, 4, (size_t) n_cells, f) == (size_t) n_cells;
    ok = ok && cudaMemcpy(buf.data(), s.gdn, s.gdn_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
         std::fwrite(buf.data(), 4, s.gdn_n, f) == s.gdn_n;
    ok = ok && cudaMemcpy(buf.data(), s.tail, s.tail_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
         std::fwrite(buf.data(), 4, s.tail_n, f) == s.tail_n;
    // the latents of cells [0, n_cells) of every DSA layer (each layer's rows start max_cells rows apart)
    ok = ok && cudaMemcpy2D(buf.data(), lat_bytes, m.latent, (size_t) m.max_cells * m.lat_row, lat_bytes, (size_t) nD,
                            cudaMemcpyDeviceToHost) == cudaSuccess &&
         std::fwrite(buf.data(), 1, (size_t) nD * lat_bytes, f) == (size_t) nD * lat_bytes;
    const size_t pool_bytes = pool_rows * (size_t) m.ID * 2, pool_pitch = (size_t) (m.max_cells / 4 + 1) * m.ID * 2;
    ok = ok && cudaMemcpy2D(buf.data(), pool_bytes, m.pooled, pool_pitch, pool_bytes, (size_t) nD,
                            cudaMemcpyDeviceToHost) == cudaSuccess &&
         std::fwrite(buf.data(), 1, (size_t) nD * pool_bytes, f) == (size_t) nD * pool_bytes;
    ok = std::fclose(f) == 0 && ok;
    if (!ok) { std::remove(tmp.c_str()); err = "glm5 slot: writing " + path + " failed"; return false; }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { err = "glm5 slot: cannot rename " + tmp; return false; }
    return true;
}

static bool glm5_slot_head(FILE* f, uint64_t fingerprint, SlotHeader& h, std::vector<int32_t>& toks) {
    if (std::fread(&h, sizeof h, 1, f) != 1 || std::memcmp(h.magic, kSlotMagic, 8) != 0 ||
        h.fingerprint != fingerprint || h.n_cells < 1 || h.n_cells > (int64_t) 1 << 22)
        return false;
    toks.resize((size_t) h.n_cells);
    return std::fread(toks.data(), 4, toks.size(), f) == toks.size();
}

bool Verifier::glm5_slot_peek(const std::string& path, uint64_t fingerprint, std::vector<int32_t>& toks) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    SlotHeader h{};
    const bool ok = glm5_slot_head(f, fingerprint, h, toks);
    std::fclose(f);
    return ok;
}

bool Verifier::glm5_slot_read(const std::string& path, uint64_t fingerprint, std::vector<int32_t>& toks,
                              std::string& err) {
    if (glm_ == nullptr || remote_ != nullptr) { err = "glm5 slot: no local glm5 window"; return false; }
    const Glm5& m = *glm_;
    const OnDevice on_device(device_);
    const int64_t nD = g_->n_qsa_layers();
    const StateSpans s = glm5_spans(*g_, *ss_, m.tail, nD, m.ID);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "glm5 slot: cannot read " + path; return false; }
    SlotHeader h{};
    if (!glm5_slot_head(f, fingerprint, h, toks) || h.n_dsa != nD || h.lat_row != (int64_t) m.lat_row ||
        h.id != m.ID || h.gdn_n != (int64_t) s.gdn_n || h.tail_n != (int64_t) s.tail_n || h.n_cells > m.max_cells) {
        std::fclose(f);
        err = "glm5 slot: " + path + " was written for another model or setting";
        return false;
    }
    const size_t lat_bytes = (size_t) h.n_cells * m.lat_row, pool_rows = (size_t) h.n_cells / 4 + 1;
    const size_t pool_bytes = pool_rows * (size_t) m.ID * 2, pool_pitch = (size_t) (m.max_cells / 4 + 1) * m.ID * 2;
    std::vector<uint8_t> buf(std::max<size_t>({s.gdn_n * 4, s.tail_n * 4, (size_t) nD * lat_bytes,
                                               (size_t) nD * pool_bytes}));
    cudaStreamSynchronize(cs_);
    // each copy on cs_ and waited for before buf is reused (2026-10-13; see glm5_state_restore)
    auto landed = [&](cudaError_t e) { return e == cudaSuccess && cudaStreamSynchronize(cs_) == cudaSuccess; };
    bool ok = std::fread(buf.data(), 4, s.gdn_n, f) == s.gdn_n &&
              landed(cudaMemcpyAsync(s.gdn, buf.data(), s.gdn_n * 4, cudaMemcpyHostToDevice, cs_));
    ok = ok && std::fread(buf.data(), 4, s.tail_n, f) == s.tail_n &&
         landed(cudaMemcpyAsync(s.tail, buf.data(), s.tail_n * 4, cudaMemcpyHostToDevice, cs_));
    ok = ok && std::fread(buf.data(), 1, (size_t) nD * lat_bytes, f) == (size_t) nD * lat_bytes &&
         landed(cudaMemcpy2DAsync(m.latent, (size_t) m.max_cells * m.lat_row, buf.data(), lat_bytes, lat_bytes,
                                  (size_t) nD, cudaMemcpyHostToDevice, cs_));
    ok = ok && std::fread(buf.data(), 1, (size_t) nD * pool_bytes, f) == (size_t) nD * pool_bytes &&
         landed(cudaMemcpy2DAsync(m.pooled, pool_pitch, buf.data(), pool_bytes, pool_bytes, (size_t) nD,
                                  cudaMemcpyHostToDevice, cs_));
    std::fclose(f);
    if (!ok) { err = "glm5 slot: reading " + path + " failed"; return false; }
    return true;
}

// ================================ THE PROMPT PATH (2026-10-08) ================================
//
// Layer by layer over a chunk of C tokens.  The attention runs in slices of 8 with the window's kernels and the
// commit's semantics (KDA state and conv history written per slice, latents and indexer keys appended), so it is
// the windows' arithmetic in the windows' order.  The FFN inputs of the whole chunk then meet their routed experts
// at once: each distinct expert of the layer is one group of native_expert_grouped, its blob a VRAM cache slot or
// copied from host memory into a small staging ring - every blob read once per chunk instead of once per window.
bool Verifier::prefill_glm5(const int32_t* tokens, int64_t n_all, int64_t pos0, int chunk, const Glm5Fetch& f,
                            std::string& err, int64_t l_begin, int64_t l_end, const float* hand_in, float* hand_out) {
    using namespace strata::kernels;
    if (glm_ == nullptr) { err = "verify glm5 prefill: no glm5 window"; return false; }
    const OnDevice on_device(device_);
    const auto t_start = std::chrono::steady_clock::now();
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    Glm5& m = *glm_;
    cudaStream_t cs = cs_;
    const int64_t N = m.N, HC = g.hc, K = m.K, MT = max_t_, NE = m.NE;
    const uint64_t gdn_floats = (uint64_t) m.S * m.H * m.S + (uint64_t) m.C * (g.ssm_d_conv - 1);
    const QsaShapes s = [&] { QsaShapes x = qsa_real_shapes(); x.n_head = g.n_head; x.n_head_kv = g.n_head_kv;
                              x.head_dim = g.head_dim; x.idx_n_head = g.idx_q_heads; x.idx_dim = g.idx_key_dim; return x; }();
    const auto& lay = strata::kernels::cpu::expert_layout();
    const int64_t C = std::max<int64_t>(8, chunk), CK = C * K;
    // the slice: the window buffers this borrows (qkv_L_, xq_, scores_, the mHC partials ...) hold max_t_ tokens
    const int64_t SL = std::min<int64_t>(8, MT);
    // staged blobs per grouped launch (each half of the ring); STRATA_GLM5_PF_BLOBS, 1..32, default 16 (24 could not page-lock its 723 MiB beside a 105 GiB RAM tier)
    const int B = [] { const char* v = std::getenv("STRATA_GLM5_PF_BLOBS"); const int b = v ? std::atoi(v) : 16; return b < 1 ? 1 : b > 32 ? 32 : b; }();
    // the staging slot: MMQ (below) steps from expert to expert in whole blocks of the layer's type, so a slot holds a
    // whole number of every MMQ layer type's blocks (UD-IQ4_XS: IQ3_S 110 B, IQ4_XS 136 B: lcm 7,480, x 16 B).  A plain
    // max_blob stride read the second expert of a launch from the middle of a block (the answer was "!!!!", 2026-10-09).
    const uint64_t max_blob = [&] {
        uint64_t a = 16;
        for (const auto& fm : lay.fmt)
            if (strata::prefill::mmq::supported(fm.gu_type) && strata::prefill::mmq::supported(fm.d_type)) {
                a = std::lcm(a, (uint64_t) strata::prefill::mmq::matrix_bytes(fm.gu_type, 1, 256));
                a = std::lcm(a, (uint64_t) strata::prefill::mmq::matrix_bytes(fm.d_type, 1, 256));
            }
        return (lay.max_blob + a - 1) / a * a;
    }();
    // STRATA_GLM5_PF_DIRECT=1 (opt-in, 2026-10-09): pinned blobs DMA'd straight from the RAM tier.  It removed the host
    // copies (80 -> 3 s summed over the threads, 4,400 tokens) but the prompt was not faster (46-57 vs 57-60 s): the GPU's
    // grouped experts then set the pace (the wait for the previous half grew by as much).  PCIe gives 41-49 GB/s here.
    static const bool pf_direct = [] { const char* v = std::getenv("STRATA_GLM5_PF_DIRECT"); return v && v[0] == '1'; }();
    const uint8_t* hsrc[32] = {};   // the pinned source of a blob that goes straight to the device
    // ---- buffers for one chunk, freed when the prompt is done
    std::vector<void*> allocs;
    auto dalloc = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes + 256) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        allocs.push_back(p);
        return p;
    };
    struct Free { std::vector<void*>& a; ~Free() { for (void* p : a) cudaFree(p); } } free_all{allocs};
    int32_t* tokc = (int32_t*) dalloc((size_t) C * 4);
    int32_t* stepc = (int32_t*) dalloc((size_t) C * kStepCount * 4);
    int32_t* nkeep = (int32_t*) dalloc(16 * 4);
    float* embc = (float*) dalloc((size_t) C * N * 4);
    float* Rc = (float*) dalloc((size_t) C * HC * N * 4);
    float* boc = (float*) dalloc((size_t) C * N * 4);
    float* xmc = (float*) dalloc((size_t) C * N * 4);
    float* shc = (float*) dalloc((size_t) C * N * 4);
    float* postA = (float*) dalloc((size_t) C * 4 * 4);
    float* combA = (float*) dalloc((size_t) C * 16 * 4);
    float* postF = (float*) dalloc((size_t) C * 4 * 4);
    float* combF = (float*) dalloc((size_t) C * 16 * 4);
    float* logc = (float*) dalloc((size_t) C * NE * 4);
    int32_t* idsc = (int32_t*) dalloc((size_t) CK * 4);
    float* wc = (float*) dalloc((size_t) CK * 4);
    float* parts = (float*) dalloc((size_t) CK * N * 4);
    uint8_t* q8c = (uint8_t*) dalloc((size_t) C * (N / 32) * 36);
    // two halves of the staging ring: the host fills one (threads copy the blobs into page-locked memory, one DMA
    // takes them to the device) while the GPU computes the experts of the other
    uint8_t* stage = (uint8_t*) dalloc((size_t) 2 * B * max_blob);
    unsigned long long* dptr = (unsigned long long*) dalloc((size_t) 2 * B * 8);
    const int64_t plan_n = (B + 1) + 2 * CK + 1;
    int32_t* dplan = (int32_t*) dalloc((size_t) 2 * plan_n * 4);
    // 2026-10-10: the experts' scratch for GE entries a launch (8,192; STRATA_GLM5_PF_LAUNCH_ROWS), not CK (32,768): with a
    // 4K chunk at 262K the card had 0 MiB free after the prompt's buffers and some long prompts ran ~3x slower (WDDM
    // paging); 880 -> ~220 MiB.  The launches' entries are capped to GE (more launches, the same arithmetic).
    static const int64_t ge_env = [] { const char* v = std::getenv("STRATA_GLM5_PF_LAUNCH_ROWS"); return v ? std::atoll(v) : 8192; }();
    const int64_t GE = std::max<int64_t>(std::min<int64_t>(CK, ge_env), C);
    const size_t gscratch_bytes = [&] {
        namespace mq_ = strata::prefill::mmq;
        const size_t nat = native_expert_scratch_bytes(GE, g.n_ff);
        const size_t mm = (((size_t) GE * 2 * g.n_ff * 4 + 255) & ~(size_t) 255) + ((mq_::q8_bytes(GE, N) + 255) & ~(size_t) 255) +
                          ((mq_::q8_bytes(GE, g.n_ff) + 255) & ~(size_t) 255) + (size_t) GE * 4;
        return std::max(nat, mm);
    }();
    void* gscratch = dalloc(gscratch_bytes);
    if (!tokc || !stepc || !nkeep || !embc || !Rc || !boc || !xmc || !shc || !postA || !combA || !postF || !combF || !logc ||
        !idsc || !wc || !parts || !q8c || !stage || !dptr || !dplan || !gscratch) {
        err = "verify glm5 prefill: the chunk buffers (" + std::to_string(C) + " tokens) do not fit in VRAM";
        return false;
    }
    // 2026-10-09 (the prompt path rewrite, stage 1): the MLA layers' wk_b / wv_b per head as ONE strided-batched F32
    // GEMM a slice instead of 2 x n_head native_mmvq launches (half of the prompt path's ~1.8M launches a 4K chunk).
    // The layer's two Q8_0 tensors are dequantized once per chunk into F32 scratch.  STRATA_GLM5_PF_BLAS=0: as before.
    static const bool pf_blas_on = [] { const char* v = std::getenv("STRATA_GLM5_PF_BLAS"); return !(v && v[0] == '0'); }();
    static cublasHandle_t pf_blas = nullptr;
    float* kb_f = nullptr;   // [NH][KVR][QK]
    float* vb_f = nullptr;   // [NH][VD][KVR]
    int64_t kbvb_layer = -1;
    // The scratch is CARVED from `parts` (the routed experts' outputs, C*K*N floats = 512 MiB for a 4K chunk): parts is
    // written and read only inside a layer's FFN (moe), and the attention side of the next layer runs after that on the
    // same stream.  A cudaMalloc here would not fit: with a 4K chunk at 262K context the card has 0 MiB left, and WDDM
    // lets the allocation succeed past the card and pages it to system memory, silently - a prompt then took 217-304 s
    // instead of ~6 (2026-10-09).
    static const bool pf_kda_on = [] { const char* v = std::getenv("STRATA_GLM5_PF_KDA"); return !(v && v[0] == '0'); }();
    const int64_t KSB = std::min<int64_t>(256, C);   // stage 2: tokens a projection GEMM
    const int64_t WSCR = (int64_t) 4 << 20;          // stage 2: dequantized weight rows, floats (16 MB)
    const size_t kBlasWs = (size_t) 8 << 20;         // cuBLAS's workspace, bytes
    float *kq = nullptr, *kk = nullptr, *kv_ = nullptr, *kf = nullptr, *kz = nullptr, *ky = nullptr, *kfa = nullptr,
          *kga = nullptr, *kbl = nullptr, *wscr = nullptr;
    // stage 3 (2026-10-09): the KDA recurrence in Infernix's chunked form over the sub-batch (glm5_kda_chunk.hpp) instead
    // of slices of <= 8 tokens.  STRATA_GLM5_PF_KDA_CHUNK=0 turns it off; STRATA_GLM5_PF_KDA_CHECK=1 also runs the slice
    // path on the same inputs and prints, per layer, how far the outputs and the state are apart.
    static const bool pf_kda_chunk = [] { const char* v = std::getenv("STRATA_GLM5_PF_KDA_CHUNK"); return !(v && v[0] == '0'); }();
    static const bool pf_kda_check = [] { const char* v = std::getenv("STRATA_GLM5_PF_KDA_CHECK"); return v && v[0] == '1'; }();
    float *kh = nullptr, *kgate = nullptr, *kbeta = nullptr, *kws = nullptr;
    uint16_t* kv16 = nullptr;
    float *ck_state = nullptr, *ck_state_ref = nullptr, *ck_conv = nullptr, *ck_y = nullptr;
    // 2026-10-09: the mHC reads over a whole chunk (fused_mhc_read_batch) instead of slices of <= 8 tokens - their
    // partial dots at the END of `parts` (the scratch below is carved before it).  STRATA_GLM5_PF_MHC=0: per slice.
    static const bool pf_mhc_on = [] { const char* v = std::getenv("STRATA_GLM5_PF_MHC"); return !(v && v[0] == '0'); }();
    float* parts_end = parts + (size_t) CK * N;
    float *mhc_part = nullptr, *mhc_ss = nullptr;
    // 2026-10-09: the MLA/DSA layers in sub-batches of mla_sb tokens (projections through cuBLAS, the indexer and the
    // attention kernels given the whole sub-batch: each reads only blocks / cells at or before its own token's step, so
    // storing the sub-batch's latents and keys first changes nothing).  With F32 cuBLAS projections it took the slices'
    // time (4,400 tokens: attention 7.5-7.6 s either way); with the MMQ projections (gemm_q8) it is the faster path:
    // 12,795 tokens, attention 16.1 -> 11.8 s, the prompt 56.3 -> 48.9 s.  STRATA_GLM5_PF_MLA=0: per slice.
    static const bool pf_mla_on = [] { const char* v = std::getenv("STRATA_GLM5_PF_MLA"); return !(v && v[0] == '0'); }();
    int64_t mla_sb = 0, mla_ab = 0;
    size_t mla_per_tok = 0;
    float *ma_vb = nullptr, *ma_latT = nullptr;
    // 2026-10-10: the indexer's scores of an attention sub-batch as S = Q . K^T (cublasGemmEx, BF16 in, F32 out) a tile
    // of kScoreTile blocks at a time, then glm5_kpool_scores_reduce - the warp-per-block kernel's work grows with the
    // context squared (31,883 tokens: 5.1 s).  Q rounds to BF16 (the keys are BF16).  STRATA_GLM5_PF_SCORE_GEMM=0: the kernel
    constexpr int64_t kScoreTile = 8192;
    float* ma_S = nullptr;
    uint16_t* ma_q16 = nullptr;
    // the indexer's scores / selection bounded by the sub-batch's filled blocks instead of the context's capacity (65,538
    // at 262K); the same ids.  STRATA_GLM5_PF_ACTIVE=0: the capacity
    static const bool pf_active = [] { const char* v = std::getenv("STRATA_GLM5_PF_ACTIVE"); return !(v && v[0] == '0'); }();
    float *ma_qr = nullptr, *ma_kv = nullptr, *ma_rk = nullptr, *ma_rg = nullptr, *ma_qb = nullptr, *ma_qT = nullptr,
          *ma_qaT = nullptr, *ma_qa = nullptr, *ma_lat = nullptr, *ma_iq = nullptr, *ma_iw = nullptr, *ma_sc = nullptr,
          *ma_scr = nullptr;
    int32_t* ma_sel = nullptr;
    {
        const size_t np = (size_t) C * m.n_slices * 24, ns = (size_t) C * m.n_slices;
        const size_t need = (np + ns + 63) & ~(size_t) 63;
        if (pf_mhc_on && need < (size_t) CK * N / 2) {
            mhc_part = parts_end - need;
            mhc_ss = mhc_part + np;
            parts_end = mhc_part;
        }
    }
    if (pf_blas_on) {
        if (pf_blas == nullptr && cublasCreate(&pf_blas) != CUBLAS_STATUS_SUCCESS) pf_blas = nullptr;
        if (pf_blas != nullptr) {
            float* p = parts;
            float* const end = parts_end;
            auto take = [&](size_t n) -> float* { float* r = p; p += (n + 63) & ~(size_t) 63; return p <= end ? r : nullptr; };
            float* ws = take(kBlasWs / 4);
            kb_f = take((size_t) m.NH * m.KVR * m.QK);
            vb_f = take((size_t) m.NH * m.VD * m.KVR);
            if (ws == nullptr || kb_f == nullptr || vb_f == nullptr) {
                kb_f = vb_f = nullptr;   // a small chunk: the per-head launches
            } else {
                cublasSetStream(pf_blas, cs);
                cublasSetWorkspace(pf_blas, ws, kBlasWs);
                if (pf_kda_on) {
                    const size_t di = (size_t) KSB * m.DI;
                    kq = take(di); kk = take(di); kv_ = take(di); kf = take(di); kz = take(di); ky = take(di);
                    kfa = take((size_t) KSB * 128); kga = take((size_t) KSB * 128); kbl = take((size_t) KSB * m.H);
                    wscr = take((size_t) WSCR);
                    if (!kq || !kk || !kv_ || !kf || !kz || !ky || !kfa || !kga || !kbl) wscr = nullptr;
                    if (wscr != nullptr && pf_kda_chunk) {   // stage 3: the chunked recurrence's buffers
                        kh = take((size_t) KSB * m.C);
                        kgate = take(di);
                        kbeta = take((size_t) KSB * m.H);
                        kv16 = (uint16_t*) take(di / 2 + 1);
                        kws = take(glm5_kda_chunk_workspace_bytes((int) m.H, (int) KSB) / 4 + 1);
                        if (pf_kda_check) {
                            const size_t st_n = (size_t) m.S * m.H * m.S, cv_n = (size_t) m.C * 3;
                            ck_state = take(st_n); ck_state_ref = take(st_n); ck_conv = take(cv_n); ck_y = take(di);
                        }
                        if (!kh || !kgate || !kbeta || !kv16 || !kws ||
                            (pf_kda_check && (!ck_state || !ck_state_ref || !ck_conv || !ck_y)))
                            kh = nullptr;   // no room: the slice recurrence
                    }
                    if (wscr != nullptr && pf_mla_on && m.VD <= m.KVR) {
                        // the MLA layers: projections over groups of mla_sb tokens (the weights dequantized once per
                        // group), the indexer's scores / selection and the attention over sub-batches of mla_ab inside
                        // (their scratch is ~4 MB a token); the largest pair that fits
                        const size_t per_g = (size_t) m.QLR + m.KVR + 2 * m.ID + 2 * m.NH * m.QK + 2 * m.NH * m.KVR + m.IH * m.ID +
                                             m.IH + m.NH * m.VD;
                        const size_t per_a = (size_t) max_blocks_ + cap_ + mla_decode_attn_scratch_floats(cap_, m.NH) + 2 * m.NH * m.KVR;
                        float* const p0 = p;
                        for (int64_t g = std::min<int64_t>(256, C); g >= 16 && mla_sb == 0; g /= 2)
                            for (int64_t a = std::min<int64_t>(64, g); a >= 8 && mla_sb == 0; a /= 2) {
                                p = p0;
                                ma_qr = take(g * m.QLR); ma_kv = take(g * m.KVR); ma_rk = take(g * m.ID); ma_rg = take(g * m.ID);
                                ma_qb = take(g * m.NH * m.QK); ma_qT = take(g * m.NH * m.QK);
                                ma_qaT = take(g * m.NH * m.KVR); ma_qa = take(g * m.NH * m.KVR);
                                ma_iq = take(g * m.IH * m.ID); ma_iw = take(g * m.IH); ma_vb = take(g * m.NH * m.VD);
                                ma_lat = take(a * m.NH * m.KVR); ma_latT = take(a * m.NH * m.KVR); ma_sc = take(a * max_blocks_);
                                ma_sel = (int32_t*) take(a * cap_); ma_scr = take(a * mla_decode_attn_scratch_floats(cap_, m.NH));
                                if (ma_scr != nullptr && ma_sel && ma_sc && ma_latT && ma_lat && ma_vb && ma_iw && ma_iq && ma_qa &&
                                    ma_qaT && ma_qT && ma_qb && ma_rg && ma_rk && ma_kv && ma_qr) {
                                    mla_sb = g;
                                    mla_ab = a;
                                }
                            }
                        if (mla_sb > 0 && m.IH == 32 && m.ID == 128) {   // the indexer's scores as a GEMM (optional room)
                            ma_S = take((size_t) mla_ab * 32 * kScoreTile);
                            ma_q16 = (uint16_t*) take((size_t) mla_ab * 32 * 128 / 2);
                            if (!ma_S || !ma_q16) ma_S = nullptr;
                        }
                        mla_per_tok = per_g + per_a;
                    }
                }
            }
        }
    }
    const bool kda_batch = wscr != nullptr;
    if (std::getenv("STRATA_GLM5_PF_PROF") != nullptr) {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "strata glm5 prefill: MLA absorb %s, KDA projections %s, MLA layers %lld tokens at a time, "
                     "their attention %lld (%zu KiB a token; cap %lld, blocks %lld) (%zu MiB free after the scratch)\n",
                     kb_f ? "batched (cuBLAS)" : "per head", kda_batch ? "batched (cuBLAS)" : "per slice",
                     (long long) (mla_sb > 0 ? mla_sb : SL), (long long) (mla_sb > 0 ? mla_ab : SL), mla_per_tok * 4 >> 10,
                     (long long) cap_, (long long) max_blocks_, fr >> 20);
    }
    uint8_t* pin = nullptr;
    const size_t pin_blobs = (size_t) 2 * B * max_blob;
    const size_t pin_bytes = pin_blobs + (size_t) 2 * plan_n * 4 + 64 + (size_t) 2 * B * 8;
    if (cudaHostAlloc((void**) &pin, pin_bytes, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        err = "verify glm5 prefill: no page-locked staging (" + std::to_string(pin_bytes >> 20) + " MiB)";
        return false;
    }
    int32_t* pplan = (int32_t*) (pin + pin_blobs);                                        // [2][plan_n]
    unsigned long long* pptr = (unsigned long long*) (pin + ((pin_blobs + (size_t) 2 * plan_n * 4 + 63) & ~(size_t) 63));
    cudaStream_t copy_s = nullptr;
    cudaEvent_t ev_copied[2] = {}, ev_done[2] = {};
    cudaStreamCreateWithFlags(&copy_s, cudaStreamNonBlocking);
    for (int h = 0; h < 2; ++h) {
        cudaEventCreateWithFlags(&ev_copied[h], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_done[h], cudaEventDisableTiming);
        cudaEventRecord(ev_copied[h], copy_s);
        cudaEventRecord(ev_done[h], cs);
    }
    struct Pinned {
        uint8_t* p; cudaStream_t s; cudaEvent_t* a; cudaEvent_t* b;
        ~Pinned() {
            if (s) cudaStreamSynchronize(s);
            for (int h = 0; h < 2; ++h) { if (a[h]) cudaEventDestroy(a[h]); if (b[h]) cudaEventDestroy(b[h]); }
            if (s) cudaStreamDestroy(s);
            if (p) cudaFreeHost(p);
        }
    } pinned_guard{pin, copy_s, ev_copied, ev_done};
    {
        int32_t k9[16];
        for (int i = 0; i < 16; ++i) k9[i] = i;
        cudaMemcpy(nkeep, k9, sizeof k9, cudaMemcpyHostToDevice);
    }
    std::vector<int32_t> h_steps((size_t) C * kStepCount), h_ids((size_t) CK), h_plan((size_t) plan_n);
    std::vector<unsigned long long> h_ptr(B);
    std::vector<std::vector<int32_t>> by((size_t) NE);
    std::vector<int32_t> used;

    auto Rt = [&](int64_t t) { return Rc + (size_t) t * HC * N; };
    auto mhc_read = [&](int64_t l, const char* which, bool apply, int64_t s0, int n, const char* norm) -> bool {
        const std::string w = std::string("hc_") + which + "_";
        const WeightRef *fn = wfind(wt, l, (w + "fn.weight").c_str(), err), *sc = wfind(wt, l, (w + "scale.weight").c_str(), err),
                        *bs = wfind(wt, l, (w + "base.weight").c_str(), err), *wn = wfind(wt, l, norm, err);
        if (!fn || !sc || !bs || !wn || !fn->native_data) { if (err.empty()) err = "verify glm5 prefill: mHC weights"; return false; }
        const bool attn = which[0] == 'a';
        MhcMulti a;
        a.n_tok = n; a.n_embd = (int) N; a.n_slices = m.n_slices; a.apply = apply; a.iters = m.iters;
        a.rms_eps = m.rms_eps; a.hc_eps = m.hc_eps;
        for (int t = 0; t < n; ++t) {
            a.R[t] = Rt(s0 + t); a.R_out[t] = Rt(s0 + t); a.out_prev[t] = boc + (size_t) (s0 + t) * N;
            a.y[t] = xmc + (size_t) (s0 + t) * N;
        }
        a.post_prev = (attn ? postF : postA) + s0 * 4; a.comb_prev = (attn ? combF : combA) + s0 * 16;
        a.post_out = (attn ? postA : postF) + s0 * 4; a.comb_out = (attn ? combA : combF) + s0 * 16;
        a.fn = fn->native_data; a.scale = (const float*) sc->data; a.base = (const float*) bs->data; a.w_norm = (const float*) wn->data;
        a.part = m.part; a.ss_part = m.ssp;
        fused_mhc_read_multi(a, cs);
        return true;
    };
    // tokens [s0, s0 + n) in one batched read (the same per-token arithmetic as mhc_read); per slice when off
    auto mhc_read_all = [&](int64_t l, const char* which, bool apply, int64_t s0, int64_t n, const char* norm) -> bool {
        if (mhc_part == nullptr) {
            for (int64_t t0 = s0; t0 < s0 + n; t0 += SL)
                if (!mhc_read(l, which, apply, t0, (int) std::min<int64_t>(SL, s0 + n - t0), norm)) return false;
            return true;
        }
        const std::string w = std::string("hc_") + which + "_";
        const WeightRef *fn = wfind(wt, l, (w + "fn.weight").c_str(), err), *sc = wfind(wt, l, (w + "scale.weight").c_str(), err),
                        *bs = wfind(wt, l, (w + "base.weight").c_str(), err), *wn = wfind(wt, l, norm, err);
        if (!fn || !sc || !bs || !wn || !fn->native_data) { if (err.empty()) err = "verify glm5 prefill: mHC weights"; return false; }
        const bool attn_side = which[0] == 'a';
        MhcBatch b;
        b.R = Rt(s0); b.R_out = Rt(s0); b.apply = apply;
        b.out_prev = boc + (size_t) s0 * N;
        b.post_prev = (attn_side ? postF : postA) + s0 * 4; b.comb_prev = (attn_side ? combF : combA) + s0 * 16;
        b.post_out = (attn_side ? postA : postF) + s0 * 4; b.comb_out = (attn_side ? combA : combF) + s0 * 16;
        b.fn = fn->native_data; b.scale = (const float*) sc->data; b.base = (const float*) bs->data;
        b.w_norm = (const float*) wn->data; b.y = xmc + (size_t) s0 * N;
        b.part = mhc_part; b.ss_part = mhc_ss;
        b.n_tok = (int) n; b.n_embd = (int) N; b.n_slices = m.n_slices;
        b.rms_eps = m.rms_eps; b.hc_eps = m.hc_eps; b.iters = m.iters;
        fused_mhc_read_batch(b, cs);
        return true;
    };
    // the window's attention block on slice [s0, s0 + n) of the chunk, as a commit of those tokens
    auto attn = [&](int64_t l, int64_t s0, int n) -> bool {
        float* xm = xmc + (size_t) s0 * N;
        float* bo = boc + (size_t) s0 * N;
        int32_t* st = stepc + (size_t) s0 * kStepCount;
        const int64_t gi = m.kda_index[(size_t) l], qi = m.dsa_index[(size_t) l];
        if (gi >= 0) {
            const WeightRef *wq = wfind(wt, l, "attn_q.weight", err), *wk = wfind(wt, l, "attn_k.weight", err),
                            *wv = wfind(wt, l, "attn_v.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                            *wfa = wfind(wt, l, "ssm_f_a.weight", err), *wfb = wfind(wt, l, "ssm_f_b.weight", err),
                            *wga = wfind(wt, l, "ssm_g_a.weight", err), *wgb = wfind(wt, l, "ssm_g_b.weight", err),
                            *wbe = wfind(wt, l, "ssm_beta.weight", err), *wdt = wfind(wt, l, "ssm_dt.bias", err),
                            *wsa = wfind(wt, l, "ssm_a", err), *wnm = wfind(wt, l, "ssm_norm.weight", err);
            if (!wq || !wk || !wv || !wo || !wfa || !wfb || !wga || !wgb || !wbe || !wdt || !wsa || !wnm) return false;
            float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
            float* conv = state + (uint64_t) m.S * m.H * m.S;
            float* qkv = qkv_L_ + (size_t) gi * MT * m.C;
            float* hb = h_L_ + (size_t) gi * MT * m.C;
            float* gate = m.gate + (size_t) gi * MT * m.H * m.S;
            float* beta = beta_L_ + (size_t) gi * MT * m.H;
            native_quantize_q8_1(xm, xq_, (int) N, n, cs);
            native_mmvq(wq->native_type, wq->native_data, xq_, m.q, (int) N, (int) m.DI, n, cs);
            native_mmvq(wk->native_type, wk->native_data, xq_, m.k, (int) N, (int) m.DI, n, cs);
            native_mmvq(wv->native_type, wv->native_data, xq_, m.v, (int) N, (int) m.DI, n, cs);
            native_mmvq(wfa->native_type, wfa->native_data, xq_, m.fa, (int) N, 128, n, cs);
            native_mmvq(wga->native_type, wga->native_data, xq_, m.ga, (int) N, 128, n, cs);
            native_mmvq(wbe->native_type, wbe->native_data, xq_, m.bl, (int) N, (int) m.H, n, cs);
            glm5_pack3(m.q, m.k, m.v, qkv, (int) m.DI, n, cs);
            gdn_conv_l2_multi(conv, qkv, m.convw + (size_t) gi * m.C * 4, hb, (int) m.C, (int) (2 * m.H), 1e-6f, n, cs, 0);
            mmvq(wfb, m.fa, 128, m.f, (int) m.DI, n, m.q8, cs);
            mmvq(wgb, m.ga, 128, z_, (int) m.DI, n, m.q8, cs);
            kda_gate_beta_multi(m.f, (const float*) wdt->data, (const float*) wsa->data, m.lb, m.bl, gate, beta, (int) m.H, n, cs);
            gdn_step_norm_multi(state, hb, (int) m.C, gate, beta, z_, (const float*) wnm->data, m.rms_eps, y_, (int) m.H,
                                (int) m.H, n, nkeep + n, cs, 0, nullptr, true);   // the commit: the state written
            gdn_conv_commit(conv, qkv, (int) m.C, nkeep + n, cs);
            mmvq(wo, y_, (int) m.DI, bo, (int) N, n, m.q8, cs);
            return true;
        }
        const WeightRef *wqa = wfind(wt, l, "attn_q_a.weight", err), *wqan = wfind(wt, l, "attn_q_a_norm.weight", err),
                        *wqb = wfind(wt, l, "attn_q_b.weight", err), *wkb = wfind(wt, l, "attn_k_b.weight", err),
                        *wvb = wfind(wt, l, "attn_v_b.weight", err), *wkv = wfind(wt, l, "attn_kv_a_mqa.weight", err),
                        *wkvn = wfind(wt, l, "attn_kv_a_norm.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                        *wik = wfind(wt, l, "indexer.attn_k.weight", err), *wiqb = wfind(wt, l, "indexer.attn_q_b.weight", err),
                        *wikn = wfind(wt, l, "indexer.k_norm.weight", err), *wikb = wfind(wt, l, "indexer.k_norm.bias", err),
                        *wip = wfind(wt, l, "indexer.proj.weight", err), *wape = wfind(wt, l, "indexer_compressor_ape.weight", err),
                        *wig = wfind(wt, l, "indexer_compressor_gate.weight", err);
        if (!wqa || !wqan || !wqb || !wkb || !wvb || !wkv || !wkvn || !wo || !wik || !wiqb || !wikn || !wikb || !wip ||
            !wape || !wig) return false;
        const int64_t QLR = m.QLR, NH = m.NH, QK = m.QK, KVR = m.KVR, VD = m.VD, ID = m.ID, IH = m.IH;
        uint8_t* lat_pool = m.latent + (size_t) qi * m.max_cells * m.lat_row;
        float* tail = m.tail + (size_t) qi * 3 * 2 * ID;
        uint16_t* pooled = m.pooled + (size_t) qi * (m.max_cells / 4 + 1) * ID;
        float* rk = m.raw_k + (size_t) qi * MT * ID;
        float* rg = m.raw_g + (size_t) qi * MT * ID;
        native_quantize_q8_1(xm, xq_, (int) N, n, cs);
        native_mmvq(wqa->native_type, wqa->native_data, xq_, m.qr, (int) N, (int) QLR, n, cs);
        native_mmvq(wkv->native_type, wkv->native_data, xq_, m.kv, (int) N, (int) KVR, n, cs);
        native_mmvq(wik->native_type, wik->native_data, xq_, rk, (int) N, (int) ID, n, cs);
        native_mmvq(wig->native_type, wig->native_data, xq_, rg, (int) N, (int) ID, n, cs);
        native_qsa_rms_norm_weighted(m.qr, (const float*) wqan->data, m.qr, (int) QLR, n, m.rms_eps, cs);
        native_qsa_rms_norm_weighted(m.kv, (const float*) wkvn->data, m.kv, (int) KVR, n, m.rms_eps, cs);
        if (m.lat_q4) glm5_latent_store_q4(m.kv, lat_pool, m.page_table, st, kStepCount, 4, (int) KVR, n, cs);
        else glm5_latent_store(m.kv, (uint16_t*) lat_pool, m.page_table, st, kStepCount, 4, (int) KVR, n, cs);
        mmvq(wqb, m.qr, (int) QLR, m.qb, (int) (NH * QK), n, m.q8, cs);
        glm5_transpose_abd(m.qb, m.qT, n, (int) NH, (int) QK, cs);
        // stage 1: this layer's wk_b / wv_b in F32 once per chunk (both Q8_0: [NH][KVR][QK] and [NH][VD][KVR])
        const bool blas = kb_f != nullptr && wkb->native_type == 8 && wvb->native_type == 8;   // GGML_TYPE_Q8_0
        if (blas && kbvb_layer != l) {
            dequant_q8_0((const uint8_t*) wkb->native_data, kb_f, NH * KVR * QK, cs);
            dequant_q8_0((const uint8_t*) wvb->native_data, vb_f, NH * VD * KVR, cs);
            kbvb_layer = l;
        }
        const float one = 1.0f, zero = 0.0f;
        if (blas) {   // qaT[h] (n x KVR, row-major) = qT[h] (n x QK) . wk_b[h]^T, every head in one call
            cublasSgemmStridedBatched(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) KVR, n, (int) QK, &one,
                                      kb_f, (int) QK, (long long) KVR * QK, m.qT, (int) QK, (long long) n * QK, &zero,
                                      m.qaT, (int) KVR, (long long) n * KVR, (int) NH);
        } else {
            native_quantize_q8_1(m.qT, m.q8, (int) (QK * NH * n), 1, cs);
            const size_t row = (size_t) (QK / 32) * 34, qcol = (size_t) (QK / 32) * 36;
            for (int64_t h = 0; h < NH; ++h)
                native_mmvq(wkb->native_type, (const uint8_t*) wkb->native_data + (size_t) h * KVR * row,
                            m.q8 + (size_t) h * n * qcol, m.qaT + (size_t) h * n * KVR, (int) QK, (int) KVR, n, cs);
        }
        glm5_transpose_abd(m.qaT, m.qa, (int) NH, n, (int) KVR, cs);
        const Glm5KpoolBuffers kb{tail, pooled};
        for (int t = 0; t < n; ++t)
            glm5_kpool_append(rk + (size_t) t * ID, rg + (size_t) t * ID, st + (size_t) t * kStepCount + kStepPos,
                              (const float*) wikn->data, (const float*) wikb->data, (const float*) wape->data, m.ln_eps,
                              kb, m.max_cells, cs);
        mmvq(wiqb, m.qr, (int) QLR, m.iq, (int) (IH * ID), n, m.q8, cs);
        glm5_f32_gemv((const float*) wip->data, xm, m.iw, (int) N, (int) IH, n, 1.0f / std::sqrt((float) (ID * IH)), cs);
        glm5_kpool_scores(m.iq, m.iw, pooled, st, n, max_blocks_, scores_, cs);
        qsa_block_topk(scores_, st, n, max_blocks_, cap_, s, sel_, cs);
        if (m.lat_q4) mla_decode_attn_batch_q4(m.qa, lat_pool, m.page_table, 4, sel_, st, cap_, (int) NH, 1.0f / std::sqrt((float) QK), m.mla_scratch, m.lat, n, cs);
        else mla_decode_attn_batch(m.qa, (const uint16_t*) lat_pool, m.page_table, 4, sel_, st, cap_, (int) NH, 1.0f / std::sqrt((float) QK),
                              m.mla_scratch, m.lat, n, cs);
        glm5_transpose_abd(m.lat, m.latT, n, (int) NH, (int) KVR, cs);
        if (blas) {   // vbT[h] (n x VD) = latT[h] (n x KVR) . wv_b[h]^T, every head in one call
            cublasSgemmStridedBatched(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) VD, n, (int) KVR, &one,
                                      vb_f, (int) KVR, (long long) VD * KVR, m.latT, (int) KVR, (long long) n * KVR, &zero,
                                      m.vbT, (int) VD, (long long) n * VD, (int) NH);
        } else {
            native_quantize_q8_1(m.latT, m.q8, (int) (KVR * NH * n), 1, cs);
            const size_t row = (size_t) (KVR / 32) * 34, qcol = (size_t) (KVR / 32) * 36;
            for (int64_t h = 0; h < NH; ++h)
                native_mmvq(wvb->native_type, (const uint8_t*) wvb->native_data + (size_t) h * VD * row,
                            m.q8 + (size_t) h * n * qcol, m.vbT + (size_t) h * n * VD, (int) KVR, (int) VD, n, cs);
        }
        glm5_transpose_abd(m.vbT, m.vb, (int) NH, n, (int) VD, cs);
        mmvq(wo, m.vb, (int) (NH * VD), bo, (int) N, n, m.q8, cs);
        return true;
    };
    // the routed experts of the whole chunk (T tokens) on the GPU, the shared expert, the combine into boc
    // 2026-10-09: the routed experts through llama.cpp's MMQ (int8 tensor cores; Project Maya's GLM prompt path does
    // the same) instead of native_expert_grouped, in every layer whose gate/up and down types MMQ has (UD-IQ4_XS: 39 of
    // the 42, IQ3_S / IQ4_XS; the 3 with Q6_K downs stay native).  Each expert of a launch sits in its own staging slot
    // (a VRAM-cache resident is copied there), so one launch per product covers them all: gate|up, the clamped swiglu
    // in place, down into the routed outputs.  The gate|up rows and both q8_1 images live in gscratch.  Not bitwise the
    // native kernels (another summation order).  STRATA_GLM5_PF_MMQ=0: native_expert_grouped everywhere.
    namespace mmq = strata::prefill::mmq;
    static const bool pf_mmq_on = [] { const char* v = std::getenv("STRATA_GLM5_PF_MMQ"); return !(v && v[0] == '0'); }();
    const int64_t NFF = g.n_ff;
    std::unique_ptr<mmq::Context> mq;
    std::vector<int8_t> mmq_layer((size_t) g.n_layers, -1);
    float* mm_gu = nullptr;
    uint8_t *mm_xq = nullptr, *mm_hq = nullptr;
    int32_t* mm_iota = nullptr;
    if (pf_mmq_on && mmq::built()) {
        const size_t a_gu = ((size_t) GE * 2 * NFF * 4 + 255) & ~(size_t) 255;
        const size_t a_xq = (mmq::q8_bytes(GE, N) + 255) & ~(size_t) 255;
        const size_t a_hq = (mmq::q8_bytes(GE, NFF) + 255) & ~(size_t) 255;
        if (a_gu + a_xq + a_hq + (size_t) GE * 4 <= gscratch_bytes) {
            uint8_t* gs = (uint8_t*) gscratch;
            mm_gu = (float*) gs;
            mm_xq = gs + a_gu;
            mm_hq = gs + a_gu + a_xq;
            mm_iota = (int32_t*) (gs + a_gu + a_xq + a_hq);
            mq = std::make_unique<mmq::Context>();
        }
    }
    // the layer's experts through MMQ (its types covered and a tile that fits the card for both products)
    auto use_mmq = [&](int64_t l) -> bool {
        if (!mq) return false;
        int8_t& v = mmq_layer[(size_t) l];
        if (v < 0) {
            const auto& fm = lay.fmt[(size_t) l];
            v = (int8_t) (mmq::supported(fm.gu_type) && mmq::supported(fm.d_type) && mmq::fits(fm.gu_type, 2 * NFF) &&
                          mmq::fits(fm.d_type, N));
        }
        return v != 0;
    };
    // one launch's experts (nb, their weights max_blob apart from wbase): rows [bounds[e], bounds[e+1]) of the entries
    // read x[tok[row]] (ld floats) and write out[dst[row]] (N floats a row)
    auto run_mmq = [&](int64_t l, const uint8_t* wbase, int nb, int entries, int max_rows, const int32_t* d_bounds,
                       const int32_t* d_tok, const float* x, int64_t ld, const int32_t* d_dst, float* out) {
        mmq::iota(mm_iota, entries, cs);   // (gscratch is the native kernels' too: written each time)
        const auto& fm = lay.fmt[(size_t) l];
        const NativeExpertLayout L = native_expert_layout(fm.gu_type, fm.d_type, fm.n_embd, fm.n_ff);
        mmq::quantize(x, d_tok, mm_xq, fm.gu_type, N, ld, entries, cs);
        mmq::Product gu;
        gu.w = wbase; gu.type = fm.gu_type; gu.w_rows = 2 * NFF; gu.w_cols = N; gu.expert_bytes = max_blob; gu.n = nb;
        gu.xq = mm_xq; gu.bounds = d_bounds; gu.ids = mm_iota; gu.total_rows = entries; gu.max_rows = max_rows;
        gu.dst = mm_gu; gu.ld_dst = 2 * NFF;
        mq->run(gu, cs);
        glm5_swiglu_rows(mm_gu, entries, (int) NFF, m.clamp_exp, cs);
        mmq::quantize(mm_gu, nullptr, mm_hq, fm.d_type, NFF, 2 * NFF, entries, cs);
        mmq::Product dn;
        dn.w = wbase + L.down_off; dn.type = fm.d_type; dn.w_rows = N; dn.w_cols = NFF; dn.expert_bytes = max_blob;
        dn.n = nb; dn.xq = mm_hq; dn.bounds = d_bounds; dn.ids = d_dst; dn.total_rows = entries; dn.max_rows = max_rows;
        dn.dst = out; dn.ld_dst = N;
        mq->run(dn, cs);
    };
    // 2026-10-10: the prompt path's CPU lane.  A short prompt still streams about half of every layer's experts (85
    // tokens: 6,059 blobs, ~62 GB, 1.4 s of PCIe at the least) though most of them have one or two tokens.  Those
    // lightest RAM-tier experts are computed on host threads with the windows' CPU kernels (ggml-cpu's arithmetic,
    // native_expert.hpp) while the GPU streams the rest; their rows are copied into `parts` before the combine.
    // Chunks of <= 1,024 tokens only (beyond, the fetch's threads would compete for the cores).  85 tokens, warm:
    // 2.7 -> 1.7-1.8 s (the GPU's loop then waits on PCIe: 2,409 blobs ~ 0.54 s).
    // STRATA_GLM5_PF_CPU = the most tokens an expert may have for the CPU (8; 0 = off), STRATA_GLM5_PF_CPU_SHARE = the
    // largest share of the streamed experts it takes (0.9), STRATA_GLM5_PF_CPU_THREADS (22).
    static const int pf_cpu_rows = [] { const char* v = std::getenv("STRATA_GLM5_PF_CPU"); return v ? std::min(16, std::atoi(v)) : 8; }();
    static const double pf_cpu_share = [] { const char* v = std::getenv("STRATA_GLM5_PF_CPU_SHARE"); return v ? std::atof(v) : 0.9; }();
    static const int pf_cpu_threads = [] { const char* v = std::getenv("STRATA_GLM5_PF_CPU_THREADS"); return v ? std::max(1, std::atoi(v)) : 22; }();
    namespace kc = strata::kernels::cpu;
    std::vector<int8_t> cpu_fmt_ok((size_t) g.n_layers, -1);
    std::vector<kc::NativeFmt> cpu_fmt((size_t) g.n_layers);
    std::vector<float> cpu_x, cpu_out;
    std::vector<int32_t> cpu_dst;
    int64_t pf_cpu_n = 0;
    double pf_cpu_join_ms = 0;
    double pf_moe_ms[4] = {};   // STRATA_GLM5_PF_PROF, the chunk path's moe: router+shared on the GPU, split, loop, tail
    auto cpu_layer_ok = [&](int64_t l) -> bool {
        if (!kc::native_experts_available()) return false;
        int8_t& v = cpu_fmt_ok[(size_t) l];
        if (v < 0) {
            const auto& fm = lay.fmt[(size_t) l];
            std::string e2;
            v = (int8_t) kc::native_fmt(fm.gu_type, fm.d_type, fm.n_embd, fm.n_ff, cpu_fmt[(size_t) l], e2, m.clamp_exp);
        }
        return v != 0;
    };
    // 2026-10-10: the next MoE layer's file-tier experts read ahead into the page cache - threads touch their mapped
    // pages while this layer computes, so the fetch's blob() later finds them cached instead of faulting them in one
    // page run at a time on this thread.  Every layer's experts are nearly all used from a few hundred tokens on (and
    // a short prompt would read many it does not use): chunks / groups of >= 256 tokens.  STRATA_GLM5_PF_TOUCH = the
    // threads (8; 0 = off).
    static const int pf_touch_th = [] { const char* v = std::getenv("STRATA_GLM5_PF_TOUCH"); return v ? std::atoi(v) : 8; }();
    std::vector<std::thread> touch_th;
    struct JoinTouch {
        std::vector<std::thread>& t;
        ~JoinTouch() { for (auto& x : t) if (x.joinable()) x.join(); }
    } touch_join{touch_th};
    std::atomic<uint32_t> touch_sink{0};
    auto touch_next = [&](int64_t ln, int64_t T) {
        for (auto& t : touch_th) t.join();
        touch_th.clear();
        const int64_t l_last = l_end < 0 ? g.n_layers : l_end;
        if (pf_touch_th <= 0 || !f.ram || !f.stable || T < 256 || ln >= l_last || ln < g.n_dense_layers) return;
        auto list = std::make_shared<std::vector<int32_t>>();
        for (int32_t e = 0; e < (int32_t) NE; ++e)
            if (!(f.host_res && f.host_res[(size_t) ln * NE + e] >= 0) && f.ram(f.ctx, ln, e) == nullptr) list->push_back(e);
        if (list->empty()) return;
        auto next = std::make_shared<std::atomic<size_t>>(0);
        const uint64_t bbn = lay.blob_bytes(ln);
        for (int w = 0; w < pf_touch_th; ++w)
            touch_th.emplace_back([&f, &touch_sink, list, next, bbn, ln] {
                uint32_t s = 0;
                for (size_t k; (k = next->fetch_add(1)) < list->size();) {
                    const volatile uint8_t* p = f.stable(f.ctx, ln, (*list)[k]);
                    if (p == nullptr) continue;
                    for (uint64_t off = 0; off < bbn; off += 4096) s += p[off];
                }
                touch_sink += s;
            });
    };
    // the shared expert (or a dense layer's FFN) of T rows of X into Y through MMQ - set below, after gemm_q8
    std::function<bool(int64_t, bool, const float*, int64_t, float*)> ffn_q8;
    // 2026-10-10: the router's logits (F32 [288][4096]) for the chunk in one cublasSgemm instead of glm5_f32_gemv's
    // per-token products (another summation order, FP32 throughout).  STRATA_GLM5_PF_ROUTER_BLAS=0: the gemv
    static const bool pf_router_blas = [] { const char* v = std::getenv("STRATA_GLM5_PF_ROUTER_BLAS"); return !(v && v[0] == '0'); }();
    auto router_logits = [&](const WeightRef* wr, int64_t T) {
        const float one = 1.0f, zero = 0.0f;
        if (pf_router_blas && pf_blas != nullptr && T >= 16 &&
            cublasSgemm(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) NE, (int) T, (int) N, &one, (const float*) wr->data, (int) N,
                        xmc, (int) N, &zero, logc, (int) NE) == CUBLAS_STATUS_SUCCESS)
            return;
        glm5_f32_gemv((const float*) wr->data, xmc, logc, (int) N, (int) NE, (int) T, 1.0f, cs);
    };
    auto moe = [&](int64_t l, int64_t T) -> bool {
        touch_next(l + 1, T);
        const WeightRef *wr = wfind(wt, l, "ffn_gate_inp.weight", err), *wb = wfind(wt, l, "exp_probs_b.bias", err),
                        *wsg = wfind(wt, l, "ffn_gate_shexp.weight", err), *wsu = wfind(wt, l, "ffn_up_shexp.weight", err),
                        *wsd = wfind(wt, l, "ffn_down_shexp.weight", err);
        if (!wr || !wb || !wsg || !wsu || !wsd) return false;
        const int64_t TK = T * K;
        router_logits(wr, T);
        router_sigmoid_topk(logc, (const float*) wb->data, (int) T, (int) NE, (int) K, m.wscale, idsc, wc, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data; nsw.up_type = wsu->native_type;
        nsw.up_data = wsu->native_data; nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data; nsw.q8_1 = m.q8;
        if (!(ffn_q8 && ffn_q8(l, false, xmc, T, shc)))
            for (int64_t s0 = 0; s0 < T; s0 += SL) {
                const int n = (int) std::min<int64_t>(SL, T - s0);
                shared_expert_multi(n, xmc + (size_t) s0 * N, nullptr, nsw, nullptr, sh_gate_, sh_up_, sh_g_,
                                    shc + (size_t) s0 * N, N, m.NFF, cs, nullptr, 0, m.clamp_sh);
            }
        quantize_q8_1_rows(xmc, T, N, q8c, cs);
        cudaMemcpyAsync(h_ids.data(), idsc, (size_t) TK * 4, cudaMemcpyDeviceToHost, cs);
        const auto tm0 = std::chrono::steady_clock::now();
        if (cudaStreamSynchronize(cs) != cudaSuccess) { err = "verify glm5 prefill: the router failed on the GPU"; return false; }
        auto tm = std::chrono::steady_clock::now();
        auto mlap2 = [&](int i) {
            const auto t1 = std::chrono::steady_clock::now();
            pf_moe_ms[i] += std::chrono::duration<double, std::milli>(t1 - tm).count();
            tm = t1;
        };
        pf_moe_ms[0] += std::chrono::duration<double, std::milli>(tm - tm0).count();
        for (auto& v : by) v.clear();
        used.clear();
        const bool count_use = pf_usage.size() == (size_t) g.n_layers * NE;
        for (int64_t i = 0; i < TK; ++i) {
            const int32_t e = h_ids[(size_t) i];
            if (e < 0 || e >= NE) continue;
            if (count_use) pf_usage[(size_t) l * NE + e] += 1.0f;
            if (by[(size_t) e].empty()) used.push_back(e);
            by[(size_t) e].push_back((int32_t) i);
        }
        const auto& fm = lay.fmt[(size_t) l];
        NativeExpertLayout L = native_expert_layout(fm.gu_type, fm.d_type, fm.n_embd, fm.n_ff);
        L.swiglu_limit = m.clamp_exp;
        const uint64_t bb = lay.blob_bytes(l);
        const bool mm = use_mmq(l);
        // the CPU lane, work stealing: `used` holds the residents first, then the streamed experts by rows, most first;
        // the GPU's launches take from the front, the lane's threads from the back - the lightest - while those have at
        // most pf_cpu_rows rows (and within pf_cpu_share of the streamed ones), until the two meet.  A fixed split was
        // either idle or late by turns (85 tokens: 1.7-2.4 s).  The blobs are looked up on the lane's threads (1,500
        // lookups on this thread took 0.7 s).
        // the lane also takes file-tier experts, read through the stable mapping by its threads (many SSD reads in
        // flight): 85 tokens 2.2-2.3 -> 1.7-1.9 s, 720 tokens 8.7 -> 7.1 s.  STRATA_GLM5_PF_CPU_FILE=0: RAM tier only
        static const bool lane_file = [] { const char* v = std::getenv("STRATA_GLM5_PF_CPU_FILE"); return !(v && v[0] == '0'); }();
        const bool lane = pf_cpu_rows > 0 && pf_cpu_share > 0 && T <= 1024 && f.ram && f.blob && cpu_layer_ok(l);
        size_t q_lo = 0, q_hi = used.size(), q_el = used.size();   // the lane's eligible tail is [q_el, used.size())
        std::mutex q_mu;
        std::vector<int32_t> ro;
        if (lane) {
            // 0 = VRAM resident, 1 = streamed only, 2 = the lane may take it (RAM tier, few rows)
            std::vector<int8_t> cls((size_t) NE, 1);
            size_t n_res = 0, n_el = 0;
            for (int32_t e : used) {
                int8_t c = 1;
                if (f.host_res && f.host_res[(size_t) l * NE + e] >= 0) c = 0;
                else if ((int) by[(size_t) e].size() <= pf_cpu_rows && (lane_file || f.ram(f.ctx, l, e) != nullptr)) c = 2;
                cls[(size_t) e] = c;
                n_res += c == 0;
                n_el += c == 2;
            }
            std::stable_sort(used.begin(), used.end(), [&](int32_t a, int32_t b) {
                const int8_t ca = cls[(size_t) a], cb = cls[(size_t) b];
                if (ca != cb) return ca < cb;
                return by[(size_t) a].size() > by[(size_t) b].size();
            });
            const size_t cap = (size_t) ((double) (used.size() - n_res) * pf_cpu_share);
            q_el = used.size() - std::min(n_el, cap);
        }
        mlap2(1);
        std::atomic<bool> cpu_failed{false};
        std::vector<std::thread> cpu_th;
        struct JoinAll {
            std::vector<std::thread>& t;
            ~JoinAll() { for (auto& x : t) if (x.joinable()) x.join(); }
        } cpu_join{cpu_th};   // an early return below still waits for the lane
        if (q_el < used.size()) {
            cpu_dst.clear();
            ro.assign(used.size() - q_el + 1, 0);
            for (size_t k = q_el; k < used.size(); ++k) {
                ro[k - q_el] = (int32_t) cpu_dst.size();
                for (int32_t i : by[(size_t) used[k]]) cpu_dst.push_back(i);
            }
            ro[used.size() - q_el] = (int32_t) cpu_dst.size();
            cpu_x.resize((size_t) T * N);
            cudaMemcpy(cpu_x.data(), xmc, (size_t) T * N * 4, cudaMemcpyDeviceToHost);
            cpu_out.resize(cpu_dst.size() * (size_t) N);
            const kc::NativeFmt* cf = &cpu_fmt[(size_t) l];
            const int nth = std::min<int>(pf_cpu_threads, (int) (used.size() - q_el));
            for (int w = 0; w < nth; ++w)
                cpu_th.emplace_back([&, cf] {
                    const int MR = 16;
                    std::vector<uint8_t> abuf((size_t) MR * cf->act_bytes + 64), hbuf((size_t) MR * cf->h_bytes + 64);
                    uint8_t* ab = (uint8_t*) (((uintptr_t) abuf.data() + 63) & ~(uintptr_t) 63);
                    uint8_t* hbq = (uint8_t*) (((uintptr_t) hbuf.data() + 63) & ~(uintptr_t) 63);
                    std::vector<float> ff((size_t) MR * cf->n_ff);
                    const void* act[MR];
                    const void* hq[MR];
                    float* ffp[MR];
                    float* outp[MR];
                    for (;;) {
                        size_t k;
                        {
                            std::lock_guard<std::mutex> lk(q_mu);
                            if (q_hi <= q_lo || q_hi <= q_el) break;
                            k = --q_hi;
                        }
                        const int32_t e = used[k];
                        const int r0 = ro[k - q_el], nt = std::min(MR, ro[k - q_el + 1] - r0);
                        // (the stable pointer: blob() may hand out a staging buffer the next lookup recycles)
                        const uint8_t* blob = lane_file ? (f.stable ? f.stable(f.ctx, l, e) : nullptr) : f.ram(f.ctx, l, e);
                        if (blob == nullptr) { cpu_failed = true; continue; }
                        for (int t = 0; t < nt; ++t) {
                            const int tok = cpu_dst[(size_t) (r0 + t)] / (int) K;
                            kc::native_quant_act(*cf, cpu_x.data() + (size_t) tok * N, ab + (size_t) t * cf->act_bytes);
                            act[t] = ab + (size_t) t * cf->act_bytes;
                            ffp[t] = ff.data() + (size_t) t * cf->n_ff;
                            outp[t] = cpu_out.data() + (size_t) (r0 + t) * N;
                        }
                        kc::native_gu_rows(*cf, blob, act, nt, ffp, 0, (int) cf->n_ff);
                        for (int t = 0; t < nt; ++t) {
                            kc::native_quant_h(*cf, ffp[t], hbq + (size_t) t * cf->h_bytes);
                            hq[t] = hbq + (size_t) t * cf->h_bytes;
                        }
                        kc::native_down_rows(*cf, blob, hq, nt, outp, 0, (int) cf->n_embd);
                    }
                });
        }
        for (size_t bi = 0;; ++bi) {
            size_t b0;
            int nb;
            {
                std::lock_guard<std::mutex> lk(q_mu);
                // up to B experts and GE entries (one expert has at most T <= C <= GE)
                nb = 0;
                int64_t ne_ = 0;
                while (q_lo + (size_t) nb < q_hi && nb < B &&
                       (nb == 0 || ne_ + (int64_t) by[(size_t) used[q_lo + (size_t) nb]].size() <= GE))
                    ne_ += (int64_t) by[(size_t) used[q_lo + (size_t) nb++]].size();
                b0 = q_lo;
                q_lo += (size_t) nb;
            }
            if (nb <= 0) break;
            const int hf = (int) (bi & 1);
            uint8_t* dstage = stage + (size_t) hf * B * max_blob;
            uint8_t* hstage = pin + (size_t) hf * B * max_blob;
            int32_t* start = pplan + (size_t) hf * plan_n;
            int32_t* dst = start + (B + 1);
            int32_t* tok = dst + CK;
            int32_t* count = tok + CK;
            unsigned long long* hptr = pptr + (size_t) hf * B;
            const auto tc = std::chrono::steady_clock::now();
            // this half's last use: its DMA has read the page-locked slots, its experts have read the device slots and
            // its plan (the compute recorded ev_done after the plan copies and the launch)
            cudaEventSynchronize(ev_copied[hf]);
            cudaEventSynchronize(ev_done[hf]);
            pf_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            int ns = 0;
            int32_t stream_e[32];
            int entries = 0, max_rows = 0;
            // the streamed experts first (staging slots 0..ns-1, as the DMA below fills them), then the residents: with
            // MMQ each resident is copied into the next slot, so every expert of the launch is max_blob apart
            int ord[32], no = 0;
            for (int pass = 0; pass < 2; ++pass)
                for (int j = 0; j < nb; ++j) {
                    const int32_t e = used[b0 + (size_t) j];
                    const bool res = f.host_res && f.host_res[(size_t) l * NE + e] >= 0;
                    if (res == (pass == 1)) ord[no++] = j;
                }
            for (int q = 0; q < nb; ++q) {
                const int32_t e = used[b0 + (size_t) ord[q]];
                const int32_t slot = f.host_res ? f.host_res[(size_t) l * NE + e] : -1;
                if (slot >= 0) {
                    const uint8_t* src = f.cache_base + (f.slot_off ? (size_t) f.slot_off[slot] : (size_t) slot * f.cache_blob);
                    if (mm) {
                        cudaMemcpyAsync(dstage + (size_t) q * max_blob, src, (size_t) bb, cudaMemcpyDeviceToDevice, cs);
                        hptr[q] = (unsigned long long) (dstage + (size_t) q * max_blob);
                    } else {
                        hptr[q] = (unsigned long long) src;
                    }
                    ++prefill_resident;
                } else {
                    hptr[q] = (unsigned long long) (dstage + (size_t) ns * max_blob);
                    stream_e[ns++] = e;
                    ++prefill_streamed;
                }
                start[q] = entries;
                max_rows = std::max<int>(max_rows, (int) by[(size_t) e].size());
                for (int32_t i : by[(size_t) e]) { dst[entries] = i; tok[entries] = (int32_t) (i / K); ++entries; }
            }
            start[nb] = entries;
            count[0] = nb;
            if (ns > 0) {   // the blobs into page-locked memory on threads, then one DMA on the copy stream
                bool failed = false;
                std::vector<std::thread> th;
                th.reserve((size_t) ns);
                std::atomic<int64_t> us_blob{0}, us_copy{0};   // STRATA_GLM5_PF_PROF: summed over the threads
                const auto tf = std::chrono::steady_clock::now();
                // 2026-10-09: a blob in page-locked host memory (the RAM tier is pinned and mapped since v0.1.41) goes
                // to the device in its own DMA; only the others (the file tier) are copied into the staging first.
                // 4,400 tokens in chunks of 4,096: the copies into the staging took 89 s summed over 16 threads (18 s).
                std::atomic<bool> direct[32];
                for (int k = 0; k < ns; ++k) direct[k] = false;
                // 2026-10-10, a short prompt (<= 1,024 tokens) with PF_DIRECT: the lookups on this thread (~0.1 ms each)
                // and threads only for the blobs to copy - a thread a blob waited for a core while the CPU lane held
                // them (85 tokens: the fetch took 0.92 s, its threads 0.32 s)
                // (blob() on this thread also for the file tier: reading those on threads instead was slower - 85 tokens
                // 1.7 -> 2.3 s, 720 tokens 6.8 -> 8.1 s; the source's own read-ahead serves them in order)
                static const int64_t serial_max = [] { const char* v = std::getenv("STRATA_GLM5_PF_SERIAL_MAX"); return v ? std::atoll(v) : 1024; }();
                const bool serial = pf_direct && T <= serial_max;
                for (int k = 0; serial && k < ns; ++k) {
                    const auto a = std::chrono::steady_clock::now();
                    const uint8_t* hb = f.blob ? f.blob(f.ctx, l, stream_e[k]) : nullptr;
                    us_blob += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - a).count();
                    if (hb == nullptr) { failed = true; break; }
                    cudaPointerAttributes at{};
                    if (cudaPointerGetAttributes(&at, hb) == cudaSuccess && at.type == cudaMemoryTypeHost) {
                        hsrc[k] = hb;
                        direct[k] = true;
                    } else {
                        cudaGetLastError();
                        std::memcpy(hstage + (size_t) k * max_blob, hb, (size_t) bb);
                    }
                }
                for (int k = 0; !serial && k < ns; ++k)
                    th.emplace_back([&, k] {
                        const auto a = std::chrono::steady_clock::now();
                        const uint8_t* hb = f.blob ? f.blob(f.ctx, l, stream_e[k]) : nullptr;
                        const auto b = std::chrono::steady_clock::now();
                        if (hb == nullptr) { failed = true; return; }
                        cudaPointerAttributes at{};
                        if (pf_direct && cudaPointerGetAttributes(&at, hb) == cudaSuccess && at.type == cudaMemoryTypeHost) {
                            hsrc[k] = hb;
                            direct[k] = true;
                        } else {
                            cudaGetLastError();
                            std::memcpy(hstage + (size_t) k * max_blob, hb, (size_t) bb);
                        }
                        us_blob += std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
                        us_copy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - b).count();
                    });
                for (auto& t : th) t.join();
                pf_fetch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
                pf_blob_thread_ms += (double) us_blob.load() / 1000.0;
                pf_copy_thread_ms += (double) us_copy.load() / 1000.0;
                if (failed) { err = "verify glm5 prefill: an expert of layer " + std::to_string(l) + " is in no tier"; return false; }
                if (!pf_direct) {
                    cudaMemcpyAsync(dstage, hstage, (size_t) (ns - 1) * max_blob + bb, cudaMemcpyHostToDevice, copy_s);
                } else {   // runs of staged blobs in one DMA each, the pinned ones straight from the RAM tier
                    for (int k = 0; k < ns;) {
                        if (direct[k]) {
                            cudaMemcpyAsync(dstage + (size_t) k * max_blob, hsrc[k], (size_t) bb, cudaMemcpyHostToDevice, copy_s);
                            ++prefill_direct;
                            ++k;
                            continue;
                        }
                        int e = k + 1;
                        while (e < ns && !direct[e]) ++e;
                        cudaMemcpyAsync(dstage + (size_t) k * max_blob, hstage + (size_t) k * max_blob,
                                        (size_t) (e - k - 1) * max_blob + bb, cudaMemcpyHostToDevice, copy_s);
                        k = e;
                    }
                }
            }
            cudaEventRecord(ev_copied[hf], copy_s);
            prefill_ms_copy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            int32_t* dpl = dplan + (size_t) hf * plan_n;
            unsigned long long* dp = dptr + (size_t) hf * B;
            cudaMemcpyAsync(dpl, start, (size_t) (nb + 1) * 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dpl + (B + 1), dst, (size_t) entries * 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dpl + (B + 1) + CK, tok, (size_t) entries * 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dpl + (B + 1) + 2 * CK, count, 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dp, hptr, (size_t) nb * 8, cudaMemcpyHostToDevice, cs);
            cudaStreamWaitEvent(cs, ev_copied[hf], 0);
            if (mm)
                run_mmq(l, dstage, nb, entries, max_rows, dpl, dpl + (B + 1) + CK, xmc, N, dpl + (B + 1), parts);
            else
                native_expert_grouped(L, dp, dpl, dpl + (B + 1) + 2 * CK, dpl + (B + 1), dpl + (B + 1) + CK, B, GE,
                                      q8c, gscratch, parts, cs);
            cudaEventRecord(ev_done[hf], cs);   // the host waits for it before this half is filled again
        }
        mlap2(2);
        if (!cpu_th.empty()) {   // the CPU lane's rows into parts (through gscratch: the GPU's experts are queued before)
            const auto tj = std::chrono::steady_clock::now();
            for (auto& t : cpu_th) t.join();
            cpu_th.clear();
            if (cpu_failed) { err = "verify glm5 prefill: an expert of layer " + std::to_string(l) + " is in no tier (CPU lane)"; return false; }
            pf_cpu_join_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tj).count();
            // the lane took [q_hi, used.size()): rows [ro[q_hi - q_el], ro[end]) of its tables
            const int r_lo = ro[q_hi - q_el], r_hi = ro[used.size() - q_el], cpu_rows = r_hi - r_lo;
            pf_cpu_n += (int64_t) (used.size() - q_hi);
            // in pieces that fit gscratch (rows of N floats and their index)
            const int piece = (int) std::max<int64_t>(1, (int64_t) (gscratch_bytes / ((size_t) N * 4 + 4)) - 1);
            for (int p0 = 0; p0 < cpu_rows; p0 += piece) {
                const int pr = std::min(piece, cpu_rows - p0);
                float* src = (float*) gscratch;
                int32_t* idx = (int32_t*) (src + (size_t) pr * N);
                cudaMemcpyAsync(src, cpu_out.data() + (size_t) (r_lo + p0) * N, (size_t) pr * N * 4, cudaMemcpyHostToDevice, cs);
                cudaMemcpyAsync(idx, cpu_dst.data() + r_lo + p0, (size_t) pr * 4, cudaMemcpyHostToDevice, cs);
                glm5_copy_rows(src, idx, parts, pr, (int) N, cs);
            }
        }
        native_moe_combine_multi(parts, wc, shc, boc, N, K, (int) T, cs);
        kbvb_layer = -1;   // parts (where kb_f / vb_f are carved) now holds this layer's expert outputs
        static const bool moe_prof = std::getenv("STRATA_GLM5_PF_PROF") != nullptr;
        if (moe_prof) { cudaStreamSynchronize(cs); mlap2(3); }
        return true;
    };

    // stage 2: Y (rows x out, row-major) = X (rows x in) . W^T for a Q8_0 weight W [out][in], through cuBLAS in F32;
    // W dequantized a slice of rows at a time into `wscr`
    // 2026-10-09: through MMQ when it has a tile for the matrix (the same int8 tensor cores as the experts; the
    // activations rounded to q8_1 as the window's mmvq rounds them) - no F32 copy of the weights; wscr then holds the
    // q8_1 rows, the bounds and the row table.  STRATA_GLM5_PF_MMQ=0: the dequantize + cublasSgemm below
    auto gemm_q8 = [&](const WeightRef* w, int64_t in, int64_t out, const float* X, int64_t rows, float* Y) -> bool {
        if (w->native_type != 8 || in % 32 != 0) { err = "verify glm5 prefill: a KDA projection is not Q8_0"; return false; }
        const size_t xqb = (mmq::q8_bytes(rows, in) + 255) & ~(size_t) 255;
        // 2026-10-13: MMQ only for in % 256 == 0.  MMQ loads the weights 256 values at a time, so a 128-value row (the
        // KDA ssm_f_b / ssm_g_b) reads past its matrix's last row; those bytes times the zero-padded activations are
        // zero unless a stray f16 scale is Inf/NaN, and then the whole answer turns into "!!!!" - at random, as the
        // memory past the tensor differs run to run (seen here 2026-10-10 and by a user, fixed by turning MMQ off).
        // The same rule as Gemm::native_mmq (src/prefill/gemm.cu).  Measured: the 128-wide matrices run from copies
        // whose tail is 0xFF answered "!!!!" to all 4 test prompts; a 0x00 or 0x3C tail, and this rule, answered all 4.
        if (mq && in % 256 == 0 && xqb + (size_t) (rows + 2) * 4 <= (size_t) WSCR * 4 && mmq::fits(8, out)) {
            uint8_t* xq = (uint8_t*) wscr;
            int32_t* bnd = (int32_t*) (xq + xqb);
            int32_t* ids = bnd + 2;
            mmq::quantize(X, nullptr, xq, 8, in, in, rows, cs);
            glm5_set2(bnd, 0, (int32_t) rows, cs);
            mmq::iota(ids, rows, cs);
            mmq::Product p;
            p.w = w->native_data; p.type = 8; p.w_rows = out; p.w_cols = in; p.expert_bytes = mmq::matrix_bytes(8, out, in);
            p.n = 1; p.xq = xq; p.bounds = bnd; p.ids = ids; p.total_rows = rows; p.max_rows = rows; p.dst = Y; p.ld_dst = out;
            mq->run(p, cs);
            return true;
        }
        const size_t row_bytes = (size_t) in / 32 * 34;
        const int64_t R = std::max<int64_t>(1, std::min<int64_t>(out, WSCR / in));
        const float one = 1.0f, zero = 0.0f;
        for (int64_t r0 = 0; r0 < out; r0 += R) {
            const int64_t nr = std::min<int64_t>(R, out - r0);
            dequant_q8_0((const uint8_t*) w->native_data + (size_t) r0 * row_bytes, wscr, nr * in, cs);
            if (cublasSgemm(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) nr, (int) rows, (int) in, &one, wscr, (int) in, X,
                            (int) in, &zero, Y + r0, (int) out) != CUBLAS_STATUS_SUCCESS) {
                err = "verify glm5 prefill: cublasSgemm failed";
                return false;
            }
        }
        return true;
    };
    // 2026-10-10: the shared expert / a dense layer's FFN through gemm_q8 (MMQ) a block of rows at a time instead of
    // shared_expert_multi on slices of <= 8 tokens (12,795 tokens: the router + shared expert step took 10.6 s of the
    // walk); gate and up into gscratch, the clamped swiglu, down into Y.  STRATA_GLM5_PF_FFN_MMQ=0: the slices.
    static const bool pf_ffn_mmq = [] { const char* v = std::getenv("STRATA_GLM5_PF_FFN_MMQ"); return !(v && v[0] == '0'); }();
    ffn_q8 = [&](int64_t l, bool dense, const float* X, int64_t T, float* Y) -> bool {
        if (!pf_ffn_mmq || !mq || wscr == nullptr || gscratch == nullptr) return false;
        const WeightRef *wg = wfind(wt, l, dense ? "ffn_gate.weight" : "ffn_gate_shexp.weight", err),
                        *wu = wfind(wt, l, dense ? "ffn_up.weight" : "ffn_up_shexp.weight", err),
                        *wd = wfind(wt, l, dense ? "ffn_down.weight" : "ffn_down_shexp.weight", err);
        if (!wg || !wu || !wd || wg->native_type != 8 || wu->native_type != 8 || wd->native_type != 8) { err.clear(); return false; }
        const int64_t F = dense ? m.NFFD : m.NFF;
        const int64_t RB = std::min<int64_t>(T, F > 4096 ? 256 : 1024);
        if ((size_t) 2 * RB * F * 4 > gscratch_bytes) return false;
        float* G = (float*) gscratch;
        float* U = G + (size_t) RB * F;
        for (int64_t r0 = 0; r0 < T; r0 += RB) {
            const int64_t rb = std::min<int64_t>(RB, T - r0);
            if (!gemm_q8(wg, N, F, X + (size_t) r0 * N, rb, G) || !gemm_q8(wu, N, F, X + (size_t) r0 * N, rb, U)) return false;
            glm5_swiglu2(G, U, rb * F, m.clamp_sh, cs);
            if (!gemm_q8(wd, F, N, G, rb, Y + (size_t) r0 * N)) return false;
        }
        return true;
    };
    // stage 2: one KDA layer over the chunk: per sub-batch the mHC reads (a slice at a time), the projections at once,
    // the conv / gates / recurrence a slice at a time (the window's kernels and order), the output projection at once
    auto kda_layer = [&](int64_t l, int64_t T) -> bool {
        const int64_t gi = m.kda_index[(size_t) l];
        const WeightRef *wq = wfind(wt, l, "attn_q.weight", err), *wk = wfind(wt, l, "attn_k.weight", err),
                        *wv = wfind(wt, l, "attn_v.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                        *wfa = wfind(wt, l, "ssm_f_a.weight", err), *wfb = wfind(wt, l, "ssm_f_b.weight", err),
                        *wga = wfind(wt, l, "ssm_g_a.weight", err), *wgb = wfind(wt, l, "ssm_g_b.weight", err),
                        *wbe = wfind(wt, l, "ssm_beta.weight", err), *wdt = wfind(wt, l, "ssm_dt.bias", err),
                        *wsa = wfind(wt, l, "ssm_a", err), *wnm = wfind(wt, l, "ssm_norm.weight", err);
        if (!wq || !wk || !wv || !wo || !wfa || !wfb || !wga || !wgb || !wbe || !wdt || !wsa || !wnm) return false;
        float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
        float* conv = state + (uint64_t) m.S * m.H * m.S;
        float* qkv = qkv_L_ + (size_t) gi * MT * m.C;
        float* hb = h_L_ + (size_t) gi * MT * m.C;
        float* gate = m.gate + (size_t) gi * MT * m.H * m.S;
        float* beta = beta_L_ + (size_t) gi * MT * m.H;
        const int64_t DI = m.DI, H = m.H;
        for (int64_t b0 = 0; b0 < T; b0 += KSB) {
            const int64_t nb = std::min<int64_t>(KSB, T - b0);
            if (!mhc_read_all(l, "attn", l > 0, b0, nb, "attn_norm.weight")) return false;
            const float* X = xmc + (size_t) b0 * N;
            if (!gemm_q8(wq, N, DI, X, nb, kq) || !gemm_q8(wk, N, DI, X, nb, kk) || !gemm_q8(wv, N, DI, X, nb, kv_) ||
                !gemm_q8(wfa, N, 128, X, nb, kfa) || !gemm_q8(wga, N, 128, X, nb, kga) ||
                !gemm_q8(wbe, N, H, X, nb, kbl) || !gemm_q8(wfb, 128, DI, kfa, nb, kf) || !gemm_q8(wgb, 128, DI, kga, nb, kz))
                return false;
            const size_t st_n = (size_t) m.S * H * m.S, cv_n = (size_t) m.C * 3;
            const bool chunked = kh != nullptr;
            const bool checking = chunked && pf_kda_check;
            if (checking) {   // the slice path below runs on copies; the chunked one then starts from the same state
                cudaMemcpyAsync(ck_state, state, st_n * 4, cudaMemcpyDeviceToDevice, cs);
                cudaMemcpyAsync(ck_conv, conv, cv_n * 4, cudaMemcpyDeviceToDevice, cs);
            }
            if (!chunked || checking) {
                for (int64_t s0 = 0; s0 < nb; s0 += SL) {
                    const int n = (int) std::min<int64_t>(SL, nb - s0);
                    glm5_pack3(kq + (size_t) s0 * DI, kk + (size_t) s0 * DI, kv_ + (size_t) s0 * DI, qkv, (int) DI, n, cs);
                    gdn_conv_l2_multi(conv, qkv, m.convw + (size_t) gi * m.C * 4, hb, (int) m.C, (int) (2 * H), 1e-6f, n, cs, 0);
                    kda_gate_beta_multi(kf + (size_t) s0 * DI, (const float*) wdt->data, (const float*) wsa->data, m.lb,
                                        kbl + (size_t) s0 * H, gate, beta, (int) H, n, cs);
                    gdn_step_norm_multi(state, hb, (int) m.C, gate, beta, kz + (size_t) s0 * DI, (const float*) wnm->data,
                                        m.rms_eps, ky + (size_t) s0 * DI, (int) H, (int) H, n, nkeep + n, cs, 0, nullptr, true);
                    gdn_conv_commit(conv, qkv, (int) m.C, nkeep + n, cs);
                }
            }
            if (checking) {   // keep the slice path's results, put the state back
                cudaMemcpyAsync(ck_y, ky, (size_t) nb * DI * 4, cudaMemcpyDeviceToDevice, cs);
                cudaMemcpyAsync(ck_state_ref, state, st_n * 4, cudaMemcpyDeviceToDevice, cs);
                cudaMemcpyAsync(state, ck_state, st_n * 4, cudaMemcpyDeviceToDevice, cs);
                cudaMemcpyAsync(conv, ck_conv, cv_n * 4, cudaMemcpyDeviceToDevice, cs);
            }
            if (chunked) {   // stage 3: the conv, the gates, the recurrence and the norm over the sub-batch at once
                static const int sm_count = [] { int d = 0, c = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, d); return c; }();
                glm5_kda_conv_chunk(conv, kq, kk, kv_, m.convw + (size_t) gi * m.C * 4, kh, (int) DI, (int) (2 * H), 1e-6f,
                                    (int) nb, cs);
                glm5_kda_gate_beta_chunk(kf, (const float*) wdt->data, (const float*) wsa->data, m.lb, kbl, kgate, kbeta,
                                         (int) H, (int) nb, cs);
                glm5_kda_chunk(kh, (int) m.C, kgate, kbeta, state, ky, (int) H, (int) nb, kv16, kws, sm_count, cs);
                glm5_kda_out_norm(ky, kz, (const float*) wnm->data, m.rms_eps, (int) H, (int) nb, cs);
            }
            if (checking) {   // max |chunked - slice| over the outputs and the state, against the largest |slice|
                std::vector<float> a((size_t) nb * DI), b((size_t) nb * DI), sa(st_n), sb(st_n);
                cudaStreamSynchronize(cs);
                cudaMemcpy(a.data(), ky, a.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(), ck_y, b.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(sa.data(), state, st_n * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(sb.data(), ck_state_ref, st_n * 4, cudaMemcpyDeviceToHost);
                auto cmp = [](const std::vector<float>& x, const std::vector<float>& r, double& dmax, double& rmax, double& rms) {
                    dmax = rmax = 0; double se = 0, sr = 0;
                    for (size_t i = 0; i < x.size(); ++i) {
                        const double d = std::fabs((double) x[i] - r[i]);
                        dmax = std::max(dmax, d); rmax = std::max(rmax, std::fabs((double) r[i]));
                        se += d * d; sr += (double) r[i] * r[i];
                    }
                    rms = sr > 0 ? std::sqrt(se / sr) : 0;
                };
                double yd, yr, yrel, sd, sr, srel;
                cmp(a, b, yd, yr, yrel);
                cmp(sa, sb, sd, sr, srel);
                std::fprintf(stderr, "strata glm5 KDA check: layer %lld tokens %lld-%lld: output max|d| %.3g (max|ref| %.3g, "
                                     "rel rms %.3g); state max|d| %.3g (max|ref| %.3g, rel rms %.3g)\n", (long long) l,
                             (long long) b0, (long long) (b0 + nb - 1), yd, yr, yrel, sd, sr, srel);
            }
            if (!gemm_q8(wo, DI, N, ky, nb, boc + (size_t) b0 * N)) return false;
        }
        return true;
    };

    // a layer split: this call runs [lb, le) of one chunk; the hand-off rows are R | the last FFN's output | post | comb
    const int64_t lb = l_begin, le = l_end < 0 ? g.n_layers : l_end, HF = handoff_floats(g);
    if (lb < 0 || le > g.n_layers || lb >= le || ((lb > 0 || le < g.n_layers) && n_all > C) ||
        (lb > 0 && hand_in == nullptr) || (le < g.n_layers && hand_out == nullptr)) {
        err = "verify glm5 prefill: a layer range runs one chunk, with its hand-offs";
        return false;
    }
    // one strided copy per part: device [t][w] <-> host [t][HF] at offset `at`
    // one MLA/DSA layer over the chunk in sub-batches of mla_sb tokens (the mHC read already done for the chunk): the
    // window's kernels in the window's order, each given the whole sub-batch; Q8_0 projections through cuBLAS (others
    // a slice at a time).  The scores / selection / attention / wv_b go mla_ab tokens at a time inside (vbT reuses lat).
    static const bool mla_prof = std::getenv("STRATA_GLM5_PF_PROF") != nullptr;
    double ms_m[7] = {};   // projections, indexer scores, attention, output, key append, top-k, indexer q/w (STRATA_GLM5_PF_PROF)
    auto mla_layer = [&](int64_t l, int64_t T) -> bool {
        auto mt = std::chrono::steady_clock::now();
        auto mlap = [&](int i) {
            if (!mla_prof) return;
            cudaStreamSynchronize(cs);
            const auto t1 = std::chrono::steady_clock::now();
            ms_m[i] += std::chrono::duration<double, std::milli>(t1 - mt).count();
            mt = t1;
        };
        const WeightRef *wqa = wfind(wt, l, "attn_q_a.weight", err), *wqan = wfind(wt, l, "attn_q_a_norm.weight", err),
                        *wqb = wfind(wt, l, "attn_q_b.weight", err), *wkb = wfind(wt, l, "attn_k_b.weight", err),
                        *wvb = wfind(wt, l, "attn_v_b.weight", err), *wkv = wfind(wt, l, "attn_kv_a_mqa.weight", err),
                        *wkvn = wfind(wt, l, "attn_kv_a_norm.weight", err), *wo = wfind(wt, l, "attn_output.weight", err),
                        *wik = wfind(wt, l, "indexer.attn_k.weight", err), *wiqb = wfind(wt, l, "indexer.attn_q_b.weight", err),
                        *wikn = wfind(wt, l, "indexer.k_norm.weight", err), *wikb = wfind(wt, l, "indexer.k_norm.bias", err),
                        *wip = wfind(wt, l, "indexer.proj.weight", err), *wape = wfind(wt, l, "indexer_compressor_ape.weight", err),
                        *wig = wfind(wt, l, "indexer_compressor_gate.weight", err);
        if (!wqa || !wqan || !wqb || !wkb || !wvb || !wkv || !wkvn || !wo || !wik || !wiqb || !wikn || !wikb || !wip ||
            !wape || !wig) return false;
        if (wkb->native_type != 8 || wvb->native_type != 8) { err = "verify glm5 prefill: attn_k_b / attn_v_b are not Q8_0"; return false; }
        const int64_t qi = m.dsa_index[(size_t) l];
        const int64_t QLR = m.QLR, NH = m.NH, QK = m.QK, KVR = m.KVR, VD = m.VD, ID = m.ID, IH = m.IH;
        uint8_t* lat_pool = m.latent + (size_t) qi * m.max_cells * m.lat_row;
        float* tail = m.tail + (size_t) qi * 3 * 2 * ID;
        uint16_t* pooled = m.pooled + (size_t) qi * (m.max_cells / 4 + 1) * ID;
        if (kbvb_layer != l) {
            dequant_q8_0((const uint8_t*) wkb->native_data, kb_f, NH * KVR * QK, cs);
            dequant_q8_0((const uint8_t*) wvb->native_data, vb_f, NH * VD * KVR, cs);
            kbvb_layer = l;
        }
        auto proj = [&](const WeightRef* w, int64_t in, int64_t out, const float* X, int64_t rows, float* Y) -> bool {
            if (w->native_type == 8 && in % 32 == 0) return gemm_q8(w, in, out, X, rows, Y);
            for (int64_t r0 = 0; r0 < rows; r0 += SL)
                mmvq(w, X + (size_t) r0 * in, (int) in, Y + (size_t) r0 * out, (int) out, (int) std::min<int64_t>(SL, rows - r0), m.q8, cs);
            return true;
        };
        const float one = 1.0f, zero = 0.0f;
        const Glm5KpoolBuffers kb{tail, pooled};
        static bool types_shown = false;
        if (mla_prof && !types_shown) {
            types_shown = true;
            std::fprintf(stderr, "strata glm5 prefill: MLA weight types q_a %d kv_a %d idx_k %d idx_gate %d q_b %d idx_q_b %d out %d "
                         "(8 = Q8_0, through cuBLAS)\n", wqa->native_type, wkv->native_type, wik->native_type, wig->native_type,
                         wqb->native_type, wiqb->native_type, wo->native_type);
        }
        for (int64_t b0 = 0; b0 < T; b0 += mla_sb) {
            const int n = (int) std::min<int64_t>(mla_sb, T - b0);
            const float* X = xmc + (size_t) b0 * N;
            int32_t* st = stepc + (size_t) b0 * kStepCount;
            if (!proj(wqa, N, QLR, X, n, ma_qr) || !proj(wkv, N, KVR, X, n, ma_kv) || !proj(wik, N, ID, X, n, ma_rk) ||
                !proj(wig, N, ID, X, n, ma_rg))
                return false;
            native_qsa_rms_norm_weighted(ma_qr, (const float*) wqan->data, ma_qr, (int) QLR, n, m.rms_eps, cs);
            native_qsa_rms_norm_weighted(ma_kv, (const float*) wkvn->data, ma_kv, (int) KVR, n, m.rms_eps, cs);
            if (m.lat_q4) glm5_latent_store_q4(ma_kv, lat_pool, m.page_table, st, kStepCount, 4, (int) KVR, n, cs);
            else glm5_latent_store(ma_kv, (uint16_t*) lat_pool, m.page_table, st, kStepCount, 4, (int) KVR, n, cs);
            if (!proj(wqb, QLR, NH * QK, ma_qr, n, ma_qb)) return false;
            glm5_transpose_abd(ma_qb, ma_qT, n, (int) NH, (int) QK, cs);
            cublasSgemmStridedBatched(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) KVR, n, (int) QK, &one,
                                      kb_f, (int) QK, (long long) KVR * QK, ma_qT, (int) QK, (long long) n * QK, &zero,
                                      ma_qaT, (int) KVR, (long long) n * KVR, (int) NH);
            glm5_transpose_abd(ma_qaT, ma_qa, (int) NH, n, (int) KVR, cs);
            mlap(0);
            {   // 2026-10-10: the sub-batch's cells in two launches when they are consecutive (n launches before)
                static const bool pf_kb = [] { const char* v = std::getenv("STRATA_GLM5_PF_KAPPEND"); return !(v && v[0] == '0'); }();
                const int64_t c_first = h_steps[(size_t) b0 * kStepCount + kStepPos];
                const int64_t c_last = h_steps[(size_t) (b0 + n - 1) * kStepCount + kStepPos];
                if (pf_kb && c_last - c_first == n - 1 && c_first >= 0 && c_last < m.max_cells) {
                    glm5_kpool_append_batch(ma_rk, ma_rg, c_first, n, (const float*) wikn->data, (const float*) wikb->data,
                                            (const float*) wape->data, m.ln_eps, kb, m.max_cells, cs);
                } else {
                    for (int t = 0; t < n; ++t)
                        glm5_kpool_append(ma_rk + (size_t) t * ID, ma_rg + (size_t) t * ID, st + (size_t) t * kStepCount + kStepPos,
                                          (const float*) wikn->data, (const float*) wikb->data, (const float*) wape->data, m.ln_eps,
                                          kb, m.max_cells, cs);
                }
            }
            mlap(4);
            if (!proj(wiqb, QLR, IH * ID, ma_qr, n, ma_iq)) return false;
            {   // the indexer's head weights (F32 [32][4096]) for the group in one cublasSgemm (glm5_f32_gemv: per token)
                const float iw_scale = 1.0f / std::sqrt((float) (ID * IH));
                if (!(pf_router_blas &&
                      cublasSgemm(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) IH, n, (int) N, &iw_scale, (const float*) wip->data,
                                  (int) N, X, (int) N, &zero, ma_iw, (int) IH) == CUBLAS_STATUS_SUCCESS))
                    glm5_f32_gemv((const float*) wip->data, X, ma_iw, (int) N, (int) IH, n, iw_scale, cs);
            }
            mlap(6);
            for (int a0 = 0; a0 < n; a0 += (int) mla_ab) {   // the attention's sub-batches
                const int na = std::min<int>((int) mla_ab, n - a0);
                const int32_t* sa = st + (size_t) a0 * kStepCount;
                // the sub-batch's last token has the most blocks: the scores' grid and the selection's kernel by that count
                const int64_t active = (int64_t) h_steps[(size_t) (b0 + a0 + na - 1) * kStepCount + kStepNBid] + 1;
                static const bool score_gemm = [] { const char* v = std::getenv("STRATA_GLM5_PF_SCORE_GEMM"); return !(v && v[0] == '0'); }();
                if (score_gemm && ma_S != nullptr) {
                    glm5_f32_to_bf16(ma_iq + (size_t) a0 * IH * ID, ma_q16, (int64_t) na * IH * ID, cs);
                    for (int64_t t0 = 0; t0 < active; t0 += kScoreTile) {
                        const int tb = (int) std::min<int64_t>(kScoreTile, active - t0);
                        // S [na*32 rows][tb] (column j = query row, ld tb) = pooled[t0 .. t0+tb)^T-major . q
                        if (cublasGemmEx(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, tb, na * (int) IH, (int) ID, &one,
                                         pooled + (size_t) t0 * ID, CUDA_R_16BF, (int) ID, ma_q16, CUDA_R_16BF, (int) ID, &zero,
                                         ma_S, CUDA_R_32F, tb, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS) {
                            err = "verify glm5 prefill: the indexer's score GEMM failed";
                            return false;
                        }
                        glm5_kpool_scores_reduce(ma_S, ma_iw + (size_t) a0 * IH, sa, na, t0, tb, max_blocks_, ma_sc, cs);
                    }
                } else {
                    glm5_kpool_scores(ma_iq + (size_t) a0 * IH * ID, ma_iw + (size_t) a0 * IH, pooled, sa, na, max_blocks_, ma_sc, cs,
                                      pf_active ? active : -1);
                }
                mlap(1);
                qsa_block_topk(ma_sc, sa, na, max_blocks_, cap_, s, ma_sel, cs, pf_active ? active : -1);
                mlap(5);
                const float* qa = ma_qa + (size_t) a0 * NH * KVR;
                if (m.lat_q4) mla_decode_attn_batch_q4(qa, lat_pool, m.page_table, 4, ma_sel, sa, cap_, (int) NH, 1.0f / std::sqrt((float) QK), ma_scr, ma_lat, na, cs);
                else mla_decode_attn_batch(qa, (const uint16_t*) lat_pool, m.page_table, 4, ma_sel, sa, cap_, (int) NH,
                                           1.0f / std::sqrt((float) QK), ma_scr, ma_lat, na, cs);
                mlap(2);
                glm5_transpose_abd(ma_lat, ma_latT, na, (int) NH, (int) KVR, cs);
                float* vbT = ma_lat;   // [NH][na][VD] (VD <= KVR)
                cublasSgemmStridedBatched(pf_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) VD, na, (int) KVR, &one,
                                          vb_f, (int) KVR, (long long) VD * KVR, ma_latT, (int) KVR, (long long) na * KVR, &zero,
                                          vbT, (int) VD, (long long) na * VD, (int) NH);
                glm5_transpose_abd(vbT, ma_vb + (size_t) a0 * NH * VD, (int) NH, na, (int) VD, cs);
                mlap(3);
            }
            if (!proj(wo, NH * VD, N, ma_vb, n, boc + (size_t) b0 * N)) return false;
            mlap(3);
        }
        return true;
    };
    auto hand_copy = [&](float* dev, const float* host_in, float* host_out, int64_t at, int64_t w, int64_t T) {
        if (host_in) cudaMemcpy2DAsync(dev, (size_t) w * 4, host_in + at, (size_t) HF * 4, (size_t) w * 4, (size_t) T,
                                       cudaMemcpyHostToDevice, cs);
        else cudaMemcpy2DAsync(host_out + at, (size_t) HF * 4, dev, (size_t) w * 4, (size_t) w * 4, (size_t) T,
                               cudaMemcpyDeviceToHost, cs);
    };
    auto hand_parts = [&](const float* hin, float* hout, int64_t T) {
        hand_copy(Rc, hin, hout, 0, HC * N, T);
        hand_copy(boc, hin, hout, HC * N, N, T);
        hand_copy(postF, hin, hout, HC * N + N, HC, T);
        hand_copy(combF, hin, hout, HC * N + N + HC, HC * HC, T);
    };
    // STRATA_GLM5_PF_PROF=1 (2026-10-09): sync between the parts and print where a prompt's time goes
    static const bool prof = std::getenv("STRATA_GLM5_PF_PROF") != nullptr;
    double ms_attn = 0, ms_norm = 0, ms_ffn = 0;
    double w_ms[4] = {};   // the walk's FFN side: state in, route, state out, routed experts
    const double wait0 = pf_wait_ms, fetch0 = pf_fetch_ms, blobt0 = pf_blob_thread_ms, copyt0 = pf_copy_thread_ms;
    const double copy0 = prefill_ms_copy;
    auto lap = [&](double& acc, std::chrono::steady_clock::time_point& t0) {
        if (!prof) return;
        cudaStreamSynchronize(cs);
        const auto t1 = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t0 = t1;
    };
    // ---- the layer walk (2026-10-09, after Infernix's prefill_walk): a prompt of more than one chunk goes layer by layer
    // over all its chunks, so each layer's experts are streamed ONCE per group of PG tokens instead of once per chunk
    // (4,400 tokens in 4,096 + 304: 19,835 blobs instead of ~10,700).  Between layers a chunk's state (the hand-off
    // layout: R, the block output, post/comb) waits in page-locked host memory.  Per chunk: attention, the FFN's mHC
    // read, the router and the shared expert (into boc), the q8_1 inputs to the host; per layer then: every routed
    // expert once over the group's tokens, its outputs added to the shared ones in expert order (another summation
    // order than native_moe_combine_multi: not bitwise the chunk path).  Group: PG <= C * HC tokens (the sums live
    // in Rc); the q8_1 inputs, the per-token entry lists and the expert outputs share `parts`.
    // One PC only (layers 0..n), STRATA_GLM5_PF_WALK=0: the chunk path.
    static const bool pf_walk = [] { const char* v = std::getenv("STRATA_GLM5_PF_WALK"); return !(v && v[0] == '0'); }();
    // PG halves (in whole chunks) until its page-locked state fits beside the pinned RAM tier (12,795 tokens' 1 GiB did
    // not beside 99 GiB, 2026-10-09)
    static const int64_t pg_max = [] { const char* v = std::getenv("STRATA_GLM5_PF_GROUP_MAX"); return v ? std::atoll(v) : 0; }();
    int64_t PG = std::min<int64_t>(n_all, pg_max > 0 ? std::min<int64_t>(pg_max, C * HC) : C * HC);
    const size_t q8row = (size_t) (N / 32) * 36;
    int64_t q8f = 0, E_cap = 0;
    const int64_t csrf = (int64_t) ((((size_t) 2 * (4 * CK + 1)) + 63) & ~(size_t) 63);  // 2 halves of the entry lists
    float* hostS = nullptr;
    uint8_t* hq8 = nullptr;
    int32_t* hids = nullptr;
    float* hw = nullptr;
    int32_t* pcsr = nullptr;   // page-locked [2][u_tok E | u_off E+1 | e_row E | e_w E] (E = CK)
    struct HostFree {
        void* p[5];
        ~HostFree() { for (void* q : p) if (q) cudaFreeHost(q); }
    } host_free{{nullptr, nullptr, nullptr, nullptr, nullptr}};
    bool walk = pf_walk && lb == 0 && le == g.n_layers && n_all > C;
    if (walk) {
        auto halloc = [&](size_t bytes, int i) -> void* {
            void* p = nullptr;
            if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); return nullptr; }
            host_free.p[i] = p;
            return p;
        };
        auto hfree = [&] { for (void*& q : host_free.p) { if (q) cudaFreeHost(q); q = nullptr; } };
        walk = false;
        // (one chunk smaller a try, not half: 31,883 tokens got 4 groups of 8,192 where 12,288 fits beside 98 GiB)
        for (; PG >= 2 * C || PG == n_all; PG = std::max<int64_t>(C, (PG - 1) / C * C)) {
            // floats at the front of parts: the group's FFN inputs (F32, MMQ quantizes its own rows) and their q8_1
            // image (the native kernels' layers)
            q8f = (int64_t) (((size_t) PG * N + (size_t) PG * q8row / 4 + 63) & ~(size_t) 63);
            E_cap = std::min<int64_t>(GE, (CK * N - q8f - csrf) / N);           // expert output rows a launch
            hostS = (float*) halloc((size_t) PG * HF * 4, 0);
            hq8 = (uint8_t*) halloc((size_t) PG * N * 4, 1);   // the FFN inputs, F32
            hids = (int32_t*) halloc((size_t) PG * K * 4, 2);
            hw = (float*) halloc((size_t) PG * K * 4, 3);
            pcsr = (int32_t*) halloc((size_t) 2 * (4 * CK + 1) * 4, 4);
            if (hostS && hq8 && hids && hw && pcsr && E_cap >= std::min<int64_t>(PG, C)) { walk = true; break; }
            hfree();
            if (PG <= 2 * C) break;
        }
        if (std::getenv("STRATA_GLM5_PF_PROF") != nullptr)
            std::fprintf(stderr, "strata glm5 prefill: layer walk %s (groups of %lld tokens, %lld expert rows a launch, %zu MiB "
                         "page-locked)\n", walk ? "on" : "off (no page-locked memory)", (long long) PG, (long long) E_cap,
                         ((size_t) PG * HF * 4 + (size_t) PG * N * 4 + (size_t) PG * K * 8) >> 20);
    }
    // walk: the chunk's router, shared expert (into boc) and q8_1 inputs; ids / weights / inputs to the host at token off
    auto moe_route = [&](int64_t l, int64_t T, int64_t off) -> bool {
        const WeightRef *wr = wfind(wt, l, "ffn_gate_inp.weight", err), *wb = wfind(wt, l, "exp_probs_b.bias", err),
                        *wsg = wfind(wt, l, "ffn_gate_shexp.weight", err), *wsu = wfind(wt, l, "ffn_up_shexp.weight", err),
                        *wsd = wfind(wt, l, "ffn_down_shexp.weight", err);
        if (!wr || !wb || !wsg || !wsu || !wsd) return false;
        router_logits(wr, T);
        router_sigmoid_topk(logc, (const float*) wb->data, (int) T, (int) NE, (int) K, m.wscale, idsc, wc, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data; nsw.up_type = wsu->native_type;
        nsw.up_data = wsu->native_data; nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data; nsw.q8_1 = m.q8;
        if (!ffn_q8(l, false, xmc, T, boc))
            for (int64_t s0 = 0; s0 < T; s0 += SL) {
                const int n = (int) std::min<int64_t>(SL, T - s0);
                shared_expert_multi(n, xmc + (size_t) s0 * N, nullptr, nsw, nullptr, sh_gate_, sh_up_, sh_g_,
                                    boc + (size_t) s0 * N, N, m.NFF, cs, nullptr, 0, m.clamp_sh);
            }
        cudaMemcpyAsync(hids + (size_t) off * K, idsc, (size_t) T * K * 4, cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(hw + (size_t) off * K, wc, (size_t) T * K * 4, cudaMemcpyDeviceToHost, cs);
        cudaMemcpyAsync(hq8 + (size_t) off * N * 4, xmc, (size_t) T * N * 4, cudaMemcpyDeviceToHost, cs);
        return true;
    };
    // walk: layer l's routed experts over the group's Pn tokens, each streamed once; the sums (shared + routed, in the
    // host's block-output slots) go through Rc
    std::vector<std::pair<int32_t, int32_t>> ent;   // (token, entry) of a launch, sorted by token
    double w_plan_ms = 0, w_enq_ms = 0;   // STRATA_GLM5_PF_PROF: the walk's host-side parts
    auto moe_walk = [&](int64_t l, int64_t Pn) -> bool {
        touch_next(l + 1, Pn);
        if (cudaStreamSynchronize(cs) != cudaSuccess) { err = "verify glm5 prefill: the router failed on the GPU"; return false; }
        float* acc = Rc;
        float* xa = parts;                                       // the group's FFN inputs
        uint8_t* q8a = (uint8_t*) (parts + (size_t) PG * N);     // their q8_1 image (native layers)
        int32_t* dcsr = (int32_t*) (parts + q8f);
        float* rows = parts + q8f + csrf;
        const bool mm = use_mmq(l);
        cudaMemcpyAsync(xa, hq8, (size_t) Pn * N * 4, cudaMemcpyHostToDevice, cs);
        if (!mm) quantize_q8_1_rows(xa, Pn, N, q8a, cs);
        cudaMemcpyAsync(acc, hostS + (size_t) PG * HC * N, (size_t) Pn * N * 4, cudaMemcpyHostToDevice, cs);   // the block outputs
        for (auto& v : by) v.clear();
        used.clear();
        const bool count_use = pf_usage.size() == (size_t) g.n_layers * NE;
        for (int64_t i = 0; i < Pn * K; ++i) {
            const int32_t e = hids[(size_t) i];
            if (e < 0 || e >= NE) continue;
            if (count_use) pf_usage[(size_t) l * NE + e] += 1.0f;
            if (by[(size_t) e].empty()) used.push_back(e);
            by[(size_t) e].push_back((int32_t) i);
        }
        const auto& fm = lay.fmt[(size_t) l];
        NativeExpertLayout L = native_expert_layout(fm.gu_type, fm.d_type, fm.n_embd, fm.n_ff);
        L.swiglu_limit = m.clamp_exp;
        const uint64_t bb = lay.blob_bytes(l);
        const size_t half_i = (size_t) 4 * CK + 1;
        // an expert routed by more of the group's tokens than a launch holds (E_cap; a 16,384-token group can give one
        // over 8,192) goes in rounds: round r takes entries [r * E_cap, (r + 1) * E_cap) of every expert (the rest
        // of the loop below unchanged; such an expert is streamed once a round)
        std::vector<std::vector<int32_t>> by_full;
        std::vector<int32_t> used_full;
        bool split = false;
        for (int32_t e : used) split = split || (int64_t) by[(size_t) e].size() > E_cap;
        if (split) {
            used_full = used;
            by_full.resize((size_t) NE);
            for (int32_t e : used_full) by_full[(size_t) e] = by[(size_t) e];
        }
        for (int64_t rnd = 0;; ++rnd) {
        if (split) {
            used.clear();
            for (int32_t e : used_full) {
                const auto& fl = by_full[(size_t) e];
                const size_t lo = std::min(fl.size(), (size_t) (rnd * E_cap)),
                             hi = std::max(lo, std::min(fl.size(), (size_t) ((rnd + 1) * E_cap)));
                by[(size_t) e].assign(fl.begin() + (ptrdiff_t) lo, fl.begin() + (ptrdiff_t) hi);
                if (!by[(size_t) e].empty()) used.push_back(e);
            }
            if (used.empty()) break;
        }
        for (size_t bi = 0, b0 = 0; b0 < used.size(); ++bi) {
            const int hf = (int) (bi & 1);
            int nb = 0;
            int64_t ne_ = 0;
            while (b0 + (size_t) nb < used.size() && nb < B &&
                   ne_ + (int64_t) by[(size_t) used[b0 + (size_t) nb]].size() <= E_cap)
                ne_ += (int64_t) by[(size_t) used[b0 + (size_t) nb++]].size();
            if (nb == 0) { err = "verify glm5 prefill: an expert's tokens exceed the walk's launch"; return false; }
            uint8_t* dstage = stage + (size_t) hf * B * max_blob;
            uint8_t* hstage = pin + (size_t) hf * B * max_blob;
            int32_t* start = pplan + (size_t) hf * plan_n;
            int32_t* dst = start + (B + 1);
            int32_t* tok = dst + CK;
            int32_t* count = tok + CK;
            unsigned long long* hptr = pptr + (size_t) hf * B;
            int32_t* hc = pcsr + (size_t) hf * half_i;   // u_tok | u_off | e_row | e_w
            int32_t *u_tok = hc, *u_off = hc + CK, *e_row = hc + 2 * CK + 1;
            float* e_w = (float*) (hc + 3 * CK + 1);
            const auto tc = std::chrono::steady_clock::now();
            cudaEventSynchronize(ev_copied[hf]);
            cudaEventSynchronize(ev_done[hf]);
            pf_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            const auto tpl = std::chrono::steady_clock::now();
            int ns = 0;
            int32_t stream_e[32];
            int entries = 0, max_rows = 0;
            ent.clear();
            int ord[32], no = 0;   // the streamed experts first, then the residents (as in moe)
            for (int pass = 0; pass < 2; ++pass)
                for (int j = 0; j < nb; ++j) {
                    const int32_t e = used[b0 + (size_t) j];
                    const bool res = f.host_res && f.host_res[(size_t) l * NE + e] >= 0;
                    if (res == (pass == 1)) ord[no++] = j;
                }
            for (int q = 0; q < nb; ++q) {
                const int32_t e = used[b0 + (size_t) ord[q]];
                const int32_t slot = f.host_res ? f.host_res[(size_t) l * NE + e] : -1;
                if (slot >= 0) {
                    const uint8_t* src = f.cache_base + (f.slot_off ? (size_t) f.slot_off[slot] : (size_t) slot * f.cache_blob);
                    if (mm) {
                        cudaMemcpyAsync(dstage + (size_t) q * max_blob, src, (size_t) bb, cudaMemcpyDeviceToDevice, cs);
                        hptr[q] = (unsigned long long) (dstage + (size_t) q * max_blob);
                    } else {
                        hptr[q] = (unsigned long long) src;
                    }
                    ++prefill_resident;
                } else {
                    hptr[q] = (unsigned long long) (dstage + (size_t) ns * max_blob);
                    stream_e[ns++] = e;
                    ++prefill_streamed;
                }
                start[q] = entries;
                max_rows = std::max<int>(max_rows, (int) by[(size_t) e].size());
                for (int32_t i : by[(size_t) e]) {
                    dst[entries] = entries; tok[entries] = (int32_t) (i / K);
                    e_w[entries] = hw[(size_t) i];
                    ent.emplace_back((int32_t) (i / K), entries);
                    ++entries;
                }
            }
            start[nb] = entries;
            count[0] = nb;
            std::stable_sort(ent.begin(), ent.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            int nu = 0;
            std::vector<float> w_sorted((size_t) entries);
            for (int j = 0; j < entries; ++j) {
                if (j == 0 || ent[(size_t) j].first != ent[(size_t) j - 1].first) { u_tok[nu] = ent[(size_t) j].first; u_off[nu++] = j; }
                e_row[j] = ent[(size_t) j].second;
                w_sorted[(size_t) j] = e_w[ent[(size_t) j].second];
            }
            u_off[nu] = entries;
            std::memcpy(e_w, w_sorted.data(), (size_t) entries * 4);
            w_plan_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tpl).count();
            if (ns > 0) {
                bool failed = false;
                std::vector<std::thread> th;
                th.reserve((size_t) ns);
                std::atomic<int64_t> us_blob{0}, us_copy{0};
                const auto tf = std::chrono::steady_clock::now();
                std::atomic<bool> direct[32];
                for (int k = 0; k < ns; ++k) direct[k] = false;
                for (int k = 0; k < ns; ++k)
                    th.emplace_back([&, k] {
                        const auto a = std::chrono::steady_clock::now();
                        const uint8_t* hb = f.blob ? f.blob(f.ctx, l, stream_e[k]) : nullptr;
                        const auto b = std::chrono::steady_clock::now();
                        if (hb == nullptr) { failed = true; return; }
                        cudaPointerAttributes at{};
                        if (pf_direct && cudaPointerGetAttributes(&at, hb) == cudaSuccess && at.type == cudaMemoryTypeHost) {
                            hsrc[k] = hb;
                            direct[k] = true;
                        } else {
                            cudaGetLastError();
                            std::memcpy(hstage + (size_t) k * max_blob, hb, (size_t) bb);
                        }
                        us_blob += std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
                        us_copy += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - b).count();
                    });
                for (auto& t : th) t.join();
                pf_fetch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
                pf_blob_thread_ms += (double) us_blob.load() / 1000.0;
                pf_copy_thread_ms += (double) us_copy.load() / 1000.0;
                if (failed) { err = "verify glm5 prefill: an expert of layer " + std::to_string(l) + " is in no tier"; return false; }
                for (int k = 0; k < ns;) {
                    if (direct[k]) {
                        cudaMemcpyAsync(dstage + (size_t) k * max_blob, hsrc[k], (size_t) bb, cudaMemcpyHostToDevice, copy_s);
                        ++prefill_direct;
                        ++k;
                        continue;
                    }
                    int e = k + 1;
                    while (e < ns && !direct[e]) ++e;
                    cudaMemcpyAsync(dstage + (size_t) k * max_blob, hstage + (size_t) k * max_blob,
                                    (size_t) (e - k - 1) * max_blob + bb, cudaMemcpyHostToDevice, copy_s);
                    k = e;
                }
            }
            cudaEventRecord(ev_copied[hf], copy_s);
            prefill_ms_copy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
            int32_t* dpl = dplan + (size_t) hf * plan_n;
            unsigned long long* dp = dptr + (size_t) hf * B;
            int32_t* dc = dcsr + (size_t) hf * half_i;
            const auto tq = std::chrono::steady_clock::now();
            cudaMemcpyAsync(dpl, start, (size_t) (nb + 1) * 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dpl + (B + 1), dst, (size_t) entries * 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dpl + (B + 1) + CK, tok, (size_t) entries * 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dpl + (B + 1) + 2 * CK, count, 4, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dp, hptr, (size_t) nb * 8, cudaMemcpyHostToDevice, cs);
            cudaMemcpyAsync(dc, hc, half_i * 4, cudaMemcpyHostToDevice, cs);
            cudaStreamWaitEvent(cs, ev_copied[hf], 0);
            if (mm)
                run_mmq(l, dstage, nb, entries, max_rows, dpl, dpl + (B + 1) + CK, xa, N, dpl + (B + 1), rows);
            else
                native_expert_grouped(L, dp, dpl, dpl + (B + 1) + 2 * CK, dpl + (B + 1), dpl + (B + 1) + CK, B, GE,
                                      q8a, gscratch, rows, cs);
            glm5_scatter_add(rows, dc, dc + CK, dc + 2 * CK + 1, (const float*) (dc + 3 * CK + 1), nu, acc, (int) N, cs);
            cudaEventRecord(ev_done[hf], cs);
            w_enq_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq).count();
            b0 += (size_t) nb;
        }
        if (!split) break;
        }   // the rounds
        cudaMemcpyAsync(hostS + (size_t) PG * HC * N, acc, (size_t) Pn * N * 4, cudaMemcpyDeviceToHost, cs);
        kbvb_layer = -1;   // parts (where kb_f / vb_f are carved) held this layer's expert outputs
        return true;
    };
    // the chunk's steps on the host (h_steps) and the device (stepc); its tokens' embedding into Rc when `embed`
    auto chunk_begin = [&](int64_t c0, int64_t T, bool embed) -> bool {
        for (int64_t t = 0; t < T; ++t) qsa_step_fill(h_steps.data() + (size_t) t * kStepCount, pos0 + c0 + t, s);
        cudaMemcpyAsync(stepc, h_steps.data(), (size_t) T * kStepCount * 4, cudaMemcpyHostToDevice, cs);
        if (embed) {
            cudaMemcpyAsync(tokc, tokens + c0, (size_t) T * 4, cudaMemcpyHostToDevice, cs);
            const NativeEmbed* ne = native_embed();
            if (ne == nullptr) { err = "verify glm5 prefill: needs the native token embedding"; return false; }
            ne->gather_dev(tokc, T, embc, cs);
            // 2026-10-11: an image's cells take strata-vision's rows instead (GLM-5.3 has no RoPE: nothing else changes)
            for (int64_t t = 0; t < T; ++t) {
                const int64_t p = pos0 + c0 + t;
                if (p < (int64_t) glm5_img_rows.size() && glm5_img_rows[(size_t) p] != nullptr)
                    cudaMemcpyAsync(embc + (size_t) t * N, glm5_img_rows[(size_t) p], (size_t) N * 4, cudaMemcpyHostToDevice, cs);
            }
            broadcast_streams(embc, Rc, N, (int) HC, (int) T, cs);
        }
        return true;
    };
    // layer l's attention block and the FFN's mHC read for the chunk in Rc (xmc = the FFN input after it)
    auto attn_side = [&](int64_t l, int64_t T, std::chrono::steady_clock::time_point& tp) -> bool {
        if (kda_batch && m.kda_index[(size_t) l] >= 0) {
            if (!kda_layer(l, T)) return false;
        } else if (mhc_part != nullptr) {
            if (!mhc_read_all(l, "attn", l > 0, 0, T, "attn_norm.weight")) return false;
            lap(ms_norm, tp);
            if (mla_sb > 0 && m.dsa_index[(size_t) l] >= 0) {
                if (!mla_layer(l, T)) return false;
            } else {
                for (int64_t s0 = 0; s0 < T; s0 += SL)
                    if (!attn(l, s0, (int) std::min<int64_t>(SL, T - s0))) return false;
            }
        } else {
            for (int64_t s0 = 0; s0 < T; s0 += SL) {
                const int n = (int) std::min<int64_t>(SL, T - s0);
                if (!mhc_read(l, "attn", l > 0, s0, n, "attn_norm.weight") || !attn(l, s0, n)) return false;
            }
        }
        lap(ms_attn, tp);
        if (!mhc_read_all(l, "ffn", true, 0, T, "ffn_norm.weight")) return false;
        lap(ms_norm, tp);
        return true;
    };
    auto dense_ffn = [&](int64_t l, int64_t T) -> bool {
        const WeightRef *wg = wfind(wt, l, "ffn_gate.weight", err), *wu = wfind(wt, l, "ffn_up.weight", err),
                        *wd = wfind(wt, l, "ffn_down.weight", err);
        if (!wg || !wu || !wd) return false;
        NativeSharedWeights nw;
        nw.gate_type = wg->native_type; nw.gate_data = wg->native_data; nw.up_type = wu->native_type;
        nw.up_data = wu->native_data; nw.down_type = wd->native_type; nw.down_data = wd->native_data; nw.q8_1 = m.q8;
        if (!ffn_q8(l, true, xmc, T, boc))
            for (int64_t s0 = 0; s0 < T; s0 += SL) {
                const int n = (int) std::min<int64_t>(SL, T - s0);
                shared_expert_multi(n, xmc + (size_t) s0 * N, nullptr, nw, nullptr, m.dg, m.du, sh_g_,
                                    boc + (size_t) s0 * N, N, m.NFFD, cs, nullptr, 0, m.clamp_sh);
            }
        return true;
    };
    // the walk's host state as four contiguous arrays (R | block outputs | post | comb, PG rows each): one plain copy
    // each a chunk - the hand-off layout's 2-D copies moved post / comb 16 and 64 bytes a row (31,883 tokens: the
    // state's traffic took 7.8 s)
    // parts: 1 R, 2 the block outputs, 4 post / comb (the state out splits them over two streams)
    auto walk_parts = [&](int64_t off, int64_t T, bool in, int parts_ = 7, cudaStream_t st = nullptr) {
        if (st == nullptr) st = cs;
        float* hR = hostS;
        float* hB = hR + (size_t) PG * HC * N;
        float* hP = hB + (size_t) PG * N;
        float* hC = hP + (size_t) PG * HC;
        auto cp = [&](float* dev, float* host, size_t n) {
            if (in) cudaMemcpyAsync(dev, host, n * 4, cudaMemcpyHostToDevice, st);
            else cudaMemcpyAsync(host, dev, n * 4, cudaMemcpyDeviceToHost, st);
        };
        if (parts_ & 1) cp(Rc, hR + (size_t) off * HC * N, (size_t) T * HC * N);
        if (parts_ & 2) cp(boc, hB + (size_t) off * N, (size_t) T * N);
        if (parts_ & 4) {
            cp(postF, hP + (size_t) off * HC, (size_t) T * HC);
            cp(combF, hC + (size_t) off * HC * HC, (size_t) T * HC * HC);
        }
    };
    // 2026-10-10: R, post and comb are final once the attention side is done - their copy out (on copy_s, idle between
    // the walks) runs beside the router and the shared expert, which touch neither; the block outputs follow on cs.
    // STRATA_GLM5_PF_STATE_OVL=0: all four on cs after the router
    static const bool state_ovl = [] { const char* v = std::getenv("STRATA_GLM5_PF_STATE_OVL"); return !(v && v[0] == '0'); }();
    cudaEvent_t ev_att = nullptr, ev_out = nullptr;
    cudaEventCreateWithFlags(&ev_att, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_out, cudaEventDisableTiming);
    struct EvGuard { cudaEvent_t a, b; ~EvGuard() { cudaEventDestroy(a); cudaEventDestroy(b); } } ev_guard{ev_att, ev_out};
    const double setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    try {
        if (walk) {
            int64_t last_c0 = 0, last_T = 0, last_g0 = 0;
            for (int64_t g0 = 0; g0 < n_all; g0 += PG) {
                const int64_t Pn = std::min<int64_t>(PG, n_all - g0);
                for (int64_t l = lb; l < le; ++l) {
                    for (int64_t c0 = g0; c0 < g0 + Pn; c0 += C) {
                        const int64_t T = std::min<int64_t>(C, g0 + Pn - c0);
                        float* hs = hostS + (size_t) (c0 - g0) * HF;
                        auto tp = std::chrono::steady_clock::now();
                        if (!chunk_begin(c0, T, l == lb)) return false;
                        (void) hs;
                        if (l > lb) walk_parts(c0 - g0, T, true);
                        double f0 = ms_ffn;
                        lap(ms_ffn, tp);
                        w_ms[0] += ms_ffn - f0;   // the state in (and the steps, the embedding)
                        if (!attn_side(l, T, tp)) return false;
                        if (state_ovl) {
                            cudaEventRecord(ev_att, cs);
                            cudaStreamWaitEvent(copy_s, ev_att, 0);
                            walk_parts(c0 - g0, T, false, 1 | 4, copy_s);
                            cudaEventRecord(ev_out, copy_s);
                        }
                        if (l < g.n_dense_layers ? !dense_ffn(l, T) : !moe_route(l, T, c0 - g0)) return false;
                        f0 = ms_ffn;
                        lap(ms_ffn, tp);
                        w_ms[1] += ms_ffn - f0;   // router, shared expert, inputs out
                        if (state_ovl) {
                            walk_parts(c0 - g0, T, false, 2);
                            cudaStreamWaitEvent(cs, ev_out, 0);   // Rc / postF / combF are rewritten next
                        } else {
                            walk_parts(c0 - g0, T, false);
                        }
                        f0 = ms_ffn;
                        lap(ms_ffn, tp);
                        w_ms[2] += ms_ffn - f0;   // the state out
                        last_c0 = c0; last_T = T; last_g0 = g0;
                    }
                    if (l >= g.n_dense_layers) {
                        auto tp = std::chrono::steady_clock::now();
                        const double f0 = ms_ffn;
                        if (!moe_walk(l, Pn)) return false;
                        lap(ms_ffn, tp);
                        w_ms[3] += ms_ffn - f0;   // the routed experts
                    }
                }
            }
            walk_parts(last_c0 - last_g0, last_T, true);   // the caller reads the last chunk
        } else {
            for (int64_t c0 = 0; c0 < n_all; c0 += C) {
                const int64_t T = std::min<int64_t>(C, n_all - c0);
                auto tp = std::chrono::steady_clock::now();
                if (!chunk_begin(c0, T, lb == 0)) return false;
                if (lb > 0) hand_parts(hand_in, nullptr, T);
                lap(ms_ffn, tp);
                for (int64_t l = lb; l < le; ++l) {
                    if (!attn_side(l, T, tp)) return false;
                    if (l < g.n_dense_layers ? !dense_ffn(l, T) : !moe(l, T)) return false;
                    lap(ms_ffn, tp);
                }
                if (le < g.n_layers) hand_parts(nullptr, hand_out, T);   // synced below, before the caller sends it
            }
        }
    } catch (const std::exception& e) {
        err = std::string("verify glm5 prefill: ") + e.what();
        return false;
    }
    const cudaError_t se = cudaStreamSynchronize(cs);
    if (se != cudaSuccess) { err = std::string("verify glm5 prefill: ") + cudaGetErrorString(se); return false; }
    // STRATA_GLM5_PF_DUMP=path (2026-10-10, a check): the last token's R (4 x N) and block output (N), F32, appended
    if (const char* dp = std::getenv("STRATA_GLM5_PF_DUMP"); dp != nullptr && le == g.n_layers) {
        const int64_t Tl = (n_all - 1) % C + 1;   // the last chunk's tokens (both paths leave it in Rc / boc)
        std::vector<float> h((size_t) (HC + 1) * N);
        cudaMemcpy(h.data(), Rc + (size_t) (Tl - 1) * HC * N, (size_t) HC * N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(h.data() + (size_t) HC * N, boc + (size_t) (Tl - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToHost);
        if (FILE* fp = std::fopen(dp, "ab")) { std::fwrite(h.data(), 4, h.size(), fp); std::fclose(fp); }
    }
    prefill_ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    if (prof) {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "strata glm5 prefill prof: setup (buffers, page-locked staging) %.0f ms of %.0f ms; VRAM free at the end %zu MiB\n",
                     setup_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(), fr >> 20);
    }
    if (prof)
        std::fprintf(stderr, "strata glm5 prefill prof: %lld tokens, layers %lld-%lld: attention %.0f ms, mHC/norm %.0f ms, "
                             "FFN/experts %.0f ms (host blob copies %.0f ms of it); streamed %lld (%lld straight from pinned "
                             "RAM), resident %lld so far\n",
                     (long long) n_all, (long long) lb, (long long) le - 1, ms_attn, ms_norm, ms_ffn,
                     prefill_ms_copy - copy0, (long long) prefill_streamed, (long long) prefill_direct,
                     (long long) prefill_resident);
    if (prof && !walk)
        std::fprintf(stderr, "strata glm5 prefill prof: moe - router+shared (GPU) %.0f ms, split %.0f ms, expert loop %.0f ms, "
                     "lane join + combine %.0f ms\n", pf_moe_ms[0], pf_moe_ms[1], pf_moe_ms[2], pf_moe_ms[3]);
    if (prof && pf_cpu_n > 0)
        std::fprintf(stderr, "strata glm5 prefill prof: CPU lane %lld experts, the GPU then waited %.0f ms for it\n",
                     (long long) pf_cpu_n, pf_cpu_join_ms);
    if (prof && walk)
        std::fprintf(stderr, "strata glm5 prefill prof: walk host side - entry lists %.0f ms, launches %.0f ms; state in %.0f ms, "
                     "route %.0f ms, state out %.0f ms, routed experts %.0f ms\n", w_plan_ms, w_enq_ms, w_ms[0], w_ms[1], w_ms[2], w_ms[3]);
    if (prof && mla_sb > 0)
        std::fprintf(stderr, "strata glm5 prefill prof: MLA layers %.0f ms of the attention (projections %.0f, key append %.0f, "
                             "indexer q/w %.0f, scores %.0f, top-k %.0f, attention %.0f, output %.0f)\n",
                     ms_m[0] + ms_m[1] + ms_m[2] + ms_m[3] + ms_m[4] + ms_m[5] + ms_m[6], ms_m[0], ms_m[4], ms_m[6], ms_m[1], ms_m[5], ms_m[2], ms_m[3]);
    if (prof)
        std::fprintf(stderr, "strata glm5 prefill prof: of the host blob copies - waiting for the GPU's previous half %.0f ms, "
                             "fetching the streamed blobs %.0f ms (threads: in blob() %.0f ms, copying %.0f ms, summed)\n",
                     pf_wait_ms - wait0, pf_fetch_ms - fetch0, pf_blob_thread_ms - blobt0, pf_copy_thread_ms - copyt0);
    return true;
}

}  // namespace strata::core
