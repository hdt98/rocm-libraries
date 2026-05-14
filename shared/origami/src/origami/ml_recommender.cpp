// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/ml_recommender.hpp"

#include "origami/gemm.hpp"
#include "origami/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace origami {
namespace ml_recommender {

namespace {

float scalar_dot(const float* a, const float* b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t j = 0; j < n; ++j) sum += a[j] * b[j];
  return sum;
}

#ifdef __AVX512F__
float avx512_dot(const float* a, const float* b, std::size_t n) {
  __m512 acc    = _mm512_setzero_ps();
  std::size_t j = 0;
  for (; j + 16 <= n; j += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j), acc);
  float sum = _mm512_reduce_add_ps(acc);
  for (; j < n; ++j) sum += a[j] * b[j];
  return sum;
}
#endif

using dot_fn_t = float (*)(const float*, const float*, std::size_t);

dot_fn_t select_dot_fn() {
#ifdef __AVX512F__
#ifdef __x86_64__
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1u << 16))) return avx512_dot;
#endif
#endif
  return scalar_dot;
}

static const dot_fn_t s_dot = select_dot_fn();

struct HwConst {
  int   N_CU;
  std::int64_t LDS;
  std::int64_t L2;
  int   parallel_mi_cu;
  float bw_a, bw_b, bw_c;
};

constexpr HwConst kHwGfx950 = {256, 65536, 4194304, 4, 0.0f, 0.008f, 0.0f};

const HwConst& hw_for_arch(const std::string& ) {

  return kHwGfx950;
}

float bpe_for_dtype(data_type_t dt) {
  switch (dt) {
    case data_type_t::Float: case data_type_t::XFloat32: return 4.0f;
    case data_type_t::Double: return 8.0f;
    case data_type_t::Half: case data_type_t::BFloat16: return 2.0f;
    case data_type_t::Float8_fnuz: case data_type_t::BFloat8_fnuz:
    case data_type_t::Float8: case data_type_t::BFloat8:
    case data_type_t::Int8: return 1.0f;
    case data_type_t::Int32: return 4.0f;
    case data_type_t::Float4: return 0.5f;
    case data_type_t::Float6: case data_type_t::BFloat6: return 0.75f;
    case data_type_t::Int4: return 0.5f;
    case data_type_t::Float8BFloat8: case data_type_t::BFloat8Float8: return 1.0f;
    case data_type_t::Float8BFloat8_fnuz: case data_type_t::BFloat8Float8_fnuz: return 1.0f;
    default: return 2.0f;
  }
}

int dtype_id(data_type_t dt) {
  switch (dt) {
    case data_type_t::Float: return 0;
    case data_type_t::XFloat32: return 1;
    case data_type_t::Half: return 2;
    case data_type_t::BFloat16: return 3;
    case data_type_t::Float8: case data_type_t::Float8_fnuz: return 4;
    case data_type_t::BFloat8: case data_type_t::BFloat8_fnuz: return 5;
    case data_type_t::Float6: return 6;
    case data_type_t::BFloat6: return 7;
    case data_type_t::Float4: return 8;
    case data_type_t::Int8: return 9;
    case data_type_t::Int32: return 10;
    case data_type_t::Float8BFloat8: case data_type_t::Float8BFloat8_fnuz: return 11;
    case data_type_t::BFloat8Float8: case data_type_t::BFloat8Float8_fnuz: return 12;
    case data_type_t::Double: return 13;
    case data_type_t::Int4: return 14;
    default: return 0;
  }
}

struct MiKey { int m, n, k; data_type_t dt; };
struct MiKeyHash {
  std::size_t operator()(const MiKey& k) const noexcept {
    return std::hash<std::int64_t>()(
      (std::int64_t(k.m) << 40) ^ (std::int64_t(k.n) << 24) ^
      (std::int64_t(k.k) << 8) ^ static_cast<int>(k.dt));
  }
};
struct MiKeyEq {
  bool operator()(const MiKey& a, const MiKey& b) const noexcept {
    return a.m == b.m && a.n == b.n && a.k == b.k && a.dt == b.dt;
  }
};

float get_mi_latency(int mi_m, int mi_n, int mi_k, data_type_t mi_dt) {
  static const std::unordered_map<MiKey, int, MiKeyHash, MiKeyEq> table = {
    {{32, 32, 2,  data_type_t::Float},    64},
    {{32, 32, 1,  data_type_t::Float},    64},
    {{16, 16, 4,  data_type_t::Float},    32},
    {{16, 16, 1,  data_type_t::Float},    32},
    {{32, 32, 16, data_type_t::BFloat16}, 32},
    {{16, 16, 32, data_type_t::BFloat16}, 16},
    {{32, 32, 8,  data_type_t::Half},     32},
    {{32, 32, 16, data_type_t::Half},     32},
    {{16, 16, 16, data_type_t::Half},     16},
    {{16, 16, 32, data_type_t::Half},     16},
    {{32, 32, 64, data_type_t::Float8},   32},
    {{16, 16,128, data_type_t::Float8},   16},
    {{32, 32, 64, data_type_t::BFloat8},  32},
    {{16, 16,128, data_type_t::BFloat8},  16},
    {{32, 32, 8,  data_type_t::XFloat32}, 32},
    {{16, 16, 16, data_type_t::XFloat32}, 16},
  };
  auto it = table.find({mi_m, mi_n, mi_k, mi_dt});
  int raw = (it == table.end()) ? 32 : it->second;
  return static_cast<float>(raw) / 4.0f;
}

constexpr std::size_t kQueryDim = 64;
constexpr std::size_t kItemDim  = 20;
constexpr std::size_t kInterDim = 47;
constexpr std::size_t kFeatureCount = kQueryDim + kItemDim + kInterDim;

constexpr std::array<int, 131> kFeaturePermutation = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 16, 18, 20, 22, 24,
  26, 17, 19, 21, 23, 25, 27, 28, 30, 32, 29, 31, 33, 34, 36, 38,
  40, 35, 37, 39, 41, 42, 43, 44, 45, 49, 51, 50, 52, 46, 47, 48,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
  98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
  114, 125, 126, 127, 128, 129, 130, 84, 85, 115, 116, 117, 118, 119, 120, 121,
  122, 123, 124,
};

inline float safe_log2(float x, float floor_v = 1.0f) {
  return std::log2(std::max(x, floor_v));
}

void compute_features_raw(const problem_t& p, const config_t& c,
                          const HwConst& hw, float* out) {
  const float m_f  = static_cast<float>(p.size.m);
  const float n_f  = static_cast<float>(p.size.n);
  const float k_f  = static_cast<float>(p.size.k);
  const float b_f  = static_cast<float>(p.batch);
  const float mt_mf = static_cast<float>(std::max<std::size_t>(c.mt.m, 1));
  const float mt_nf = static_cast<float>(std::max<std::size_t>(c.mt.n, 1));
  const float mt_kf = static_cast<float>(std::max<std::size_t>(c.mt.k, 1));
  const float mi_mf = static_cast<float>(std::max<std::size_t>(c.mi.m, 1));
  const float mi_nf = static_cast<float>(std::max<std::size_t>(c.mi.n, 1));
  const float mi_kf = static_cast<float>(std::max<std::size_t>(c.mi.k, 1));
  const float occ_f = static_cast<float>(std::max<int>(c.occupancy, 1));
  const float grvw_af = static_cast<float>(std::max<std::size_t>(c.grvw_a, 1));
  const float grvw_bf = static_cast<float>(std::max<std::size_t>(c.grvw_b, 1));
  const float gwvw_df = static_cast<float>(std::max<std::size_t>(c.gwvw_d, 1));

  const float bpe_a = bpe_for_dtype(p.a_dtype);
  const float bpe_b = bpe_for_dtype(p.b_dtype);
  const float bpe_c = bpe_for_dtype(p.c_dtype);
  const float bpe_d = bpe_for_dtype(p.d_dtype);
  const int cms = c.hand_optimized_main_loop ? 1 : 0;
  const int nta = c.cache_hints_a;
  const int ntb = c.cache_hints_b;
  const int N_CU = hw.N_CU;

  const float mn = m_f * n_f;
  const float mk = m_f * k_f;
  const float nk = n_f * k_f;
  const float total_flops = 2.0f * m_f * n_f * k_f * b_f;
  const float a_bytes = mk * bpe_a * b_f;
  const float b_bytes = nk * bpe_b * b_f;
  const float c_bytes = mn * bpe_c * b_f;
  const float d_bytes = mn * bpe_d * b_f;
  const float total_bytes = a_bytes + b_bytes + c_bytes + d_bytes;
  const float ai_prob = total_flops / std::max(total_bytes, 1.0f);

  const float nt_m = std::ceil(m_f / mt_mf);
  const float nt_n = std::ceil(n_f / mt_nf);
  const float num_tiles = nt_m * nt_n;
  const float num_tiles_total = num_tiles * b_f;
  const float k_iters = std::ceil(k_f / mt_kf);
  const float tile_area = mt_mf * mt_nf;
  const float tile_volume = mt_mf * mt_nf * mt_kf;
  const float tile_coverage = tile_area / std::max(mn, 1.0f);
  const float waves = std::ceil(num_tiles / N_CU);
  const float wave_eff = waves > 0 ? num_tiles / (waves * N_CU) : 1.0f;
  const float rho = num_tiles_total / N_CU;
  const float batch_tiles_ratio = b_f * num_tiles / N_CU;
  const float launched_m = nt_m * mt_mf;
  const float launched_n = nt_n * mt_nf;
  const float launched_k = k_iters * mt_kf;
  const float util_out = mn / std::max(launched_m * launched_n, 1.0f);
  const float util_3d = (m_f * n_f * k_f) /
    std::max(launched_m * launched_n * launched_k, 1.0f);
  const float lds_bytes = mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b;
  const float lds_ratio = lds_bytes / static_cast<float>(hw.LDS);
  const float l2_fit_ratio = total_bytes / static_cast<float>(hw.L2);
  const float bw_per_cu = total_bytes / N_CU;
  const float l2_working_set =
    (nt_m * mt_mf * mt_kf * bpe_a) + (nt_n * mt_nf * mt_kf * bpe_b);
  const float l2_fit_ws = std::min(l2_working_set / static_cast<float>(hw.L2), 2.0f) / 2.0f;

  const float L_MI = get_mi_latency(static_cast<int>(c.mi.m), static_cast<int>(c.mi.n),
                                    static_cast<int>(c.mi.k), p.mi_dtype);
  const float n_mi = std::ceil(mt_mf / mi_mf) * std::ceil(mt_nf / mi_nf) * std::ceil(mt_kf / mi_kf);
  const float L_MT = n_mi * L_MI;

  const float ai_tile = (2.0f * mt_mf * mt_nf * mt_kf) /
    (mt_mf * mt_kf + mt_nf * mt_kf + mt_mf * mt_nf);

  const float active_cus = std::min(num_tiles_total, static_cast<float>(N_CU));
  const float bw_occ = std::min(
    1.0f, hw.bw_a * active_cus * active_cus + hw.bw_b * active_cus + hw.bw_c);

  const float pgr_alpha = 0.5f;
  const float pgr_eff = 1.0f + pgr_alpha * std::min(1.0f, k_iters - 1.0f);
  const float t_math_proxy = L_MT;
  const float nlca_proxy = std::max(1.0f, mt_mf / std::max(grvw_af * bpe_a, 1.0f));
  const float nlcb_proxy = std::max(1.0f, mt_nf / std::max(grvw_bf * bpe_b, 1.0f));
  const float wave_pack_proxy = (mt_mf / mi_mf) * (mt_nf / mi_nf) * occ_f / N_CU;

  auto is_pow2 = [](float x) {
    int xi = static_cast<int>(x);
    return (xi > 0 && (xi & (xi - 1)) == 0) ? 1.0f : 0.0f;
  };
  std::size_t i = 0;

  out[i++] = safe_log2(m_f);
  out[i++] = safe_log2(n_f);
  out[i++] = safe_log2(k_f);
  out[i++] = safe_log2(b_f);
  out[i++] = safe_log2(mn);
  out[i++] = safe_log2(mk);
  out[i++] = safe_log2(nk);
  out[i++] = std::min(ai_prob, 1.0e6f);
  out[i++] = std::log2(std::max(ai_prob, 0.001f));
  out[i++] = std::min(std::max(m_f / std::max(n_f, 1.0f), 0.001f), 10000.0f);
  out[i++] = std::min(std::max(m_f / std::max(k_f, 1.0f), 0.001f), 10000.0f);
  out[i++] = std::min(std::max(n_f / std::max(k_f, 1.0f), 0.001f), 10000.0f);
  out[i++] = std::log2(std::max(m_f / std::max(n_f, 1.0f), 0.001f));
  out[i++] = is_pow2(m_f);
  out[i++] = is_pow2(n_f);
  out[i++] = is_pow2(k_f);
  int mi_int = static_cast<int>(m_f), ni_int = static_cast<int>(n_f);
  for (int base : {256, 128, 64, 32, 16, 8}) {
    out[i++] = static_cast<float>(mi_int % base);
    out[i++] = static_cast<float>(ni_int % base);
  }
  for (int base : {64, 128, 256}) {
    int mr = mi_int % base, nr = ni_int % base;
    out[i++] = static_cast<float>(std::min(mr, base - mr)) / base;
    out[i++] = static_cast<float>(std::min(nr, base - nr)) / base;
  }
  for (int st : {32, 64, 128, 256}) {
    float s_nt_m = std::ceil(m_f / st);
    float s_nt_n = std::ceil(n_f / st);
    out[i++] = std::log2(std::max(s_nt_m * s_nt_n, 1.0f));
    out[i++] = mn / std::max(s_nt_m * st * s_nt_n * st, 1.0f);
  }
  for (int kd : {32, 64, 128, 256}) {
    out[i++] = std::log2(std::max(std::ceil(k_f / kd), 1.0f));
  }
  out[i++] = std::log2(std::max(mk * bpe_a / (1024.0f * 1024.0f), 0.001f));
  out[i++] = std::log2(std::max(nk * bpe_b / (1024.0f * 1024.0f), 0.001f));
  out[i++] = std::log2(std::max(mn * bpe_c / (1024.0f * 1024.0f), 0.001f));
  for (int st : {128, 256}) {
    float s_nt = std::ceil(m_f / st) * std::ceil(n_f / st);
    float s_w = std::ceil(s_nt / N_CU);
    out[i++] = std::log2(std::max(s_w, 1.0f));
    out[i++] = s_nt / std::max(s_w * N_CU, 1.0f);
  }

  out[i++] = static_cast<float>(p.a_transpose == transpose_t::T ? 1 : 0);
  out[i++] = static_cast<float>(p.b_transpose == transpose_t::T ? 1 : 0);
  out[i++] = static_cast<float>(dtype_id(p.a_dtype));
  out[i++] = static_cast<float>(dtype_id(p.b_dtype));
  out[i++] = static_cast<float>(dtype_id(p.c_dtype));
  out[i++] = static_cast<float>(dtype_id(p.d_dtype));
  out[i++] = static_cast<float>(dtype_id(p.mi_dtype));
  out[i++] = bpe_a;
  out[i++] = bpe_b;
  out[i++] = std::log2(static_cast<float>(std::max<std::size_t>(p.a_mx_block_size, 1)));
  out[i++] = std::log2(static_cast<float>(std::max<std::size_t>(p.b_mx_block_size, 1)));

  out[i++] = safe_log2(mt_mf);
  out[i++] = safe_log2(mt_nf);
  out[i++] = safe_log2(mt_kf);
  out[i++] = safe_log2(mi_mf);
  out[i++] = safe_log2(mi_nf);
  out[i++] = safe_log2(mi_kf);
  out[i++] = safe_log2(tile_area);
  out[i++] = safe_log2(tile_volume);
  out[i++] = mt_mf / mt_nf;
  out[i++] = mt_mf / mt_kf;
  out[i++] = mt_nf / mt_kf;
  out[i++] = static_cast<float>(cms);
  out[i++] = nta / 7.0f;
  out[i++] = ntb / 7.0f;
  out[i++] = occ_f;
  out[i++] = occ_f / 9.0f;
  out[i++] = static_cast<float>(c.workgroup_mapping) / 16.0f;
  out[i++] = grvw_af / 8.0f;
  out[i++] = grvw_bf / 8.0f;
  out[i++] = gwvw_df / 8.0f;

  out[i++] = std::log2(std::max(grvw_af * bpe_a, 1.0f));
  out[i++] = std::log2(std::max(grvw_bf * bpe_b, 1.0f));
  out[i++] = std::log2(std::max(num_tiles, 1.0f));
  out[i++] = std::log2(std::max(num_tiles_total, 1.0f));
  out[i++] = std::log2(std::max(k_iters, 1.0f));
  out[i++] = std::log2(std::max(waves, 1.0f));
  out[i++] = wave_eff;
  out[i++] = std::log2(std::max(rho, 0.001f));
  out[i++] = std::log2(std::max(batch_tiles_ratio, 0.001f));
  out[i++] = util_out;
  out[i++] = util_3d;
  out[i++] = std::min(1.0f / std::max(util_out, 0.01f), 10.0f) / 10.0f;
  out[i++] = std::min(tile_coverage, 1.0f);
  out[i++] = std::log2(std::max(lds_bytes, 1.0f));
  out[i++] = lds_ratio;
  out[i++] = std::min(l2_fit_ratio, 4.0f) / 4.0f;
  out[i++] = l2_fit_ws;
  out[i++] = std::log2(std::max(bw_per_cu, 1.0f));
  out[i++] = std::log2(std::max(total_bytes, 1.0f));
  out[i++] = std::log2(std::max(total_flops, 1.0f));
  out[i++] = std::min(1.0f, mt_mf / std::max(m_f, 1.0f));
  out[i++] = std::min(1.0f, mt_nf / std::max(n_f, 1.0f));
  out[i++] = (k_f - (k_iters - 1.0f) * mt_kf) / std::max(k_f, 1.0f);
  out[i++] = (std::fmod(k_f * bpe_a, 128.0f) == 0.0f &&
              std::fmod(mt_kf * bpe_a, 128.0f) == 0.0f) ? 1.0f : 0.0f;
  out[i++] = (m_f <= 2.0f * mt_mf) ? 1.0f : 0.0f;
  out[i++] = (n_f <= 2.0f * mt_nf) ? 1.0f : 0.0f;
  out[i++] = (b_f > 1.0f) ? 1.0f : 0.0f;
  out[i++] = std::log2(std::max(mt_mf * mt_kf * bpe_a, 1.0f));
  out[i++] = std::log2(std::max(mt_nf * mt_kf * bpe_b, 1.0f));
  float compute_density = (mt_mf * mt_nf * mt_kf * 2.0f) /
    std::max(mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b, 1.0f);
  out[i++] = std::log2(std::max(compute_density, 0.001f));
  out[i++] = std::log2(std::max(k_iters * num_tiles * b_f, 1.0f));
  out[i++] = ai_tile;
  out[i++] = L_MI;
  out[i++] = std::log2(std::max(L_MT, 1.0f));
  out[i++] = bw_occ;
  out[i++] = active_cus / N_CU;
  out[i++] = pgr_eff;
  out[i++] = std::log2(std::max(t_math_proxy, 1.0f));
  out[i++] = std::log2(std::max(nlca_proxy, 1.0f));
  out[i++] = std::log2(std::max(nlcb_proxy, 1.0f));
  out[i++] = wave_pack_proxy;
  out[i++] = std::log2(std::max(b_f * k_iters, 1.0f));
  out[i++] = std::log2(std::max(b_f * num_tiles, 1.0f));
  out[i++] = b_f * k_iters / std::max(num_tiles, 1.0f);
  out[i++] = cms * std::log2(std::max(mn, 1.0f));
  out[i++] = cms * std::log2(std::max(num_tiles, 1.0f));
  out[i++] = (num_tiles_total > 0)
    ? num_tiles_total / (std::ceil(num_tiles_total / N_CU) * N_CU)
    : 1.0f;
}

void compute_features_raw_inter_only(const problem_t& p, const config_t& c,
                                     const HwConst& hw, float* out_inter) {
  const float m_f  = static_cast<float>(p.size.m);
  const float n_f  = static_cast<float>(p.size.n);
  const float k_f  = static_cast<float>(p.size.k);
  const float b_f  = static_cast<float>(p.batch);
  const float mt_mf = static_cast<float>(std::max<std::size_t>(c.mt.m, 1));
  const float mt_nf = static_cast<float>(std::max<std::size_t>(c.mt.n, 1));
  const float mt_kf = static_cast<float>(std::max<std::size_t>(c.mt.k, 1));
  const float mi_mf = static_cast<float>(std::max<std::size_t>(c.mi.m, 1));
  const float mi_nf = static_cast<float>(std::max<std::size_t>(c.mi.n, 1));
  const float mi_kf = static_cast<float>(std::max<std::size_t>(c.mi.k, 1));
  const float occ_f = static_cast<float>(std::max<int>(c.occupancy, 1));
  const float grvw_af = static_cast<float>(std::max<std::size_t>(c.grvw_a, 1));
  const float grvw_bf = static_cast<float>(std::max<std::size_t>(c.grvw_b, 1));

  const float bpe_a = bpe_for_dtype(p.a_dtype);
  const float bpe_b = bpe_for_dtype(p.b_dtype);
  const float bpe_c = bpe_for_dtype(p.c_dtype);
  const float bpe_d = bpe_for_dtype(p.d_dtype);
  const int cms = c.hand_optimized_main_loop ? 1 : 0;
  const int N_CU = hw.N_CU;

  const float mn = m_f * n_f;
  const float mk = m_f * k_f;
  const float nk = n_f * k_f;
  const float total_flops = 2.0f * m_f * n_f * k_f * b_f;
  const float a_bytes = mk * bpe_a * b_f;
  const float b_bytes = nk * bpe_b * b_f;
  const float c_bytes = mn * bpe_c * b_f;
  const float d_bytes = mn * bpe_d * b_f;
  const float total_bytes = a_bytes + b_bytes + c_bytes + d_bytes;

  const float nt_m = std::ceil(m_f / mt_mf);
  const float nt_n = std::ceil(n_f / mt_nf);
  const float num_tiles = nt_m * nt_n;
  const float num_tiles_total = num_tiles * b_f;
  const float k_iters = std::ceil(k_f / mt_kf);
  const float tile_area = mt_mf * mt_nf;
  const float tile_coverage = tile_area / std::max(mn, 1.0f);
  const float waves = std::ceil(num_tiles / N_CU);
  const float wave_eff = waves > 0 ? num_tiles / (waves * N_CU) : 1.0f;
  const float rho = num_tiles_total / N_CU;
  const float batch_tiles_ratio = b_f * num_tiles / N_CU;
  const float launched_m = nt_m * mt_mf;
  const float launched_n = nt_n * mt_nf;
  const float launched_k = k_iters * mt_kf;
  const float util_out = mn / std::max(launched_m * launched_n, 1.0f);
  const float util_3d = (m_f * n_f * k_f) /
    std::max(launched_m * launched_n * launched_k, 1.0f);
  const float lds_bytes = mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b;
  const float lds_ratio = lds_bytes / static_cast<float>(hw.LDS);
  const float l2_fit_ratio = total_bytes / static_cast<float>(hw.L2);
  const float bw_per_cu = total_bytes / N_CU;
  const float l2_working_set =
    (nt_m * mt_mf * mt_kf * bpe_a) + (nt_n * mt_nf * mt_kf * bpe_b);
  const float l2_fit_ws = std::min(l2_working_set / static_cast<float>(hw.L2), 2.0f) / 2.0f;

  const float L_MI = get_mi_latency(static_cast<int>(c.mi.m), static_cast<int>(c.mi.n),
                                    static_cast<int>(c.mi.k), p.mi_dtype);
  const float n_mi = std::ceil(mt_mf / mi_mf) * std::ceil(mt_nf / mi_nf) * std::ceil(mt_kf / mi_kf);
  const float L_MT = n_mi * L_MI;

  const float ai_tile = (2.0f * mt_mf * mt_nf * mt_kf) /
    (mt_mf * mt_kf + mt_nf * mt_kf + mt_mf * mt_nf);

  const float pgr_eff = 1.0f + 0.5f * std::min(1.0f, k_iters - 1.0f);
  const float t_math_proxy = L_MT;
  const float nlca_proxy = std::max(1.0f, mt_mf / std::max(grvw_af * bpe_a, 1.0f));
  const float nlcb_proxy = std::max(1.0f, mt_nf / std::max(grvw_bf * bpe_b, 1.0f));
  const float wave_pack_proxy = (mt_mf / mi_mf) * (mt_nf / mi_nf) * occ_f / N_CU;

  const float active_cus = std::min(num_tiles_total, static_cast<float>(N_CU));
  const float bw_occ = std::min(
    1.0f, hw.bw_a * active_cus * active_cus + hw.bw_b * active_cus + hw.bw_c);

  std::size_t i = 0;
  out_inter[i++] = std::log2(std::max(grvw_af * bpe_a, 1.0f));
  out_inter[i++] = std::log2(std::max(grvw_bf * bpe_b, 1.0f));
  out_inter[i++] = std::log2(std::max(num_tiles, 1.0f));
  out_inter[i++] = std::log2(std::max(num_tiles_total, 1.0f));
  out_inter[i++] = std::log2(std::max(k_iters, 1.0f));
  out_inter[i++] = std::log2(std::max(waves, 1.0f));
  out_inter[i++] = wave_eff;
  out_inter[i++] = std::log2(std::max(rho, 0.001f));
  out_inter[i++] = std::log2(std::max(batch_tiles_ratio, 0.001f));
  out_inter[i++] = util_out;
  out_inter[i++] = util_3d;
  out_inter[i++] = std::min(1.0f / std::max(util_out, 0.01f), 10.0f) / 10.0f;
  out_inter[i++] = std::min(tile_coverage, 1.0f);
  out_inter[i++] = std::log2(std::max(lds_bytes, 1.0f));
  out_inter[i++] = lds_ratio;
  out_inter[i++] = std::min(l2_fit_ratio, 4.0f) / 4.0f;
  out_inter[i++] = l2_fit_ws;
  out_inter[i++] = std::log2(std::max(bw_per_cu, 1.0f));
  out_inter[i++] = std::log2(std::max(total_bytes, 1.0f));
  out_inter[i++] = std::log2(std::max(total_flops, 1.0f));
  out_inter[i++] = std::min(1.0f, mt_mf / std::max(m_f, 1.0f));
  out_inter[i++] = std::min(1.0f, mt_nf / std::max(n_f, 1.0f));
  out_inter[i++] = (k_f - (k_iters - 1.0f) * mt_kf) / std::max(k_f, 1.0f);
  out_inter[i++] = (std::fmod(k_f * bpe_a, 128.0f) == 0.0f &&
                    std::fmod(mt_kf * bpe_a, 128.0f) == 0.0f) ? 1.0f : 0.0f;
  out_inter[i++] = (m_f <= 2.0f * mt_mf) ? 1.0f : 0.0f;
  out_inter[i++] = (n_f <= 2.0f * mt_nf) ? 1.0f : 0.0f;
  out_inter[i++] = (b_f > 1.0f) ? 1.0f : 0.0f;
  out_inter[i++] = std::log2(std::max(mt_mf * mt_kf * bpe_a, 1.0f));
  out_inter[i++] = std::log2(std::max(mt_nf * mt_kf * bpe_b, 1.0f));
  float compute_density = (mt_mf * mt_nf * mt_kf * 2.0f) /
    std::max(mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b, 1.0f);
  out_inter[i++] = std::log2(std::max(compute_density, 0.001f));
  out_inter[i++] = std::log2(std::max(k_iters * num_tiles * b_f, 1.0f));
  out_inter[i++] = ai_tile;
  out_inter[i++] = L_MI;
  out_inter[i++] = std::log2(std::max(L_MT, 1.0f));
  out_inter[i++] = bw_occ;
  out_inter[i++] = active_cus / N_CU;
  out_inter[i++] = pgr_eff;
  out_inter[i++] = std::log2(std::max(t_math_proxy, 1.0f));
  out_inter[i++] = std::log2(std::max(nlca_proxy, 1.0f));
  out_inter[i++] = std::log2(std::max(nlcb_proxy, 1.0f));
  out_inter[i++] = wave_pack_proxy;
  out_inter[i++] = std::log2(std::max(b_f * k_iters, 1.0f));
  out_inter[i++] = std::log2(std::max(b_f * num_tiles, 1.0f));
  out_inter[i++] = b_f * k_iters / std::max(num_tiles, 1.0f);
  out_inter[i++] = cms * std::log2(std::max(mn, 1.0f));
  out_inter[i++] = cms * std::log2(std::max(num_tiles, 1.0f));
  out_inter[i++] = (num_tiles_total > 0)
    ? num_tiles_total / (std::ceil(num_tiles_total / N_CU) * N_CU)
    : 1.0f;

}

struct __attribute__((packed)) tile_signature_t {
  std::uint16_t mt_m, mt_n, mt_k;
  std::uint8_t  mi_m, mi_n, mi_k;
  std::uint8_t  cms;
  std::uint8_t  nta, ntb;
  std::uint8_t  grvw_a, grvw_b, gwvw_d;
  std::uint8_t  reserved;
};
static_assert(sizeof(tile_signature_t) == 16, "tile_signature_t must be 16 bytes");

struct sig_hash_t {
  std::size_t operator()(const tile_signature_t& s) const noexcept {
    std::uint64_t lo, hi;
    std::memcpy(&lo, reinterpret_cast<const char*>(&s), 8);
    std::memcpy(&hi, reinterpret_cast<const char*>(&s) + 8, 8);
    return std::hash<std::uint64_t>()(lo) ^ (std::hash<std::uint64_t>()(hi) << 1);
  }
};
struct sig_eq_t {
  bool operator()(const tile_signature_t& a, const tile_signature_t& b) const noexcept {
    return std::memcmp(&a, &b, sizeof(tile_signature_t)) == 0;
  }
};

inline tile_signature_t signature_for_config(const config_t& c) {
  tile_signature_t s{};
  s.mt_m   = static_cast<std::uint16_t>(c.mt.m);
  s.mt_n   = static_cast<std::uint16_t>(c.mt.n);
  s.mt_k   = static_cast<std::uint16_t>(c.mt.k);
  s.mi_m   = static_cast<std::uint8_t>(c.mi.m);
  s.mi_n   = static_cast<std::uint8_t>(c.mi.n);
  s.mi_k   = static_cast<std::uint8_t>(c.mi.k);
  s.cms    = c.hand_optimized_main_loop ? 1u : 0u;
  s.nta    = static_cast<std::uint8_t>(c.cache_hints_a);
  s.ntb    = static_cast<std::uint8_t>(c.cache_hints_b);
  s.grvw_a = static_cast<std::uint8_t>(c.grvw_a);
  s.grvw_b = static_cast<std::uint8_t>(c.grvw_b);
  s.gwvw_d = static_cast<std::uint8_t>(c.gwvw_d);
  s.reserved = 0;
  return s;
}

struct ClusterModel {
  std::uint32_t cluster_id;
  std::uint32_t embed_dim;
  std::uint32_t hidden_dim;
  std::uint32_t inter_hidden;
  std::uint32_t n_tiles;
  float temperature;

  std::vector<tile_signature_t> tile_order_signatures;
  std::vector<float> query_mean, query_std;
  std::vector<float> item_mean, item_std;
  std::vector<float> inter_mean, inter_std;
  std::vector<float> q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  std::vector<float> i_w0, i_b0, i_w2, i_b2;
  std::vector<float> x_w0, x_b0, x_w2, x_b2;

  std::unordered_map<tile_signature_t, std::vector<float>, sig_hash_t, sig_eq_t> item_emb_cache;
};

struct MLWeights {
  std::uint32_t query_dim = 0, item_dim = 0, inter_dim = 0, n_clusters = 0;
  double w_b = 0.0;
  std::vector<float> centroids;
  std::vector<ClusterModel> clusters;
};

static MLWeights g_w;
static bool g_loaded = false;
static std::once_flag g_load_once;

template <typename T>
static bool read_pod(std::istream& f, T* out) {
  f.read(reinterpret_cast<char*>(out), sizeof(T));
  return f.gcount() == static_cast<std::streamsize>(sizeof(T));
}
static bool read_floats(std::istream& f, std::size_t n, std::vector<float>* out) {
  out->resize(n);
  f.read(reinterpret_cast<char*>(out->data()), sizeof(float) * n);
  return f.gcount() == static_cast<std::streamsize>(sizeof(float) * n);
}
static bool read_bf16_as_floats(std::istream& f, std::size_t n, std::vector<float>* out) {
  std::vector<std::uint16_t> tmp(n);
  f.read(reinterpret_cast<char*>(tmp.data()), sizeof(std::uint16_t) * n);
  if (f.gcount() != static_cast<std::streamsize>(sizeof(std::uint16_t) * n)) return false;
  out->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t bits = static_cast<std::uint32_t>(tmp[i]) << 16;
    float v;
    std::memcpy(&v, &bits, sizeof(float));
    (*out)[i] = v;
  }
  return true;
}

static bool read_int8_as_floats(std::istream& f,
                                 std::size_t out_chan,
                                 std::size_t in_per_chan,
                                 std::vector<float>* out) {

  std::vector<float> scales(out_chan);
  f.read(reinterpret_cast<char*>(scales.data()), sizeof(float) * out_chan);
  if (f.gcount() != static_cast<std::streamsize>(sizeof(float) * out_chan)) return false;

  std::size_t n_total = out_chan * in_per_chan;
  std::vector<std::int8_t> q(n_total);
  f.read(reinterpret_cast<char*>(q.data()), n_total);
  if (f.gcount() != static_cast<std::streamsize>(n_total)) return false;

  out->resize(n_total);
  for (std::size_t i = 0; i < out_chan; ++i) {
    float s = scales[i];
    for (std::size_t j = 0; j < in_per_chan; ++j) {
      (*out)[i * in_per_chan + j] = s * static_cast<float>(q[i * in_per_chan + j]);
    }
  }
  return true;
}

constexpr std::uint8_t kDtypeFp32 = 0;
constexpr std::uint8_t kDtypeBf16 = 1;
constexpr std::uint8_t kDtypeInt8 = 2;

static bool load_binary_stream(std::istream& f, MLWeights* out) {
  if (!f) return false;
  char magic[8]; f.read(magic, 8);
  if (std::memcmp(magic, "STRICTML", 8) != 0) return false;
  std::uint32_t version;
  if (!read_pod(f, &version)) return false;
  if (version != 3) return false;
  if (!read_pod(f, &out->query_dim)) return false;
  if (!read_pod(f, &out->item_dim)) return false;
  if (!read_pod(f, &out->inter_dim)) return false;
  if (!read_pod(f, &out->n_clusters)) return false;
  if (!read_pod(f, &out->w_b)) return false;

  std::uint8_t dtype_tag = kDtypeFp32;
  if (!read_pod(f, &dtype_tag)) return false;
  char reserved[7];
  f.read(reserved, 7);
  if (f.gcount() != 7) return false;
  if (dtype_tag != kDtypeFp32 && dtype_tag != kDtypeBf16 && dtype_tag != kDtypeInt8) return false;

  auto read_weight = [&](std::size_t out_chan, std::size_t in_per_chan,
                          std::vector<float>* dst) -> bool {
    if (dtype_tag == kDtypeInt8) {
      return read_int8_as_floats(f, out_chan, in_per_chan, dst);
    }
    std::size_t n = out_chan * in_per_chan;
    if (dtype_tag == kDtypeBf16) return read_bf16_as_floats(f, n, dst);
    return read_floats(f, n, dst);
  };
  if (!read_floats(f, out->n_clusters * 4, &out->centroids)) return false;

  out->clusters.resize(out->n_clusters);
  for (std::uint32_t i = 0; i < out->n_clusters; ++i) {
    ClusterModel& cm = out->clusters[i];
    if (!read_pod(f, &cm.cluster_id)) return false;
    if (!read_pod(f, &cm.embed_dim)) return false;
    if (!read_pod(f, &cm.hidden_dim)) return false;
    if (!read_pod(f, &cm.inter_hidden)) return false;
    if (!read_pod(f, &cm.n_tiles)) return false;
    if (!read_pod(f, &cm.temperature)) return false;
    cm.tile_order_signatures.resize(cm.n_tiles);
    f.read(reinterpret_cast<char*>(cm.tile_order_signatures.data()),
           sizeof(tile_signature_t) * cm.n_tiles);
    if (f.gcount() != static_cast<std::streamsize>(sizeof(tile_signature_t) * cm.n_tiles))
      return false;

    const std::uint32_t qd = out->query_dim, id = out->item_dim, xd = out->inter_dim;
    const std::uint32_t hd = cm.hidden_dim, ed = cm.embed_dim, ih = cm.inter_hidden;

    if (!read_floats(f, qd, &cm.query_mean)) return false;
    if (!read_floats(f, qd, &cm.query_std)) return false;
    if (!read_floats(f, id, &cm.item_mean)) return false;
    if (!read_floats(f, id, &cm.item_std)) return false;
    if (!read_floats(f, xd, &cm.inter_mean)) return false;
    if (!read_floats(f, xd, &cm.inter_std)) return false;

    if (!read_weight(hd, qd, &cm.q_w0)) return false;
    if (!read_floats(f, hd, &cm.q_b0)) return false;
    if (!read_weight(hd, hd, &cm.q_w2)) return false;
    if (!read_floats(f, hd, &cm.q_b2)) return false;
    if (!read_weight(ed, hd, &cm.q_w4)) return false;
    if (!read_floats(f, ed, &cm.q_b4)) return false;
    if (!read_weight(hd, id, &cm.i_w0)) return false;
    if (!read_floats(f, hd, &cm.i_b0)) return false;
    if (!read_weight(ed, hd, &cm.i_w2)) return false;
    if (!read_floats(f, ed, &cm.i_b2)) return false;
    if (!read_weight(ih, xd, &cm.x_w0)) return false;
    if (!read_floats(f, ih, &cm.x_b0)) return false;
    if (!read_weight(1, ih, &cm.x_w2)) return false;
    if (!read_floats(f, 1, &cm.x_b2)) return false;
  }
  return true;
}

static void dump_load_diagnostics(const char* tag, const MLWeights& w) {
  if (std::getenv("ML_DIAG") == nullptr) return;
  std::fprintf(stderr,
    "[ML_DIAG %s] q=%u i=%u x=%u nclusters=%u w_b=%g clusters.size=%zu\n",
    tag, w.query_dim, w.item_dim, w.inter_dim, w.n_clusters, w.w_b,
    w.clusters.size());
  std::size_t total_tiles = 0;
  for (const auto& cm : w.clusters) total_tiles += cm.tile_order_signatures.size();
  std::fprintf(stderr, "[ML_DIAG %s] total_tiles_across_clusters=%zu\n",
               tag, total_tiles);
  std::fflush(stderr);
}

static bool load_binary(const char* path, MLWeights* out) {
  std::ifstream f(path, std::ios::binary);
  bool ok = load_binary_stream(f, out);
  if (ok) dump_load_diagnostics("FILE", *out);
  return ok;
}

static bool load_binary_from_buffer(const unsigned char* buf, std::size_t len,
                                    MLWeights* out) {
  std::string s(reinterpret_cast<const char*>(buf), len);
  std::istringstream iss(std::move(s), std::ios::binary);
  bool ok = load_binary_stream(iss, out);
  if (ok) dump_load_diagnostics("EMBED", *out);
  return ok;
}

#ifdef __AVX512F__
inline void linear_relu(const float* __restrict__ W, const float* __restrict__ b,
                        const float* __restrict__ x,
                        std::size_t m, std::size_t k, float* __restrict__ out) {
  for (std::size_t i = 0; i < m; ++i) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    const float* Wi = W + i * k;
    for (; j + 16 <= k; j += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(Wi + j), _mm512_loadu_ps(x + j), acc);
    float v = b[i] + _mm512_reduce_add_ps(acc);
    for (; j < k; ++j) v += Wi[j] * x[j];
    out[i] = v > 0.0f ? v : 0.0f;
  }
}
inline void linear(const float* __restrict__ W, const float* __restrict__ b,
                   const float* __restrict__ x,
                   std::size_t m, std::size_t k, float* __restrict__ out) {
  for (std::size_t i = 0; i < m; ++i) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t j = 0;
    const float* Wi = W + i * k;
    for (; j + 16 <= k; j += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(Wi + j), _mm512_loadu_ps(x + j), acc);
    float v = b[i] + _mm512_reduce_add_ps(acc);
    for (; j < k; ++j) v += Wi[j] * x[j];
    out[i] = v;
  }
}
#else
inline void linear_relu(const float* W, const float* b, const float* x,
                        std::size_t m, std::size_t k, float* out) {
  for (std::size_t i = 0; i < m; ++i) {
    float acc = b[i] + s_dot(W + i * k, x, k);
    out[i] = acc > 0.0f ? acc : 0.0f;
  }
}
inline void linear(const float* W, const float* b, const float* x,
                   std::size_t m, std::size_t k, float* out) {
  for (std::size_t i = 0; i < m; ++i) {
    out[i] = b[i] + s_dot(W + i * k, x, k);
  }
}
#endif

static void compute_item_embedding(const ClusterModel& cm,
                                   const float* item_features,
                                   float* out_embedding,
                                   std::size_t item_dim) {
  thread_local std::vector<float> normed(64);
  normed.resize(item_dim);
  for (std::size_t i = 0; i < item_dim; ++i) {
    float s = std::max(cm.item_std[i], 1e-6f);
    normed[i] = (item_features[i] - cm.item_mean[i]) / s;
  }
  thread_local std::vector<float> h(256);
  h.resize(cm.hidden_dim);
  linear_relu(cm.i_w0.data(), cm.i_b0.data(), normed.data(),
              cm.hidden_dim, item_dim, h.data());
  linear(cm.i_w2.data(), cm.i_b2.data(), h.data(),
         cm.embed_dim, cm.hidden_dim, out_embedding);
}

static void build_item_emb_cache(MLWeights& w,
                                 const std::vector<config_t>& configs) {
  std::unordered_map<tile_signature_t, const config_t*, sig_hash_t, sig_eq_t> by_sig;
  by_sig.reserve(configs.size());
  for (const auto& c : configs) by_sig[signature_for_config(c)] = &c;
  const HwConst& hw = hw_for_arch("gfx950");
  for (auto& cm : w.clusters) {
    cm.item_emb_cache.reserve(cm.n_tiles);
    for (const auto& sig : cm.tile_order_signatures) {
      auto it = by_sig.find(sig);
      if (it == by_sig.end()) continue;

      problem_t synth = {};
      synth.size = dim3_t{1024, 1024, 1024};
      synth.batch = 1;
      synth.a_dtype = synth.b_dtype = synth.c_dtype = synth.d_dtype = data_type_t::BFloat16;
      synth.mi_dtype = data_type_t::BFloat16;
      std::array<float, kFeatureCount> raw;
      compute_features_raw(synth, *it->second, hw, raw.data());

      std::vector<float> item_in(w.item_dim);
      for (std::size_t k = 0; k < w.item_dim; ++k) {
        item_in[k] = raw[kFeaturePermutation[w.query_dim + k]];
      }
      std::vector<float> emb(cm.embed_dim);
      compute_item_embedding(cm, item_in.data(), emb.data(), w.item_dim);
      cm.item_emb_cache[sig] = std::move(emb);
    }
  }
}

}

bool load_weights(const std::string& bin_path) {
  MLWeights w;
  if (!load_binary(bin_path.c_str(), &w)) return false;
  g_w = std::move(w);
  g_loaded = true;
  return true;
}

bool weights_loaded() { return g_loaded; }

static void ensure_weights() {
  if (g_loaded) return;
  std::call_once(g_load_once, []() {
    if (g_loaded) return;
    const char* env = std::getenv("ML_RECOMMENDER_WEIGHTS");
    if (env && load_weights(env)) return;
    if (load_weights("./ml_recommender_weights.bin")) return;
    if (load_weights("/opt/rocm/share/origami/ml_recommender_weights.bin")) return;
    std::fprintf(stderr, "[ML] no weights loaded; set "
                 "ML_RECOMMENDER_WEIGHTS or place "
                 "ml_recommender_weights.bin in cwd or /opt/rocm/share/origami/\n");
  });
}

static int route_cluster(const MLWeights& w, const problem_t& p) {
  const float coord[4] = {
    static_cast<float>(std::log2(std::max<double>(p.size.m, 1.0))),
    static_cast<float>(std::log2(std::max<double>(p.size.n, 1.0))),
    static_cast<float>(std::log2(std::max<double>(p.size.k, 1.0))),
    static_cast<float>(w.w_b * std::log2(std::max<double>(p.batch, 1.0))),
  };
  int best = 0;
  float best_d = std::numeric_limits<float>::max();
  for (std::uint32_t i = 0; i < w.n_clusters; ++i) {
    const float* c = &w.centroids[i * 4];
    float d = 0.0f;
    for (int j = 0; j < 4; ++j) { float dj = coord[j] - c[j]; d += dj * dj; }
    if (d < best_d) { best_d = d; best = static_cast<int>(i); }
  }
  return best;
}

bool cluster_uses_ml(int ) { return true; }

int route_cluster_for_problem(const problem_t& problem) {
  ensure_weights();
  if (!g_loaded) return -1;

  const char* env = std::getenv("ML_FORCE_CLUSTER");
  int force_cluster = env ? std::atoi(env) : -1;
  if (force_cluster >= 0) return force_cluster;
  return route_cluster(g_w, problem);
}

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& ,
                                              const std::vector<config_t>& configs) {
  ensure_weights();
  if (!g_loaded || configs.empty()) {

    std::vector<prediction_result_t> out;
    out.reserve(configs.size());
    for (const auto& c : configs) {
      prediction_result_t r{std::numeric_limits<double>::quiet_NaN(), c};
      out.push_back(r);
    }
    return out;
  }

  static const bool s_ml_timing = []() {
    const char* env = std::getenv("ML_RECOMMENDER_TIMING");
    return env && (env[0] == '1' || env[0] == 't' || env[0] == 'T');
  }();
  std::chrono::steady_clock::time_point t_total_start, t_route_start, t_route_end,
      t_cache_end, t_score_end, t_sort_end;
  if (s_ml_timing) t_total_start = std::chrono::steady_clock::now();

  static const int s_top_k_only = []() {
    const char* env = std::getenv("ML_RECOMMENDER_TOP_K_ONLY");
    return env ? std::atoi(env) : 16;
  }();

  const HwConst& hw = hw_for_arch("gfx950");

  static std::once_flag s_cache_once;
  std::call_once(s_cache_once, [&]() { build_item_emb_cache(g_w, configs); });
  if (s_ml_timing) t_cache_end = std::chrono::steady_clock::now();

  if (s_ml_timing) t_route_start = std::chrono::steady_clock::now();
  const char* env_force = std::getenv("ML_FORCE_CLUSTER");
  int force_cluster = env_force ? std::atoi(env_force) : -1;
  int cid = (force_cluster >= 0 && force_cluster < (int)g_w.n_clusters)
            ? force_cluster
            : route_cluster(g_w, problem);
  if (s_ml_timing) t_route_end = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t_setup_end;
  const ClusterModel& cm = g_w.clusters[cid];

  thread_local const config_t* s_last_configs_data = nullptr;
  thread_local std::size_t s_last_configs_size = 0;
  thread_local std::unordered_map<tile_signature_t, std::uint32_t, sig_hash_t, sig_eq_t> s_sig_to_local;

  bool _ml_did_rebuild = false;
  if (configs.data() != s_last_configs_data || configs.size() != s_last_configs_size) {
    _ml_did_rebuild = true;
    s_sig_to_local.clear();
    s_sig_to_local.reserve(configs.size());
    for (std::size_t j = 0; j < configs.size(); ++j) {
      s_sig_to_local.emplace(signature_for_config(configs[j]),
                             static_cast<std::uint32_t>(j));
    }
    s_last_configs_data = configs.data();
    s_last_configs_size = configs.size();
  }

  thread_local std::vector<std::uint32_t> cand_idx;
  cand_idx.clear();
  cand_idx.reserve(cm.tile_order_signatures.size());
  for (const auto& sig : cm.tile_order_signatures) {
    auto it = s_sig_to_local.find(sig);
    if (it != s_sig_to_local.end()) cand_idx.push_back(it->second);
  }
  if (cand_idx.empty()) {
    const float coord[4] = {
      static_cast<float>(std::log2(std::max<double>(problem.size.m, 1.0))),
      static_cast<float>(std::log2(std::max<double>(problem.size.n, 1.0))),
      static_cast<float>(std::log2(std::max<double>(problem.size.k, 1.0))),
      static_cast<float>(g_w.w_b * std::log2(std::max<double>(problem.batch, 1.0))),
    };
    thread_local std::vector<std::pair<float, std::uint32_t>> by_dist;
    by_dist.clear();
    by_dist.reserve(g_w.n_clusters);
    for (std::uint32_t i = 0; i < g_w.n_clusters; ++i) {
      if (i == static_cast<std::uint32_t>(cid)) continue;
      const float* c = &g_w.centroids[i * 4];
      float d = 0.0f;
      for (int j = 0; j < 4; ++j) { float dj = coord[j] - c[j]; d += dj * dj; }
      by_dist.emplace_back(d, i);
    }
    std::sort(by_dist.begin(), by_dist.end());
    for (const auto& kv : by_dist) {
      const ClusterModel& ncm = g_w.clusters[kv.second];
      for (const auto& sig : ncm.tile_order_signatures) {
        auto it = s_sig_to_local.find(sig);
        if (it != s_sig_to_local.end()) cand_idx.push_back(it->second);
      }
      if (!cand_idx.empty()) break;
    }

    if (cand_idx.empty()) {
      std::size_t cap = std::min<std::size_t>(64, configs.size());
      cand_idx.resize(cap);
      for (std::size_t j = 0; j < cap; ++j) cand_idx[j] = static_cast<std::uint32_t>(j);
    }
  }

  if (s_ml_timing) t_setup_end = std::chrono::steady_clock::now();

  thread_local std::array<float, kFeatureCount> raw0;

  std::chrono::steady_clock::time_point t_q_features_start, t_q_features_end,
      t_q_mlp_end;
  if (s_ml_timing) t_q_features_start = std::chrono::steady_clock::now();
  compute_features_raw(problem, configs[cand_idx[0]], hw, raw0.data());
  if (s_ml_timing) t_q_features_end = std::chrono::steady_clock::now();
  thread_local std::vector<float> q_norm(64);
  q_norm.resize(g_w.query_dim);
  for (std::size_t k = 0; k < g_w.query_dim; ++k) {
    float v = raw0[kFeaturePermutation[k]];
    float s = std::max(cm.query_std[k], 1e-6f);
    q_norm[k] = (v - cm.query_mean[k]) / s;
  }
  thread_local std::vector<float> q_h0(256), q_h2(256), q_emb(64);
  q_h0.resize(cm.hidden_dim); q_h2.resize(cm.hidden_dim); q_emb.resize(cm.embed_dim);
  linear_relu(cm.q_w0.data(), cm.q_b0.data(), q_norm.data(),
              cm.hidden_dim, g_w.query_dim, q_h0.data());
  linear_relu(cm.q_w2.data(), cm.q_b2.data(), q_h0.data(),
              cm.hidden_dim, cm.hidden_dim, q_h2.data());
  linear(cm.q_w4.data(), cm.q_b4.data(), q_h2.data(),
         cm.embed_dim, cm.hidden_dim, q_emb.data());
  if (s_ml_timing) t_q_mlp_end = std::chrono::steady_clock::now();

  const float temp = std::max(std::abs(cm.temperature), 0.1f);

  thread_local std::array<float, kFeatureCount> raw_j;
  thread_local std::vector<float> ix_norm(64), ix_h(64);
  ix_norm.resize(g_w.inter_dim); ix_h.resize(cm.inter_hidden);
  std::vector<std::pair<int, float>> scored;
  scored.reserve(cand_idx.size());

  static const std::vector<float> empty_emb;
  for (std::uint32_t cidx : cand_idx) {
    const config_t& cfg = configs[cidx];
    auto sig = signature_for_config(cfg);

    auto it = cm.item_emb_cache.find(sig);
    const float* item_emb_ptr = nullptr;
    std::vector<float> item_emb_local;
    bool features_already_computed = false;
    if (it != cm.item_emb_cache.end()) {
      item_emb_ptr = it->second.data();
    } else {
      compute_features_raw(problem, cfg, hw, raw_j.data());
      features_already_computed = true;
      std::vector<float> item_in(g_w.item_dim);
      for (std::size_t k = 0; k < g_w.item_dim; ++k)
        item_in[k] = raw_j[kFeaturePermutation[g_w.query_dim + k]];
      item_emb_local.resize(cm.embed_dim);
      compute_item_embedding(cm, item_in.data(), item_emb_local.data(), g_w.item_dim);
      item_emb_ptr = item_emb_local.data();
    }

    thread_local std::array<float, 47> inter_only;
    if (!features_already_computed) {
      compute_features_raw_inter_only(problem, cfg, hw, inter_only.data());
      for (std::size_t k = 0; k < g_w.inter_dim; ++k) {
        int raw_idx = kFeaturePermutation[g_w.query_dim + g_w.item_dim + k];
        float v = inter_only[raw_idx - 84];
        float s = std::max(cm.inter_std[k], 1e-6f);
        ix_norm[k] = (v - cm.inter_mean[k]) / s;
      }
    } else {

      for (std::size_t k = 0; k < g_w.inter_dim; ++k) {
        float v = raw_j[kFeaturePermutation[g_w.query_dim + g_w.item_dim + k]];
        float s = std::max(cm.inter_std[k], 1e-6f);
        ix_norm[k] = (v - cm.inter_mean[k]) / s;
      }
    }
    linear_relu(cm.x_w0.data(), cm.x_b0.data(), ix_norm.data(),
                cm.inter_hidden, g_w.inter_dim, ix_h.data());
    float inter_score = cm.x_b2[0];
    inter_score += s_dot(cm.x_w2.data(), ix_h.data(), cm.inter_hidden);

    float dotp = s_dot(q_emb.data(), item_emb_ptr, cm.embed_dim);
    float total = dotp / temp + inter_score;
    scored.emplace_back(static_cast<int>(cidx), total);
  }

  if (s_ml_timing) t_score_end = std::chrono::steady_clock::now();

  if (s_top_k_only > 0 && (std::size_t)s_top_k_only < scored.size()) {
    std::partial_sort(scored.begin(), scored.begin() + s_top_k_only, scored.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
  } else {
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
  }
  if (s_ml_timing) t_sort_end = std::chrono::steady_clock::now();

  std::vector<prediction_result_t> result;

  std::size_t n_top = (s_top_k_only > 0) ? std::min((std::size_t)s_top_k_only, scored.size())
                                          : scored.size();
  result.reserve(s_top_k_only > 0 ? n_top : configs.size());
  for (std::size_t i = 0; i < n_top; ++i) {
    auto& s = scored[i];
    prediction_result_t r{static_cast<double>(-s.second), configs[s.first]};
    result.push_back(r);
  }

  if (s_top_k_only == 0) {
    std::vector<bool> used(configs.size(), false);
    for (auto& s : scored) used[s.first] = true;
    for (std::size_t j = 0; j < configs.size(); ++j) {
      if (!used[j]) {
        prediction_result_t r{std::numeric_limits<double>::quiet_NaN(), configs[j]};
        result.push_back(r);
      }
    }
  }
  if (s_ml_timing) {
    auto t_end = std::chrono::steady_clock::now();
    auto us = [](auto a, auto b){ return std::chrono::duration<double, std::micro>(b - a).count(); };
    std::fprintf(stderr,
      "[ML_TIMING] sel_us=%.2f cache_us=%.2f route_us=%.2f score_us=%.2f sort_us=%.2f result_us=%.2f setup_us=%.2f q_feat_us=%.2f q_mlp_us=%.2f per_cand_us=%.2f rebuild=%d cid=%d n_cand=%zu n_configs=%zu top_k=%d\n",
      us(t_total_start, t_end),
      us(t_total_start, t_cache_end),
      us(t_route_start, t_route_end),
      us(t_route_end, t_score_end),
      us(t_score_end, t_sort_end),
      us(t_sort_end, t_end),
      us(t_route_end, t_setup_end),
      us(t_q_features_start, t_q_features_end),
      us(t_q_features_end, t_q_mlp_end),
      us(t_q_mlp_end, t_score_end),
      _ml_did_rebuild ? 1 : 0,
      cid, scored.size(), configs.size(), s_top_k_only);
    std::fflush(stderr);
  }
  return result;
}

namespace {
struct EagerInit {
  EagerInit() {
    if (std::getenv("ML_RECOMMENDER_NO_EAGER_INIT") != nullptr) return;
    if (std::getenv("ML_RECOMMENDER_WEIGHTS") == nullptr) return;
    ensure_weights();
  }
};
static EagerInit s_eager_init_instance;
}

}
}
