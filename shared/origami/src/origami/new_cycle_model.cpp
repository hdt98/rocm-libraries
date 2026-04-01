#include "origami/new_cycle_model/predict.hpp"
#include "origami/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

namespace origami {
namespace new_cycle_model {

namespace {

int cdiv(int a, int b) {
  if (b <= 0) return a;
  return (a + b - 1) / b;
}

struct kernel_info_t {
  int mt_m, mt_n, du;
  int mi_m, mi_n, mi_k;
  int miwt_m, miwt_n;
  int pgr;
  int grvw_a, grvw_b;
  bool dtl_a, dtl_b;
  int nta, ntb;
  int sk;
  bool cms;
  int lsu;
  int wave_num;
  int mcul;
  int a_bpe, b_bpe, d_bpe;
};

kernel_info_t extract_info(const problem_t& problem,
                           const config_t& config) {
  kernel_info_t ki{};
  ki.mt_m = static_cast<int>(config.mt.m);
  ki.mt_n = static_cast<int>(config.mt.n);
  ki.mi_m = static_cast<int>(config.mi.m);
  ki.mi_n = static_cast<int>(config.mi.n);
  ki.mi_k = static_cast<int>(config.mi.k);

  const auto& tp = config.tensile();
  ki.du = tp.depth_u > 0 ? static_cast<int>(tp.depth_u) : static_cast<int>(config.mt.k);
  ki.pgr = tp.prefetch_global_read;
  ki.grvw_a = static_cast<int>(config.grvw_a);
  ki.grvw_b = static_cast<int>(config.grvw_b);
  ki.dtl_a = tp.direct_to_lds_a;
  ki.dtl_b = tp.direct_to_lds_b;
  ki.nta = config.cache_hints_a;
  ki.ntb = config.cache_hints_b;
  ki.sk = (tp.global_split_u > 1 || tp.global_accumulation >= 2) ? 3 : 0;
  ki.cms = config.hand_optimized_main_loop;
  ki.lsu = tp.local_split_u;
  ki.wave_num = static_cast<int>(tp.wave_num);
  ki.mcul = tp.math_clocks_unrolled_loop;

  ki.miwt_m = tp.wave_group_m;
  ki.miwt_n = tp.wave_group_n;

  ki.a_bpe = std::max(1, static_cast<int>(data_type_to_bytes(problem.a_dtype)));
  ki.b_bpe = std::max(1, static_cast<int>(data_type_to_bytes(problem.b_dtype)));
  ki.d_bpe = std::max(1, static_cast<int>(data_type_to_bytes(problem.d_dtype)));

  return ki;
}

double estimate_compute_per_iter(const kernel_info_t& ki, const model_params_t& p) {
  if (ki.mcul > 0) return static_cast<double>(ki.mcul);

  int lsu = std::max(ki.lsu, 1);
  int du_per_wave = ki.du / (lsu > 1 ? lsu : 1);
  int mi_k = std::max(ki.mi_k, 1);
  int k_steps = cdiv(du_per_wave, mi_k);
  int n_mfma_per_step = ki.miwt_m * ki.miwt_n;
  int n_mfma = n_mfma_per_step * k_steps;

  int issue_lat = 14;
  if (ki.a_bpe == 4) issue_lat = 26;

  double cycles_qc = static_cast<double>(n_mfma) * issue_lat;
  if (ki.cms) cycles_qc *= p.cms_discount;
  return cycles_qc * 4.0;
}

double estimate_memory_per_iter(const kernel_info_t& ki,
                                const problem_t& problem,
                                int n_active,
                                const model_params_t& p) {
  double la = static_cast<double>(ki.mt_m) * ki.du * ki.a_bpe;
  double lb = static_cast<double>(ki.mt_n) * ki.du * ki.b_bpe;
  double load_bytes = la + lb;

  double bw_frac = std::min(1.0, p.bw_ramp_sqrt + p.bw_ramp_linear *
                   (static_cast<double>(n_active) / p.n_cu));
  double total_bw = p.hbm_read_bpc * bw_frac;
  double bw_per_cu = total_bw / std::max(n_active, 1);

  if (ki.nta && (problem.size.m * problem.size.k * ki.a_bpe > p.l2_capacity_per_xcd * p.n_xcd))
    bw_per_cu *= p.nt_large_benefit;
  else if (ki.nta)
    bw_per_cu *= p.nt_small_penalty;
  if (ki.ntb && (problem.size.n * problem.size.k * ki.b_bpe > p.l2_capacity_per_xcd * p.n_xcd))
    bw_per_cu *= p.nt_large_benefit;
  else if (ki.ntb)
    bw_per_cu *= p.nt_small_penalty;

  double ws_per_iter = (static_cast<double>(ki.mt_m) * ki.a_bpe +
                        static_cast<double>(ki.mt_n) * ki.b_bpe) * ki.du;
  double ws_ratio = ws_per_iter / p.l2_capacity_per_xcd;
  int cus_per_xcd = std::max(1, cdiv(n_active, p.n_xcd));
  double total_ws = ws_ratio * cus_per_xcd;
  if (total_ws > 1.0) {
    double pressure = std::min(total_ws, 4.0);
    bw_per_cu /= (1.0 + (pressure - 1.0) * p.l2_pressure_scale);
  }

  double avg_grvw = (ki.grvw_a + ki.grvw_b) / 2.0;
  double grvw_eff = std::min(1.0, p.grvw_eff_base + p.grvw_eff_slope * (avg_grvw / 8.0));
  bw_per_cu *= grvw_eff;

  return load_bytes / std::max(bw_per_cu, 1e-9);
}

struct sched_result_t {
  int tiles;
  int n_active;
  int n_waves;
  int sf;
  bool sk_expanded;
};

sched_result_t compute_scheduling(const kernel_info_t& ki,
                                  const problem_t& problem,
                                  const model_params_t& p) {
  int gm = cdiv(static_cast<int>(problem.size.m), ki.mt_m);
  int gn = cdiv(static_cast<int>(problem.size.n), ki.mt_n);
  int tiles = gm * gn * static_cast<int>(problem.batch);
  int kpt = cdiv(static_cast<int>(problem.size.k), ki.du);

  sched_result_t r{};
  r.tiles = tiles;
  r.sk_expanded = false;

  if (tiles >= p.n_cu) {
    r.sf = 1;
    r.n_active = p.n_cu;
    r.n_waves = cdiv(tiles, p.n_cu);
  } else if (ki.sk > 0 && kpt >= p.sk_kpt_min) {
    int best_grid = tiles;
    for (int f : {16, 12, 8, 6, 4, 3, 2, 1}) {
      if (tiles * f <= p.n_cu && kpt / f >= p.sk_kpt_min) {
        best_grid = tiles * f;
        break;
      }
    }
    r.sf = std::max(1, cdiv(best_grid, tiles));
    int n_wgs = tiles * r.sf;
    r.n_active = std::min(n_wgs, p.n_cu);
    r.n_waves = cdiv(n_wgs, p.n_cu);
    r.sk_expanded = (n_wgs >= static_cast<int>(p.n_cu * p.sk_cu_fill_threshold));
  } else {
    r.sf = 1;
    r.n_active = std::min(tiles, p.n_cu);
    r.n_waves = cdiv(tiles, p.n_cu);
  }
  return r;
}

} // anonymous namespace

double predict(const problem_t& problem,
               const hardware_t& hardware,
               const config_t& config,
               const model_params_t& params) {
  auto ki = extract_info(problem, config);

  if (ki.mt_m <= 0 || ki.mt_n <= 0 || ki.du <= 0)
    return std::numeric_limits<double>::max();

  if (ki.mi_m == 1 && ki.mi_n == 1 && ki.mi_k == 64 &&
      static_cast<int>(problem.size.m) > 2)
    return std::numeric_limits<double>::max();

  auto sched = compute_scheduling(ki, problem, params);
  if (sched.n_active == 0)
    return std::numeric_limits<double>::max();

  double compute = estimate_compute_per_iter(ki, params);
  double memory = estimate_memory_per_iter(ki, problem, sched.n_active, params);

  int gm = cdiv(static_cast<int>(problem.size.m), ki.mt_m);
  int gn = cdiv(static_cast<int>(problem.size.n), ki.mt_n);

  double iter_cost;
  if (sched.sk_expanded) {
    double cu_fill_ratio = std::min(1.0,
        static_cast<double>(sched.tiles * sched.sf) / params.n_cu);
    double cw = params.sk_blend_base + params.sk_blend_cu_scale * cu_fill_ratio;
    double mw = 1.0 - cw;
    int ni_split = std::max(1,
        cdiv(cdiv(static_cast<int>(problem.size.k), sched.sf), ki.du));
    iter_cost = cw * compute + mw * memory +
                params.sk_coord_cycles * sched.sf / std::max(ni_split, 1);
  } else if (ki.pgr >= 1) {
    iter_cost = std::max(compute, memory);
  } else {
    iter_cost = compute + memory;
  }

  double m_os = ki.mt_m > static_cast<int>(problem.size.m) ?
      static_cast<double>(ki.mt_m) / std::max(static_cast<int>(problem.size.m), 1) : 1.0;
  double n_os = ki.mt_n > static_cast<int>(problem.size.n) ?
      static_cast<double>(ki.mt_n) / std::max(static_cast<int>(problem.size.n), 1) : 1.0;
  double oversize = std::max(m_os, n_os);

  double launched_mn = static_cast<double>(gm * ki.mt_m) * (gn * ki.mt_n);
  double problem_mn = static_cast<double>(problem.size.m) * problem.size.n;
  double mn_waste = launched_mn / std::max(problem_mn, 1.0);

  double util_scale = 1.0 + (mn_waste - 1.0) * params.mn_waste_weight +
                      std::max(0.0, oversize - 1.5) * params.oversize_weight;
  iter_cost *= util_scale;

  int k_per_split = cdiv(static_cast<int>(problem.size.k), sched.sf);
  int n_iters = cdiv(k_per_split, ki.du);

  bool dtl = ki.dtl_a || ki.dtl_b;
  double pio = params.per_iter_overhead * (dtl ? params.dtl_overhead_discount : 1.0);
  double loop_cost = n_iters * (iter_cost + pio);

  int occ_tiles = std::min(std::max(1, cdiv(gm * gn * static_cast<int>(problem.batch) * sched.sf,
                                            params.n_cu)), 10);
  double occ_factor = std::pow(params.occ_decay_base, occ_tiles);

  double prologue;
  if (sched.sk_expanded)
    prologue = compute * 0.3 * occ_factor;
  else
    prologue = memory * ki.pgr * params.pgr_prologue_scale * occ_factor;

  int d_lines = cdiv(ki.mt_m * ki.d_bpe, params.cache_line_bytes);
  double d_write = static_cast<double>(d_lines) * params.cache_line_bytes * ki.mt_n;
  double w_bw = params.hbm_write_bpc *
                std::min(static_cast<double>(sched.n_active) / params.n_cu, 1.0) /
                std::max(std::min(sched.tiles, sched.n_active), 1);
  double epilogue = d_write / std::max(w_bw, 1e-9);

  int kr = k_per_split % ki.du;
  double tail = (kr > 0 && ki.du > 1) ?
      iter_cost * (static_cast<double>(kr) / ki.du) * params.tail_factor : 0.0;

  epilogue = (epilogue + tail + params.epilogue_base) * occ_factor;

  double tile_sched = (sched.n_active < params.n_cu) ?
      params.wave_startup * ki.wave_num : 0.0;

  double lsu_cycles = 0.0;
  if (ki.lsu > 1)
    lsu_cycles = static_cast<double>(ki.mt_m) * ki.mt_n * 4.0 / 256 + 200 * ki.lsu;

  double sk_cycles = 0.0;
  if (sched.sf > 1) {
    double partial = static_cast<double>(gm) * gn * ki.mt_m * ki.mt_n * 4.0 * 2;
    sk_cycles = partial / std::max(params.hbm_write_bpc, 1e-9) +
                params.sk_reduce_base + params.sk_per_split * sched.sf;
  }

  double per_tile = prologue + loop_cost + epilogue + lsu_cycles;
  double total = per_tile * sched.n_waves + params.dispatch_cycles + tile_sched + sk_cycles;

  return total / (params.sclk_ghz * 1000.0);
}

bool is_enabled() {
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ORIGAMI_NEW_CYCLE_MODEL");
    cached = (env && std::string(env) != "0") ? 1 : 0;
  }
  return cached == 1;
}

} // namespace new_cycle_model
} // namespace origami
