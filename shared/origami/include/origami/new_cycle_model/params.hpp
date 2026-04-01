#pragma once

namespace origami {
namespace new_cycle_model {

struct model_params_t {
  double sclk_ghz = 2.2;
  double hbm_read_bpc = 2.413;
  double hbm_write_bpc = 1.207;
  double l2_bpc = 8.0;
  int n_cu = 256;
  int n_xcd = 8;
  int cache_line_bytes = 128;
  double l2_capacity_per_xcd = 32.0 * 1024 * 1024;

  double bw_ramp_sqrt = 0.280;
  double bw_ramp_linear = 2.867;
  double l2_bonus_eff = 1.153;
  double nt_large_benefit = 3.539;
  double nt_small_penalty = 5.901;
  double util_weight = 0.15;
  double oversize_weight = 0.3;
  double pgr_prologue_scale = 0.044;
  double occ_decay_base = 1.129;
  double dispatch_cycles = 89468.0;
  double wave_startup = 11238.0;
  double per_iter_overhead = 5951.0;
  double dtl_overhead_discount = 0.743;
  double cms_discount = 2.224;
  double sk_coord_cycles = 481.0;
  double sk_reduce_base = 14942.0;
  double sk_per_split = 164.0;
  double epilogue_base = 1308.0;
  double tail_factor = 1.5;
  double l2_pressure_scale = 1.758;
  double grvw_eff_base = 0.038;
  double grvw_eff_slope = 0.082;

  int sk_kpt_min = 24;
  double sk_cu_fill_threshold = 1.0;

  double sk_blend_base = 0.955;
  double sk_blend_cu_scale = 4.766;
  double mn_waste_weight = 4.582;

  double xf32_mfma_multiplier = 7.572;
  double xf32_cvt_overhead = 79.75;
};

} // namespace new_cycle_model
} // namespace origami
