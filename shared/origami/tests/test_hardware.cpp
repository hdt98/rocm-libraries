// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

using Catch::Approx;

TEST_CASE("hardware_t: get_hardware_for_arch populates formocast fields for gfx942", "[hardware]") {
  auto arch = origami::hardware_t::architecture_t::gfx942;
  auto hw   = origami::hardware_t::get_hardware_for_arch(arch, 304, 65536, 4194304, 1100000);

  SECTION("cache hierarchy") {
    REQUIRE(hw.L1_capacity == 32768);
    REQUIRE(hw.L2_capacity == 4194304);
    REQUIRE(hw.L3_capacity == 268435456);
    REQUIRE(hw.L1_cache_line_size == 128);
  }

  SECTION("bus widths") {
    REQUIRE(hw.L1_bus_width_per_cu == 64);
    REQUIRE(hw.L2_bus_width_per_cu == 128);
    REQUIRE(hw.L1_write_bus_width_per_cu == 64);
    REQUIRE(hw.L2_write_bus_width_per_cu == 64);
  }

  SECTION("frequencies and bandwidth") {
    REQUIRE(hw.wavefront_size == 64);
    REQUIRE(hw.hbm_bandwidth == Approx(30000.0 / 13.0));
    REQUIRE(hw.L3_bandwidth == Approx(60000.0 / 13.0));
    REQUIRE(hw.boost_clock_ghz == Approx(2.2));
    REQUIRE(hw.mem_frequency_mhz > 0);
  }

  SECTION("model parameters") {
    REQUIRE(hw.initial_cost == Approx(2.7));
    REQUIRE(hw.L2_read_arb_eff == Approx(0.9));
    REQUIRE(hw.L2_write_arb_eff == Approx(0.58));
  }

  SECTION("LDS latency model") {
    REQUIRE(hw.local_read_latency_b128 == 10);
    REQUIRE(hw.local_read_latency_b64 == 5);
    REQUIRE(hw.local_read_latency_b32 == 2);
    REQUIRE(hw.local_read_conflict_b128 == 6);
    REQUIRE(hw.local_read_conflict_b64 == 3);
    REQUIRE(hw.local_read_conflict_b32 == 3);
    REQUIRE(hw.local_write_latency_b128 == 10);
    REQUIRE(hw.local_write_latency_b64 == 10);
    REQUIRE(hw.local_write_latency_b32 == 10);
    REQUIRE(hw.local_write_conflict_b128 == 4);
    REQUIRE(hw.local_write_conflict_b64 == 2);
    REQUIRE(hw.local_write_conflict_b32 == 1);
  }

  SECTION("existing fields still correct") {
    REQUIRE(hw.arch == arch);
    REQUIRE(hw.N_CU == 304);
    REQUIRE(hw.NUM_XCD == 8);
    REQUIRE(hw.compute_clock_ghz == Approx(1.1));
  }
}

TEST_CASE("hardware_t: get_hardware_for_arch populates formocast fields for gfx950", "[hardware]") {
  auto arch = origami::hardware_t::architecture_t::gfx950;
  auto hw   = origami::hardware_t::get_hardware_for_arch(arch, 256, 163840, 4194304, 1800000);

  SECTION("cache hierarchy") {
    REQUIRE(hw.L1_capacity == 32768);
    REQUIRE(hw.L3_capacity == 268435456);
    REQUIRE(hw.L1_cache_line_size == 128);
  }

  SECTION("bus widths") {
    REQUIRE(hw.L1_bus_width_per_cu == 64);
    REQUIRE(hw.L2_bus_width_per_cu == 128);
    REQUIRE(hw.L1_write_bus_width_per_cu == 64);
    REQUIRE(hw.L2_write_bus_width_per_cu == 64);
  }

  SECTION("frequencies and bandwidth") {
    REQUIRE(hw.wavefront_size == 64);
    REQUIRE(hw.hbm_bandwidth == Approx(30000.0 / 19.0));
    REQUIRE(hw.L3_bandwidth == Approx(60000.0 / 19.0));
    REQUIRE(hw.boost_clock_ghz == Approx(2.35));
  }

  SECTION("model parameters") {
    REQUIRE(hw.initial_cost == Approx(2.6));
    REQUIRE(hw.L2_read_arb_eff == Approx(0.9));
    REQUIRE(hw.L2_write_arb_eff == Approx(0.75));
  }

  SECTION("LDS latency model") {
    REQUIRE(hw.local_read_latency_b128 == 14);
    REQUIRE(hw.local_read_latency_b64 == 10);
    REQUIRE(hw.local_read_latency_b32 == 10);
    REQUIRE(hw.local_read_conflict_b128 == 6);
    REQUIRE(hw.local_read_conflict_b64 == 3);
    REQUIRE(hw.local_read_conflict_b32 == 3);
  }

  SECTION("existing fields still correct") {
    REQUIRE(hw.arch == arch);
    REQUIRE(hw.N_CU == 256);
    REQUIRE(hw.NUM_XCD == 8);
  }
}

TEST_CASE("hardware_t: get_hardware_for_arch populates formocast fields for gfx1201", "[hardware]") {
  auto arch = origami::hardware_t::architecture_t::gfx1201;
  auto hw   = origami::hardware_t::get_hardware_for_arch(arch, 64, 65536, 8388608, 2350000);

  SECTION("cache hierarchy") {
    REQUIRE(hw.L1_capacity == 32768);
    REQUIRE(hw.L3_capacity == 67108864);
    REQUIRE(hw.L1_cache_line_size == 128);
  }

  SECTION("bus widths — different from CDNA") {
    REQUIRE(hw.L1_bus_width_per_cu == 128);
    REQUIRE(hw.L2_bus_width_per_cu == 128);
    REQUIRE(hw.L1_write_bus_width_per_cu == 64);
    REQUIRE(hw.L2_write_bus_width_per_cu == 128);
  }

  SECTION("RDNA wavefront and frequencies") {
    REQUIRE(hw.wavefront_size == 32);
    REQUIRE(hw.hbm_bandwidth == Approx(61.04));
    REQUIRE(hw.L3_bandwidth == Approx(439.45));
    REQUIRE(hw.boost_clock_ghz == Approx(2.5));
  }

  SECTION("model parameters") {
    REQUIRE(hw.initial_cost == Approx(14.6));
    REQUIRE(hw.L2_read_arb_eff == Approx(0.9));
    REQUIRE(hw.L2_write_arb_eff == Approx(0.75));
  }
}

TEST_CASE("hardware_t: direct constructor leaves formocast fields at defaults", "[hardware]") {
  auto hw = origami::hardware_t(origami::hardware_t::architecture_t::gfx942,
                                304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.4, 1,
                                std::make_tuple(0.0, 0.015, 0.0));

  REQUIRE(hw.L1_capacity == 0);
  REQUIRE(hw.wavefront_size == 0);
  REQUIRE(hw.hbm_bandwidth == 0.0);
  REQUIRE(hw.local_read_latency_b128 == 0);
}

TEST_CASE("hardware_t: copy preserves formocast fields", "[hardware]") {
  auto arch = origami::hardware_t::architecture_t::gfx950;
  auto hw1  = origami::hardware_t::get_hardware_for_arch(arch, 256, 163840, 4194304, 1800000);
  auto hw2  = hw1;

  REQUIRE(hw2.L1_capacity == hw1.L1_capacity);
  REQUIRE(hw2.L3_capacity == hw1.L3_capacity);
  REQUIRE(hw2.wavefront_size == hw1.wavefront_size);
  REQUIRE(hw2.hbm_bandwidth == hw1.hbm_bandwidth);
  REQUIRE(hw2.L3_bandwidth == hw1.L3_bandwidth);
  REQUIRE(hw2.boost_clock_ghz == hw1.boost_clock_ghz);
  REQUIRE(hw2.mem_frequency_mhz == hw1.mem_frequency_mhz);
  REQUIRE(hw2.initial_cost == hw1.initial_cost);
  REQUIRE(hw2.L2_read_arb_eff == hw1.L2_read_arb_eff);
  REQUIRE(hw2.L2_write_arb_eff == hw1.L2_write_arb_eff);
  REQUIRE(hw2.local_read_latency_b128 == hw1.local_read_latency_b128);
  REQUIRE(hw2.local_write_conflict_b32 == hw1.local_write_conflict_b32);
}

TEST_CASE("hardware_t: wavefront_size set for all architectures", "[hardware]") {
  struct ArchWavefront {
    origami::hardware_t::architecture_t arch;
    size_t expected_wavefront;
  };

  std::vector<ArchWavefront> cases = {
    {origami::hardware_t::architecture_t::gfx90a,  64},
    {origami::hardware_t::architecture_t::gfx942,  64},
    {origami::hardware_t::architecture_t::gfx950,  64},
    {origami::hardware_t::architecture_t::gfx1201, 32},
    {origami::hardware_t::architecture_t::gfx1100, 32},
    {origami::hardware_t::architecture_t::gfx1150, 32},
    {origami::hardware_t::architecture_t::gfx1151, 32},
    {origami::hardware_t::architecture_t::gfx1152, 32},
    {origami::hardware_t::architecture_t::gfx1153, 32},
  };

  for (const auto& tc : cases) {
    auto hw = origami::hardware_t::get_hardware_for_arch(tc.arch, 64, 65536, 4000000, 1000000);
    REQUIRE(hw.wavefront_size == tc.expected_wavefront);
  }
}
