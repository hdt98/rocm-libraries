// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Implementation of the header-only HIP/GPU mxDataGenerator backend.
// Included from `DataGeneratorGPU.hpp`. Do NOT include directly.

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// Provides __amd_cvt_floatx{2,8,32}_to_fp{4,6,8}*_scale and __amd_scale_t.
// Hits the gfx950 hardware MX convert instructions where available
// (v_cvt_scalef32_pk_fp4_f32 / pk32_f32_fp6 / pk_fp8_f32 etc.); falls back to
// the canonical fcbx software path on other architectures. Same wrapper
// hipblaslt's hipblaslt_float4 / _float6 / _float8 types use, so generator
// output is bit-identical to what those types produce.
#include <hip/hip_ext_ocp.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataTypeInfo.hpp"
#include "ocp_e2m1_mxfp4.hpp"
#include "ocp_e2m3_mxfp6.hpp"
#include "ocp_e3m2_mxfp6.hpp"
#include "ocp_e4m3_mxfp8.hpp"
#include "ocp_e5m2_mxfp8.hpp"

namespace DGen
{
    namespace gpu_detail
    {
        // ----------------------------------------------------------------------
        // HIP error helpers
        // ----------------------------------------------------------------------
        inline void checkHipStatus(hipError_t status, char const* what)
        {
            if(status != hipSuccess)
            {
                throw std::runtime_error(std::string("DataGeneratorGPU: ") + what + ": "
                                         + hipGetErrorString(status));
            }
        }

#define DGEN_GPU_CHECK_HIP(call) \
    ::DGen::gpu_detail::checkHipStatus((call), #call)

        // ----------------------------------------------------------------------
        // Device-side init mode tags
        //
        // We can't pass the host-side `std::variant<...>` to a kernel, so we
        // mirror the variant alternatives as a plain enum and dispatch on the
        // host before launching the kernel.
        // ----------------------------------------------------------------------
        enum class DeviceInitMode : int
        {
            Bounded                = 0,
            BoundedAlternatingSign = 1,
            Unbounded              = 2,
            Identity               = 3,
            Ones                   = 4,
            Zeros                  = 5,
            Sequential             = 6,
            RowIndex               = 7,
            ColIndex               = 8,
            Checkerboard           = 9,
            ScaledDiagonal         = 10,
            TrigonometricFromFloat = 11,
            NormalFromFloat        = 12,
        };

        struct DeviceInitConfig
        {
            DeviceInitMode mode    = DeviceInitMode::Bounded;
            float          minVal  = -1.0f;
            float          maxVal  = 1.0f;
            float          mean    = 0.0f;
            float          stdDev  = 1.0f;
            // `dim0` is the size of the fastest-varying dimension (row count
            // for column-major matrices). Used by RowIndex/ColIndex/Identity.
            uint64_t       dim0    = 0;
        };

        inline DeviceInitConfig makeConfig(DataGeneratorOptions const& options,
                                           std::vector<index_t> const& sizes)
        {
            DeviceInitConfig cfg;
            cfg.minVal = static_cast<float>(options.min);
            cfg.maxVal = static_cast<float>(options.max);
            cfg.dim0   = sizes.empty() ? 0 : static_cast<uint64_t>(sizes[0]);

            std::visit(
                [&](auto const& mode) {
                    using T = std::decay_t<decltype(mode)>;
                    if constexpr(std::is_same_v<T, Bounded>)
                        cfg.mode = DeviceInitMode::Bounded;
                    else if constexpr(std::is_same_v<T, BoundedAlternatingSign>)
                        cfg.mode = DeviceInitMode::BoundedAlternatingSign;
                    else if constexpr(std::is_same_v<T, Unbounded>)
                        cfg.mode = DeviceInitMode::Unbounded;
                    else if constexpr(std::is_same_v<T, Identity>)
                        cfg.mode = DeviceInitMode::Identity;
                    else if constexpr(std::is_same_v<T, Ones>)
                        cfg.mode = DeviceInitMode::Ones;
                    else if constexpr(std::is_same_v<T, Zeros>)
                        cfg.mode = DeviceInitMode::Zeros;
                    else if constexpr(std::is_same_v<T, Sequential>)
                        cfg.mode = DeviceInitMode::Sequential;
                    else if constexpr(std::is_same_v<T, RowIndex>)
                        cfg.mode = DeviceInitMode::RowIndex;
                    else if constexpr(std::is_same_v<T, ColIndex>)
                        cfg.mode = DeviceInitMode::ColIndex;
                    else if constexpr(std::is_same_v<T, Checkerboard>)
                        cfg.mode = DeviceInitMode::Checkerboard;
                    else if constexpr(std::is_same_v<T, ScaledDiagonal>)
                        cfg.mode = DeviceInitMode::ScaledDiagonal;
                    else if constexpr(std::is_same_v<T, TrigonometricFromFloat>)
                        cfg.mode = DeviceInitMode::TrigonometricFromFloat;
                    else if constexpr(std::is_same_v<T, NormalFromFloat>)
                    {
                        cfg.mode   = DeviceInitMode::NormalFromFloat;
                        cfg.mean   = static_cast<float>(mode.mean);
                        cfg.stdDev = static_cast<float>(mode.std_dev);
                    }
                    // Note: the constant-fill / pathological modes (Twos,
                    // NegOnes, MaxVals, DenormMins, DenormMaxs, NaNs, Infs)
                    // and RandInt are handled by the host-fallback path in
                    // `generateInto` and never reach this visitor.
                },
                options.initMode);

            return cfg;
        }

        // The on-device kernel quantises a float-per-element through the
        // block-max scale derivation, which can't reproduce the exact bit
        // patterns these modes need (e.g. NaN propagation, denorm-min, a
        // constant 2.0 with scale=1.0 instead of an auto-derived scale, or
        // exact-integer values that would lose their integer-ness once they
        // pass through the device PRNG and the auto-derived scale).  Route
        // them through the host CPU `DataGenerator<DTYPE>` and then
        // hipMemcpy the bytes to device.
        inline bool requiresHostFallback(DataInitMode const& initMode)
        {
            return std::visit(
                [](auto const& mode) -> bool {
                    using T = std::decay_t<decltype(mode)>;
                    return std::is_same_v<T, Twos> || std::is_same_v<T, NegOnes>
                           || std::is_same_v<T, MaxVals> || std::is_same_v<T, DenormMins>
                           || std::is_same_v<T, DenormMaxs> || std::is_same_v<T, NaNs>
                           || std::is_same_v<T, Infs> || std::is_same_v<T, RandInt>;
                },
                initMode);
        }

        // ----------------------------------------------------------------------
        // Per-DTYPE compile-time traits
        //
        // Centralises the dtype-dependent constants used by the generic
        // device-side quantizer so we don't have to specialise every kernel.
        // ----------------------------------------------------------------------
        template <typename DTYPE>
        struct GpuTraits;

        template <>
        struct GpuTraits<ocp_e2m1_mxfp4>
        {
            static constexpr int   signBits     = 1;
            static constexpr int   expBits      = 2;
            static constexpr int   mantBits     = 1;
            static constexpr int   bias         = 1;
            static constexpr float maxNormal    = 6.0f;
            static constexpr float minSubnormal = 0.5f;
            static constexpr int   bitsPerElem  = 4;
            using Scale                         = ScaleInfo<ScaleType::E8M0>;
            static constexpr ScaleType scaleKind = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e2m1_mxfp4_e4m3>
        {
            static constexpr int   signBits     = 1;
            static constexpr int   expBits      = 2;
            static constexpr int   mantBits     = 1;
            static constexpr int   bias         = 1;
            static constexpr float maxNormal    = 6.0f;
            static constexpr float minSubnormal = 0.5f;
            static constexpr int   bitsPerElem  = 4;
            using Scale                         = ScaleInfo<ScaleType::E4M3>;
            static constexpr ScaleType scaleKind = ScaleType::E4M3;
        };

        template <>
        struct GpuTraits<ocp_e2m1_mxfp4_e5m3>
        {
            static constexpr int   signBits     = 1;
            static constexpr int   expBits      = 2;
            static constexpr int   mantBits     = 1;
            static constexpr int   bias         = 1;
            static constexpr float maxNormal    = 6.0f;
            static constexpr float minSubnormal = 0.5f;
            static constexpr int   bitsPerElem  = 4;
            using Scale                         = ScaleInfo<ScaleType::E5M3>;
            static constexpr ScaleType scaleKind = ScaleType::E5M3;
        };

        template <>
        struct GpuTraits<ocp_e2m3_mxfp6>
        {
            static constexpr int   signBits     = 1;
            static constexpr int   expBits      = 2;
            static constexpr int   mantBits     = 3;
            static constexpr int   bias         = 1;
            static constexpr float maxNormal    = 7.5f;
            static constexpr float minSubnormal = 0.125f;
            static constexpr int   bitsPerElem  = 6;
            using Scale                         = ScaleInfo<ScaleType::E8M0>;
            static constexpr ScaleType scaleKind = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e3m2_mxfp6>
        {
            static constexpr int   signBits     = 1;
            static constexpr int   expBits      = 3;
            static constexpr int   mantBits     = 2;
            static constexpr int   bias         = 3;
            static constexpr float maxNormal    = 28.0f;
            static constexpr float minSubnormal = 0.0625f;
            static constexpr int   bitsPerElem  = 6;
            using Scale                         = ScaleInfo<ScaleType::E8M0>;
            static constexpr ScaleType scaleKind = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e4m3_mxfp8>
        {
            static constexpr int   signBits     = 1;
            static constexpr int   expBits      = 4;
            static constexpr int   mantBits     = 3;
            static constexpr int   bias         = 7;
            static constexpr float maxNormal    = 448.0f;
            static constexpr float minSubnormal = 0.001953125f;
            static constexpr int   bitsPerElem  = 8;
            using Scale                         = ScaleInfo<ScaleType::E8M0>;
            static constexpr ScaleType scaleKind = ScaleType::E8M0;
        };

        template <>
        struct GpuTraits<ocp_e5m2_mxfp8>
        {
            static constexpr int   signBits     = 1;
            static constexpr int   expBits      = 5;
            static constexpr int   mantBits     = 2;
            static constexpr int   bias         = 15;
            static constexpr float maxNormal    = 57344.0f;
            static constexpr float minSubnormal = 1.52587890625e-05f;
            static constexpr int   bitsPerElem  = 8;
            using Scale                         = ScaleInfo<ScaleType::E8M0>;
            static constexpr ScaleType scaleKind = ScaleType::E8M0;
        };

        // ----------------------------------------------------------------------
        // Device PRNG (xorshift32 seeded per element)
        //
        // The seeding pattern intentionally mirrors the host CPU `m_seed + tid`
        // convention: `(seed, idx)` -> a 32-bit state with reasonable bit
        // diffusion. Sufficient for statistical tests; not cryptographic.
        // ----------------------------------------------------------------------
        __device__ __forceinline__ uint32_t seedMix(uint32_t seed, uint64_t idx)
        {
            uint32_t s = seed ^ static_cast<uint32_t>(idx * 2654435761u)
                         ^ static_cast<uint32_t>((idx >> 32) * 40503u);
            // A few xorshift rounds to spread out bits before sampling.
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if(s == 0)
                s = 0x9E3779B9u;
            return s;
        }

        __device__ __forceinline__ uint32_t xorshift32(uint32_t& s)
        {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            return s;
        }

        __device__ __forceinline__ float prngFloat01(uint32_t& s)
        {
            // 24-bit mantissa worth of randomness, scaled to [0, 1).
            uint32_t r = xorshift32(s) >> 8;
            return static_cast<float>(r) * (1.0f / 16777216.0f);
        }

        // ----------------------------------------------------------------------
        // Device-side raw value generation per init mode
        //
        // `dataIdx` is the linear element index (0 ... totalElements-1).
        // For 2D matrices column-major, row = dataIdx % dim0, col = dataIdx / dim0.
        // ----------------------------------------------------------------------
        __device__ __forceinline__ float
            genValue(DeviceInitMode mode, uint32_t seed, uint64_t dataIdx, DeviceInitConfig const& cfg)
        {
            uint64_t const dim0 = cfg.dim0 ? cfg.dim0 : 1;
            uint64_t const row  = dataIdx % dim0;
            uint64_t const col  = dataIdx / dim0;

            switch(mode)
            {
            case DeviceInitMode::Zeros:
                return 0.0f;
            case DeviceInitMode::Ones:
                return 1.0f;
            case DeviceInitMode::Identity:
                return (row == col) ? 1.0f : 0.0f;
            case DeviceInitMode::Sequential:
                return static_cast<float>(dataIdx);
            case DeviceInitMode::RowIndex:
                return static_cast<float>(row);
            case DeviceInitMode::ColIndex:
                return static_cast<float>(col);
            case DeviceInitMode::Checkerboard:
                return ((row ^ col) & 1ull) ? -1.0f : 1.0f;
            case DeviceInitMode::ScaledDiagonal:
                return (row == col) ? static_cast<float>(row + 1) : 0.0f;
            case DeviceInitMode::TrigonometricFromFloat:
            {
                // Match the CPU path's `generate_data_trigonometric_from_float`:
                // draw a uniform random angle in [0, 2pi) and return its
                // cosine (so the values are uniformly distributed on [-1, 1]).
                // The deterministic `sinf(dataIdx)` we used here previously
                // produced a smooth oscillating pattern with no PRNG mixing,
                // which exposed pathological cancellations in tests that
                // multiply A*B by per-row vectors (e.g. scaleAlphaVec) and
                // then norm-check the activated/biased result -- the
                // resulting reference values were small enough that the
                // relative error denominator blew up.
                uint32_t       s     = seedMix(seed, dataIdx);
                float          u     = prngFloat01(s);
                constexpr float kTwoPi = 6.283185307179586f;
                return __builtin_cosf(kTwoPi * u);
            }
            case DeviceInitMode::Bounded:
            {
                uint32_t s = seedMix(seed, dataIdx);
                float    u = prngFloat01(s);
                return cfg.minVal + (cfg.maxVal - cfg.minVal) * u;
            }
            case DeviceInitMode::BoundedAlternatingSign:
            {
                uint32_t s = seedMix(seed, dataIdx);
                float    u = prngFloat01(s);
                float    v = cfg.minVal + (cfg.maxVal - cfg.minVal) * u;
                return ((dataIdx & 1ull) ? -__builtin_fabsf(v) : __builtin_fabsf(v));
            }
            case DeviceInitMode::Unbounded:
            {
                uint32_t s   = seedMix(seed, dataIdx);
                float    u   = prngFloat01(s);
                // Map to a wide log-spaced range, sign random.
                float    sign = (xorshift32(s) & 1u) ? -1.0f : 1.0f;
                float    mag  = __builtin_ldexpf(u + 1.0f, static_cast<int>((xorshift32(s) % 16) - 8));
                return sign * mag;
            }
            case DeviceInitMode::NormalFromFloat:
            {
                // Box-Muller transform from two uniforms.
                uint32_t s  = seedMix(seed, dataIdx);
                float    u1 = __builtin_fmaxf(prngFloat01(s), 1.0e-7f);
                float    u2 = prngFloat01(s);
                float    r  = __builtin_sqrtf(-2.0f * __builtin_logf(u1));
                float    z0 = r * __builtin_cosf(6.2831853f * u2);
                return cfg.mean + cfg.stdDev * z0;
            }
            }
            return 0.0f;
        }

        // ----------------------------------------------------------------------
        // Device-side scale encoding
        //
        // For E8M0: stored as a single byte = unbiased_exp + bias (127).
        // For E4M3 / E5M3: stored as a byte (S?EEEE.MMM) with the unbiased
        // exponent we want and a normalised (1.0) mantissa.
        // ----------------------------------------------------------------------
        // Per-scale-format clamp: returns the unbiased exponent we will
        // actually encode in the scale byte. Mirrors the saturation rules
        // baked into encodeScale<ST> below; exposed separately so the data
        // convert path passes the same clamped value to the hardware MX
        // convert (keeping the kernel-visible scale and the scale used for
        // data quantisation in lockstep).
        template <ScaleType ST>
        __device__ __forceinline__ int clampScaleExp(int unbiasedExp);

        template <>
        __device__ __forceinline__ int clampScaleExp<ScaleType::E8M0>(int unbiasedExp)
        {
            // Byte range [0, 254] -> unbiased [-127, 127]. 255 is reserved
            // for NaN; we never produce it from numeric data.
            if(unbiasedExp < -127)
                return -127;
            if(unbiasedExp > 127)
                return 127;
            return unbiasedExp;
        }

        template <>
        __device__ __forceinline__ int clampScaleExp<ScaleType::E4M3>(int unbiasedExp)
        {
            if(unbiasedExp < -6)
                return -6;
            if(unbiasedExp > 7)
                return 7;
            return unbiasedExp;
        }

        template <>
        __device__ __forceinline__ int clampScaleExp<ScaleType::E5M3>(int unbiasedExp)
        {
            if(unbiasedExp < -14)
                return -14;
            if(unbiasedExp > 15)
                return 15;
            return unbiasedExp;
        }

        template <ScaleType ST>
        __device__ __forceinline__ uint8_t encodeScale(int unbiasedExp);

        template <>
        __device__ __forceinline__ uint8_t encodeScale<ScaleType::E8M0>(int unbiasedExp)
        {
            int v = unbiasedExp + 127;
            if(v < 0)
                v = 0;
            if(v > 254)
                v = 254;
            return static_cast<uint8_t>(v);
        }

        template <>
        __device__ __forceinline__ uint8_t encodeScale<ScaleType::E4M3>(int unbiasedExp)
        {
            // Byte layout S=1 EEEE MMM. Mantissa = 0 (normalised 1.0), so the
            // encoded scale value is exactly 1.0 * 2^unbiasedExp.
            constexpr int bias   = 7;
            constexpr int maxExp = 7;
            constexpr int minExp = -6;
            int           biased = unbiasedExp + bias;
            if(unbiasedExp > maxExp)
                biased = maxExp + bias;
            if(unbiasedExp < minExp)
                biased = 1;
            return static_cast<uint8_t>(biased << 3);
        }

        template <>
        __device__ __forceinline__ uint8_t encodeScale<ScaleType::E5M3>(int unbiasedExp)
        {
            // Byte layout S? EEEEE MMM. Mantissa = 0; encoded scale = 2^unbiasedExp.
            constexpr int bias   = 15;
            constexpr int maxExp = 15;
            constexpr int minExp = -14;
            int           biased = unbiasedExp + bias;
            if(unbiasedExp > maxExp)
                biased = maxExp + bias;
            if(unbiasedExp < minExp)
                biased = 1;
            return static_cast<uint8_t>(biased << 3);
        }

        template <ScaleType ST>
        __device__ __forceinline__ float decodeScale(uint8_t scale);

        template <>
        __device__ __forceinline__ float decodeScale<ScaleType::E8M0>(uint8_t scale)
        {
            int unbiased = static_cast<int>(scale) - 127;
            return __builtin_ldexpf(1.0f, unbiased);
        }

        template <>
        __device__ __forceinline__ float decodeScale<ScaleType::E4M3>(uint8_t scale)
        {
            int biasedExp = (scale >> 3) & 0x0f;
            int mant      = scale & 0x07;
            float m       = 1.0f + static_cast<float>(mant) / 8.0f;
            int unbiased  = biasedExp - 7;
            if(biasedExp == 0)
            {
                // Subnormal: implicit 0.M, exponent = 1 - bias.
                m        = static_cast<float>(mant) / 8.0f;
                unbiased = 1 - 7;
            }
            return m * __builtin_ldexpf(1.0f, unbiased);
        }

        template <>
        __device__ __forceinline__ float decodeScale<ScaleType::E5M3>(uint8_t scale)
        {
            int   biasedExp = (scale >> 3) & 0x1f;
            int   mant      = scale & 0x07;
            float m         = 1.0f + static_cast<float>(mant) / 8.0f;
            int   unbiased  = biasedExp - 15;
            if(biasedExp == 0)
            {
                m        = static_cast<float>(mant) / 8.0f;
                unbiased = 1 - 15;
            }
            return m * __builtin_ldexpf(1.0f, unbiased);
        }

        // ----------------------------------------------------------------------
        // Device-side data quantisation + packing.
        //
        // Each per-DTYPE specialisation takes one MX scale block worth of
        // float values, applies the supplied unbiased scale exponent, RNE
        // quantises to the target MX format, and writes the packed bytes
        // directly into `outBytes`.
        //
        // Implementation defers to `__amd_cvt_*_scale` from <hip/hip_ext_ocp.h>
        // - the same wrapper hipblaslt's hipblaslt_float4 / _float6 / _float8
        // types use. On gfx950 these emit the hardware MX convert
        // instructions (v_cvt_scalef32_pk_fp4_f32, pk32_f32_fp6, pk_fp8_f32);
        // off gfx950 they fall back to the canonical fcbx software path so
        // generator output is bit-identical to those production types.
        //
        // `blockSize` must equal `Tr::elementsPerScaleBlock`-style power-of-two
        // (16 or 32 in current use). FP6 always converts a full 32-element
        // group at once via __amd_cvt_floatx32_to_fp6x32_scale (the only
        // F32->FP6 wrapper); for blockSize<32 the trailing inputs are
        // zero-padded and the trailing output bytes discarded.
        // ----------------------------------------------------------------------
        template <typename DTYPE>
        __device__ __forceinline__ void convertBlockScaledRNE(uint8_t*      outBytes,
                                                              float const*  values,
                                                              int           blockSize,
                                                              __amd_scale_t scaleExp);

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m1_mxfp4>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            int const pairs = blockSize / 2;
            for(int i = 0; i < pairs; ++i)
            {
                __amd_floatx2_storage_t pair{values[2 * i], values[2 * i + 1]};
                outBytes[i] = static_cast<uint8_t>(
                    __amd_cvt_floatx2_to_fp4x2_scale(pair, __AMD_OCP_E2M1, scaleExp));
            }
        }

        // FP4 with non-E8M0 scale formats (E4M3, E5M3) shares the data path -
        // only the scale byte's encoding differs and that's handled by
        // encodeScale<scaleKind> at the call site.
        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m1_mxfp4_e4m3>(uint8_t*      outBytes,
                                                       float const*  values,
                                                       int           blockSize,
                                                       __amd_scale_t scaleExp)
        {
            convertBlockScaledRNE<ocp_e2m1_mxfp4>(outBytes, values, blockSize, scaleExp);
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m1_mxfp4_e5m3>(uint8_t*      outBytes,
                                                       float const*  values,
                                                       int           blockSize,
                                                       __amd_scale_t scaleExp)
        {
            convertBlockScaledRNE<ocp_e2m1_mxfp4>(outBytes, values, blockSize, scaleExp);
        }

        // FP6 / BF6: one __amd_cvt_floatx32_to_fp6x32_scale call covers a full
        // 32-element group. Pad inputs and trim outputs for blockSize < 32.
        template <__amd_fp6_interpretation_t Interp>
        __device__ __forceinline__ void convertBlockScaledRNEFp6(uint8_t*      outBytes,
                                                                 float const*  values,
                                                                 int           blockSize,
                                                                 __amd_scale_t scaleExp)
        {
            __amd_floatx32_storage_t in;
            for(int i = 0; i < 32; ++i)
                in[i] = (i < blockSize) ? values[i] : 0.0f;
            __amd_fp6x32_storage_t out
                = __amd_cvt_floatx32_to_fp6x32_scale(in, Interp, scaleExp);
            auto const* bytes  = reinterpret_cast<uint8_t const*>(&out);
            int const   outLen = blockSize * 6 / 8;
            for(int b = 0; b < outLen; ++b)
                outBytes[b] = bytes[b];
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e2m3_mxfp6>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp6<__AMD_OCP_E2M3>(outBytes, values, blockSize, scaleExp);
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e3m2_mxfp6>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp6<__AMD_OCP_E3M2>(outBytes, values, blockSize, scaleExp);
        }

        // FP8 / BF8: pk2 convert per pair of elements; each call returns a
        // 16-bit storage holding two FP8 bytes.
        template <__amd_fp8_interpretation_t Interp>
        __device__ __forceinline__ void convertBlockScaledRNEFp8(uint8_t*      outBytes,
                                                                 float const*  values,
                                                                 int           blockSize,
                                                                 __amd_scale_t scaleExp)
        {
            int const pairs = blockSize / 2;
            for(int i = 0; i < pairs; ++i)
            {
                __amd_floatx2_storage_t pair{values[2 * i], values[2 * i + 1]};
                __amd_fp8x2_storage_t   p
                    = __amd_cvt_floatx2_to_fp8x2_scale(pair, Interp, scaleExp);
                outBytes[2 * i + 0] = static_cast<uint8_t>(p & 0xffu);
                outBytes[2 * i + 1] = static_cast<uint8_t>((p >> 8) & 0xffu);
            }
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e4m3_mxfp8>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp8<__AMD_OCP_E4M3>(outBytes, values, blockSize, scaleExp);
        }

        template <>
        __device__ __forceinline__ void
            convertBlockScaledRNE<ocp_e5m2_mxfp8>(uint8_t*      outBytes,
                                                  float const*  values,
                                                  int           blockSize,
                                                  __amd_scale_t scaleExp)
        {
            convertBlockScaledRNEFp8<__AMD_OCP_E5M2>(outBytes, values, blockSize, scaleExp);
        }

        // ----------------------------------------------------------------------
        // Main generation kernel
        //
        // Launch with one thread per MX scale block (`numBlocks` threads
        // total). Each thread:
        //   1. Generates `blockSize` raw float values.
        //   2. Computes the per-block max abs.
        //   3. Picks an unbiased scale exponent so block_max / 2^scale fits
        //      in the dtype's max normal magnitude.
        //   4. Encodes the scale and writes it to `scaleOut`.
        //   5. Quantises each scaled element to the dtype bit pattern.
        //   6. Packs the result into `dataOut`.
        // ----------------------------------------------------------------------
        template <typename DTYPE>
        __global__ void generateMXBlocksKernel(uint8_t*         dataOut,
                                               uint8_t*         scaleOut,
                                               uint64_t         numBlocks,
                                               int              blockSize,
                                               uint32_t         seed,
                                               DeviceInitConfig cfg)
        {
            using Tr = GpuTraits<DTYPE>;

            uint64_t blockIdx_ = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if(blockIdx_ >= numBlocks)
                return;

            // Up to 32 elements per block; matches the only sizes used by
            // current callers (16 or 32). Stack allocation is fine.
            float values[32];
            float maxAbs = 0.0f;

            for(int i = 0; i < blockSize; ++i)
            {
                uint64_t dataIdx = blockIdx_ * static_cast<uint64_t>(blockSize) + i;
                float    v       = genValue(cfg.mode, seed, dataIdx, cfg);
                values[i]        = v;
                float a          = __builtin_fabsf(v);
                if(a > maxAbs)
                    maxAbs = a;
            }

            // Derive scale: choose unbiasedExp such that block_max <= maxNormal * 2^scaleExp.
            // i.e. scaleExp = ceil(log2(block_max / maxNormal)). Round up so
            // that division by 2^scaleExp brings every element within range.
            // For an all-zero block we pick a neutral scale of 2^0 = 1.
            int scaleExp = 0;
            if(maxAbs > 0.0f)
            {
                float ratio = maxAbs / Tr::maxNormal;
                float l2    = __builtin_log2f(ratio);
                scaleExp    = static_cast<int>(__builtin_ceilf(l2));
                while(maxAbs > Tr::maxNormal * __builtin_ldexpf(1.0f, scaleExp))
                    ++scaleExp;
            }

            // Saturate to the encodable range for this scale format BEFORE
            // touching either the scale byte or the data convert, so both
            // sides agree on the actual scale value the kernel will see.
            int const     clampedExp = clampScaleExp<Tr::scaleKind>(scaleExp);
            __amd_scale_t scaleArg   = static_cast<__amd_scale_t>(clampedExp);

            scaleOut[blockIdx_] = encodeScale<Tr::scaleKind>(clampedExp);

            uint64_t bytesPerBlock
                = static_cast<uint64_t>(blockSize) * Tr::bitsPerElem / 8u;
            uint8_t* outRow = dataOut + blockIdx_ * bytesPerBlock;
            convertBlockScaledRNE<DTYPE>(outRow, values, blockSize, scaleArg);
        }

        // ----------------------------------------------------------------------
        // Pre-swizzle scale kernel: applies the same algorithm as the host
        // `preSwizzleScalesGFX950` to a device-resident scale buffer.
        //
        // For an `(numRows, numCols)` row-major scale tensor (rounded up to
        // (paddedRows=mult32, paddedCols=mult8)), the algorithm views
        // the buffer as 6D (paddedRows/32, 2, 16, paddedCols/8, 2, 4) and
        // permutes (0,3,5,2,4,1). Each thread writes one output byte by
        // computing its source coordinates from the inverse permutation.
        // ----------------------------------------------------------------------
        __global__ void preSwizzleScalesKernel(uint8_t*       dst,
                                               uint8_t const* src,
                                               size_t         paddedRows,
                                               size_t         paddedCols)
        {
            size_t total = paddedRows * paddedCols;
            size_t tid   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if(tid >= total)
                return;

            // Decompose tid into 6D output coords matching the host-side permutation.
            // dimOrder = {1, 4, 2, 5, 3, 0} (row-major fastest-varying first).
            // Sizes (in dimOrder order): {2, 2, 16, 4, paddedCols/8, paddedRows/32}.
            size_t s1 = 2;
            size_t s4 = 2;
            size_t s2 = 16;
            size_t s5 = 4;
            size_t s3 = paddedCols / 8;
            size_t s0 = paddedRows / 32;

            size_t cur = tid;
            size_t i1  = cur % s1; cur /= s1;
            size_t i4  = cur % s4; cur /= s4;
            size_t i2  = cur % s2; cur /= s2;
            size_t i5  = cur % s5; cur /= s5;
            size_t i3  = cur % s3; cur /= s3;
            size_t i0  = cur % s0;

            // Source row/col in the original 2D layout:
            //   row = i0*32 + i1*16 + i2,  col = i3*8 + i4*4 + i5
            size_t srcRow = i0 * 32 + i1 * 16 + i2;
            size_t srcCol = i3 * 8 + i4 * 4 + i5;
            size_t srcIdx = srcRow * paddedCols + srcCol;

            dst[tid] = src[srcIdx];
        }

        // ----------------------------------------------------------------------
        // Generic host-side build helpers
        // ----------------------------------------------------------------------
        template <typename DTYPE>
        size_t computeArraySize(std::vector<index_t> const& sizes,
                                std::vector<index_t> const& strides)
        {
            // Mirror DataGenerator::generate logic: array_size =
            // strides[N-1] * sizes[N-1] (after sorting by stride).
            std::vector<size_t> sortedStrides(strides.begin(), strides.end());
            std::vector<size_t> sortedSizes(sizes.begin(), sizes.end());
            // Sort by stride ascending.
            std::vector<size_t> perm(sizes.size());
            for(size_t i = 0; i < perm.size(); ++i)
                perm[i] = i;
            std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
                return strides[a] < strides[b];
            });
            std::vector<size_t> ss(sizes.size()), st(strides.size());
            for(size_t i = 0; i < perm.size(); ++i)
            {
                ss[i] = static_cast<size_t>(sizes[perm[i]]);
                st[i] = static_cast<size_t>(strides[perm[i]]);
            }
            size_t n = ss.size();
            return st[n - 1] * ss[n - 1];
        }
    } // namespace gpu_detail

    // --------------------------------------------------------------------------
    // DataGeneratorGPU public-method definitions
    // --------------------------------------------------------------------------
    template <typename DTYPE>
    DataGeneratorGPU<DTYPE>::~DataGeneratorGPU()
    {
        if(m_ownsBuffers)
        {
            if(m_dataDevice)
                (void)hipFree(m_dataDevice);
            if(m_scaleDevice)
                (void)hipFree(m_scaleDevice);
        }
    }

    template <typename DTYPE>
    size_t DataGeneratorGPU<DTYPE>::getDataBufferBytes(std::vector<index_t> const& sizes,
                                                       DataGeneratorOptions const& options)
    {
        // We assume contiguous column-major (strides = {1, sizes[0], ...}).
        std::vector<index_t> strides(sizes.size(), 1);
        for(size_t i = 1; i < sizes.size(); ++i)
            strides[i] = strides[i - 1] * sizes[i - 1];
        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(sizes, strides);
        (void)options;
        constexpr int bitsPerElem = gpu_detail::GpuTraits<DTYPE>::bitsPerElem;
        return (arraySize * bitsPerElem + 7) / 8;
    }

    template <typename DTYPE>
    size_t DataGeneratorGPU<DTYPE>::getScaleBufferBytes(std::vector<index_t> const& sizes,
                                                        DataGeneratorOptions const& options)
    {
        std::vector<index_t> strides(sizes.size(), 1);
        for(size_t i = 1; i < sizes.size(); ++i)
            strides[i] = strides[i - 1] * sizes[i - 1];
        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(sizes, strides);
        if(options.blockScaling <= 0)
            return 0;
        return arraySize / static_cast<size_t>(options.blockScaling);
    }

    template <typename DTYPE>
    DataGeneratorGPU<DTYPE>& DataGeneratorGPU<DTYPE>::generate(std::vector<index_t>        sizes,
                                                               std::vector<index_t>        strides,
                                                               DataGeneratorOptions const& options,
                                                               hipStream_t                 stream)
    {
        m_options = options;
        m_sizes   = sizes;

        m_dataBufferBytes  = getDataBufferBytes(sizes, options);
        m_scaleBufferBytes = getScaleBufferBytes(sizes, options);

        // Free any prior allocation (size could change between calls).
        if(m_ownsBuffers)
        {
            if(m_dataDevice)
                (void)hipFree(m_dataDevice);
            if(m_scaleDevice)
                (void)hipFree(m_scaleDevice);
            m_dataDevice  = nullptr;
            m_scaleDevice = nullptr;
        }

        DGEN_GPU_CHECK_HIP(hipMalloc(&m_dataDevice, m_dataBufferBytes));
        if(m_scaleBufferBytes > 0)
            DGEN_GPU_CHECK_HIP(hipMalloc(&m_scaleDevice, m_scaleBufferBytes));
        m_ownsBuffers = true;

        generateInto(m_dataDevice, m_scaleDevice, sizes, strides, options, stream);
        return *this;
    }

    template <typename DTYPE>
    void DataGeneratorGPU<DTYPE>::generateInto(void*                       devData,
                                               void*                       devScale,
                                               std::vector<index_t>        sizes,
                                               std::vector<index_t>        strides,
                                               DataGeneratorOptions const& options,
                                               hipStream_t                 stream)
    {
        if(sizes.size() != strides.size())
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: size and stride vectors must have the same size");
        if(sizes.empty())
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: size vector must not be empty");
        if(options.blockScaling <= 0)
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: blockScaling must be > 0 for MX types");

        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(sizes, strides);
        if(arraySize % static_cast<size_t>(options.blockScaling) != 0)
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: array size must be a multiple of blockScaling");

        size_t numBlocks = arraySize / static_cast<size_t>(options.blockScaling);

        // Constant-fill / pathological modes can't go through the on-device
        // PRNG-quantize-derive-scale pipeline (the auto-derived scale would
        // pick up the constant magnitude rather than 1.0, NaN bit patterns
        // would round-trip through float quantisation, etc.).  Generate on
        // the host CPU and hipMemcpy.
        if(gpu_detail::requiresHostFallback(options.initMode))
        {
            DataGenerator<DTYPE> hostGen;
            hostGen.setSeed(m_seed);
            hostGen.generate(sizes, strides, options);
            auto hostData  = hostGen.getDataBytes();
            auto hostScale = hostGen.getScaleBytes();
            DGEN_GPU_CHECK_HIP(hipMemcpyAsync(
                devData, hostData.data(), hostData.size(), hipMemcpyHostToDevice, stream));
            if(devScale != nullptr && !hostScale.empty())
            {
                DGEN_GPU_CHECK_HIP(hipMemcpyAsync(devScale,
                                                  hostScale.data(),
                                                  hostScale.size(),
                                                  hipMemcpyHostToDevice,
                                                  stream));
            }
            DGEN_GPU_CHECK_HIP(hipStreamSynchronize(stream));
            return;
        }

        // Configuration for the kernel.
        gpu_detail::DeviceInitConfig cfg = gpu_detail::makeConfig(options, sizes);

        // Launch: 256 threads per block; one thread = one MX block.
        constexpr int kThreadsPerBlock = 256;
        size_t        gridDimX
            = (numBlocks + kThreadsPerBlock - 1) / kThreadsPerBlock;

        if(devScale == nullptr && m_scaleBufferBytes == 0)
            throw std::invalid_argument(
                "DataGeneratorGPU::generateInto: this DTYPE requires a scale buffer");

        hipLaunchKernelGGL(gpu_detail::generateMXBlocksKernel<DTYPE>,
                           dim3(static_cast<unsigned>(gridDimX)),
                           dim3(kThreadsPerBlock),
                           0,
                           stream,
                           static_cast<uint8_t*>(devData),
                           static_cast<uint8_t*>(devScale),
                           static_cast<uint64_t>(numBlocks),
                           static_cast<int>(options.blockScaling),
                           m_seed,
                           cfg);
        DGEN_GPU_CHECK_HIP(hipGetLastError());
    }

    template <typename DTYPE>
    std::vector<uint8_t> DataGeneratorGPU<DTYPE>::getDataBytes() const
    {
        std::vector<uint8_t> host(m_dataBufferBytes);
        if(m_dataBufferBytes == 0 || m_dataDevice == nullptr)
            return host;
        DGEN_GPU_CHECK_HIP(
            hipMemcpy(host.data(), m_dataDevice, m_dataBufferBytes, hipMemcpyDeviceToHost));
        return host;
    }

    template <typename DTYPE>
    std::vector<uint8_t> DataGeneratorGPU<DTYPE>::getScaleBytes() const
    {
        std::vector<uint8_t> host(m_scaleBufferBytes);
        if(m_scaleBufferBytes == 0 || m_scaleDevice == nullptr)
            return host;
        DGEN_GPU_CHECK_HIP(
            hipMemcpy(host.data(), m_scaleDevice, m_scaleBufferBytes, hipMemcpyDeviceToHost));
        return host;
    }

    template <typename DTYPE>
    std::vector<float> DataGeneratorGPU<DTYPE>::getReferenceFloat() const
    {
        // Materialise the reference floats on the host using the existing
        // CPU `toFloat<DTYPE>` helper - that's the bit-exact converter the
        // CPU backend uses and shares with all validation paths.
        auto dataHost  = getDataBytes();
        auto scaleHost = getScaleBytes();

        // Recover unpacked array_size from the buffer dimensions.
        std::vector<index_t> strides(m_sizes.size(), 1);
        for(size_t i = 1; i < m_sizes.size(); ++i)
            strides[i] = strides[i - 1] * m_sizes[i - 1];
        size_t arraySize = gpu_detail::computeArraySize<DTYPE>(m_sizes, strides);

        std::vector<float> ret(arraySize);
        size_t blockSize = static_cast<size_t>(m_options.blockScaling);
        for(size_t i = 0; i < arraySize; ++i)
        {
            size_t scaleIdx = i / blockSize;
            ret[i]
                = toFloatPacked<DTYPE>(scaleHost.data(), dataHost.data(),
                                       static_cast<index_t>(scaleIdx), static_cast<index_t>(i));
        }
        return ret;
    }

    template <typename DTYPE>
    void DataGeneratorGPU<DTYPE>::preSwizzleScalesGFX950Device(
        std::vector<size_t> const& scaleSizes, hipStream_t stream)
    {
        if(scaleSizes.size() != 2)
            throw std::invalid_argument(
                "DataGeneratorGPU::preSwizzleScalesGFX950Device: scaleSizes must have 2 elements");
        if(m_scaleDevice == nullptr)
            throw std::runtime_error(
                "DataGeneratorGPU::preSwizzleScalesGFX950Device: no scale buffer allocated");

        size_t numRows    = scaleSizes[0];
        size_t numCols    = scaleSizes[1];
        size_t paddedRows = ((numRows + 31) / 32) * 32;
        size_t paddedCols = ((numCols + 7) / 8) * 8;
        size_t paddedSize = paddedRows * paddedCols;

        // Source buffer needs padded dimensions; if the existing buffer is
        // already padded, alias it; otherwise pad in a temporary.
        uint8_t* src = nullptr;
        bool     ownsSrc = false;
        if(paddedRows == numRows && paddedCols == numCols)
        {
            src = m_scaleDevice;
        }
        else
        {
            DGEN_GPU_CHECK_HIP(hipMalloc(&src, paddedSize));
            DGEN_GPU_CHECK_HIP(hipMemsetAsync(src, 0, paddedSize, stream));
            // Copy each row of the original into the padded buffer.
            for(size_t r = 0; r < numRows; ++r)
            {
                DGEN_GPU_CHECK_HIP(hipMemcpyAsync(src + r * paddedCols,
                                                  m_scaleDevice + r * numCols,
                                                  numCols,
                                                  hipMemcpyDeviceToDevice,
                                                  stream));
            }
            ownsSrc = true;
        }

        uint8_t* dst = nullptr;
        DGEN_GPU_CHECK_HIP(hipMalloc(&dst, paddedSize));

        constexpr int kThreadsPerBlock = 256;
        size_t        gridDimX = (paddedSize + kThreadsPerBlock - 1) / kThreadsPerBlock;

        hipLaunchKernelGGL(gpu_detail::preSwizzleScalesKernel,
                           dim3(static_cast<unsigned>(gridDimX)),
                           dim3(kThreadsPerBlock),
                           0,
                           stream,
                           dst,
                           src,
                           paddedRows,
                           paddedCols);
        DGEN_GPU_CHECK_HIP(hipGetLastError());

        // Replace the owned scale buffer with the swizzled (padded) one.
        if(m_ownsBuffers && m_scaleDevice)
            (void)hipFree(m_scaleDevice);
        m_scaleDevice      = dst;
        m_scaleBufferBytes = paddedSize;
        m_ownsBuffers      = true;

        if(ownsSrc)
            (void)hipFree(src);
    }

} // namespace DGen

#endif // HIP guard
