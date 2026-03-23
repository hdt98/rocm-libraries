// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Data type utilities for rocm_ck kpack examples.
// Provides DataType enum, bit-width queries, host-side type conversions,
// and verification tolerances.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rocm_ck {

/// Data type tag for compile-time kernel configuration.
/// Independent of CK Tile headers. No UNDEFINED — every config must specify
/// a valid type. Covers all types used by CK operations.
///
/// Naming: FP8 = e4m3, BF8 = e5m2 (CK convention). FNUZ = MI300 native
/// (gfx942 hardware), OCP = MI350 native (gfx950 hardware, software on MI300).
enum class DataType
{
    // Floating point — standard widths
    FP64,
    FP32,
    FP16,
    BF16,

    // FP8 variants
    FP8_FNUZ, // e4m3, gfx942 hardware
    BF8_FNUZ, // e5m2, gfx942 hardware
    FP8_OCP,  // e4m3, gfx950 hardware
    BF8_OCP,  // e5m2, gfx950 hardware

    // Integer types — signed and unsigned at each width
    I4,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64
};

/// Returns the bit-width of a DataType. Uses bits (not bytes) so future
/// sub-byte types (fp4, fp6, int4) are clean integers.
/// No default case — lets -Wswitch catch unhandled enum values.
constexpr int data_type_bits(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return 64;
    case DataType::FP32: return 32;
    case DataType::FP16: return 16;
    case DataType::BF16: return 16;
    case DataType::FP8_FNUZ: return 8;
    case DataType::BF8_FNUZ: return 8;
    case DataType::FP8_OCP: return 8;
    case DataType::BF8_OCP: return 8;
    case DataType::I4: return 4;
    case DataType::I8: return 8;
    case DataType::I16: return 16;
    case DataType::I32: return 32;
    case DataType::I64: return 64;
    case DataType::U8: return 8;
    case DataType::U16: return 16;
    case DataType::U32: return 32;
    case DataType::U64: return 64;
    }
    return 0;
}

/// Returns a short string name for the data type (e.g. "FP32").
constexpr const char* data_type_name(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return "FP64";
    case DataType::FP32: return "FP32";
    case DataType::FP16: return "FP16";
    case DataType::BF16: return "BF16";
    case DataType::FP8_FNUZ: return "FP8_FNUZ";
    case DataType::BF8_FNUZ: return "BF8_FNUZ";
    case DataType::FP8_OCP: return "FP8_OCP";
    case DataType::BF8_OCP: return "BF8_OCP";
    case DataType::I4: return "I4";
    case DataType::I8: return "I8";
    case DataType::I16: return "I16";
    case DataType::I32: return "I32";
    case DataType::I64: return "I64";
    case DataType::U8: return "U8";
    case DataType::U16: return "U16";
    case DataType::U32: return "U32";
    case DataType::U64: return "U64";
    }
    return "???";
}

// --- Host-side type conversion utilities ---
// These are compiled with GCC (not hipcc), so hip's half/bfloat16 operator
// overloads are not available. We use _Float16 (GCC built-in) for fp16 and
// raw bit manipulation for bf16.

/// Convert float to bf16 using round-to-nearest-even.
inline std::uint16_t float_to_bf16_bits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even: add rounding bias based on LSB + trailing bits
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<std::uint16_t>(u >> 16);
}

/// Convert bf16 bits back to float.
inline float bf16_bits_to_float(std::uint16_t bits)
{
    std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/// Convert a float to the device type and store into a byte buffer.
inline void float_to_typed(DataType dt, float value, void* dst)
{
    switch(dt)
    {
    case DataType::FP64: *static_cast<double*>(dst) = static_cast<double>(value); break;
    case DataType::FP32: *static_cast<float*>(dst) = value; break;
    case DataType::FP16: *static_cast<_Float16*>(dst) = static_cast<_Float16>(value); break;
    case DataType::BF16: *static_cast<std::uint16_t*>(dst) = float_to_bf16_bits(value); break;
    case DataType::FP8_FNUZ:
    case DataType::BF8_FNUZ:
    case DataType::FP8_OCP:
    case DataType::BF8_OCP:
        std::fprintf(stderr, "%s host conversion not implemented\n", data_type_name(dt));
        std::abort();
    case DataType::I4:
    case DataType::I8:
    case DataType::I16:
    case DataType::I32:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64:
        std::fprintf(stderr, "%s: use integer-specific conversion\n", data_type_name(dt));
        std::abort();
    }
}

/// Read a typed value from a byte buffer and convert to float.
inline float typed_to_float(DataType dt, const void* src)
{
    switch(dt)
    {
    case DataType::FP64: return static_cast<float>(*static_cast<const double*>(src));
    case DataType::FP32: return *static_cast<const float*>(src);
    case DataType::FP16: return static_cast<float>(*static_cast<const _Float16*>(src));
    case DataType::BF16: return bf16_bits_to_float(*static_cast<const std::uint16_t*>(src));
    case DataType::FP8_FNUZ:
    case DataType::BF8_FNUZ:
    case DataType::FP8_OCP:
    case DataType::BF8_OCP:
        std::fprintf(stderr, "%s host conversion not implemented\n", data_type_name(dt));
        std::abort();
    case DataType::I4:
    case DataType::I8:
    case DataType::I16:
    case DataType::I32:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64:
        std::fprintf(stderr, "%s: use integer-specific conversion\n", data_type_name(dt));
        std::abort();
    }
    return 0.0f;
}

/// Tolerance for verification based on data type.
inline float tolerance_for(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return 1e-12f;
    case DataType::FP32: return 1e-5f;
    case DataType::FP16: return 1e-2f;
    case DataType::BF16: return 1e-1f;
    case DataType::FP8_FNUZ:
    case DataType::BF8_FNUZ:
    case DataType::FP8_OCP:
    case DataType::BF8_OCP: return 1.0f;
    case DataType::I4:
    case DataType::I8:
    case DataType::I16:
    case DataType::I32:
    case DataType::I64:
    case DataType::U8:
    case DataType::U16:
    case DataType::U32:
    case DataType::U64: return 0.0f; // exact for integer types
    }
    return 1e-5f;
}

} // namespace rocm_ck
