// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Data type utilities for rocm_ck kpack examples.
// Provides DataType enum, bit-width queries, host-side type conversions,
// and verification tolerances.

#pragma once

#include <cstdint>
#include <cstring>

namespace rocm_ck {

/// Data type tag for compile-time kernel configuration.
/// Modeled on ck_tile::builder::DataType but independent of CK Tile headers.
/// No UNDEFINED — every config must specify a valid type.
enum class DataType
{
    FP32,
    FP16,
    BF16,
    FP8
};

/// Returns the bit-width of a DataType. Uses bits (not bytes) so future
/// sub-byte types (fp4, fp6, int4) are clean integers.
/// No default case — lets -Wswitch catch unhandled enum values.
constexpr int data_type_bits(DataType dt)
{
    switch(dt)
    {
    case DataType::FP32: return 32;
    case DataType::FP16: return 16;
    case DataType::BF16: return 16;
    case DataType::FP8: return 8;
    }
    return 0;
}

/// Returns a short string name for the data type (e.g. "FP32").
constexpr const char* data_type_name(DataType dt)
{
    switch(dt)
    {
    case DataType::FP32: return "FP32";
    case DataType::FP16: return "FP16";
    case DataType::BF16: return "BF16";
    case DataType::FP8: return "FP8";
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
    case DataType::FP32: *static_cast<float*>(dst) = value; break;
    case DataType::FP16: *static_cast<_Float16*>(dst) = static_cast<_Float16>(value); break;
    case DataType::BF16: *static_cast<std::uint16_t*>(dst) = float_to_bf16_bits(value); break;
    case DataType::FP8: break; // not used
    }
}

/// Read a typed value from a byte buffer and convert to float.
inline float typed_to_float(DataType dt, const void* src)
{
    switch(dt)
    {
    case DataType::FP32: return *static_cast<const float*>(src);
    case DataType::FP16: return static_cast<float>(*static_cast<const _Float16*>(src));
    case DataType::BF16: return bf16_bits_to_float(*static_cast<const std::uint16_t*>(src));
    case DataType::FP8: return 0.0f;
    }
    return 0.0f;
}

/// Tolerance for verification based on data type.
inline float tolerance_for(DataType dt)
{
    switch(dt)
    {
    case DataType::FP32: return 1e-5f;
    case DataType::FP16: return 1e-2f;
    case DataType::BF16: return 1e-1f;
    case DataType::FP8: return 1.0f;
    }
    return 1e-5f;
}

} // namespace rocm_ck
