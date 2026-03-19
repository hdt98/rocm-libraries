#pragma once

#include "bunnies.hpp"

namespace bunnies {

struct arch_cdna4
{
    static constexpr int wave_size = 64;
    using buffer_t                 = __amdgpu_buffer_rsrc_t;

    template <uint32_t flags = 0>
    struct mma
    {
        __device__ static void wmma(floatx4& d, fp16x8& a, fp16x8& b, floatx4& c)
        {
            d = __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, 0);
        }
        __device__ static void wmma(floatx4& d, bf16x8& a, bf16x8& b, floatx4& c)
        {
            d = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0);
        }
    };

    template <typename T, int Rows, int Cols, use Use>
    struct layout_config;
    template <half_type T>
    struct layout_config<T, 16, 32, use::A>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] % 16, x[0] / 16 * 8 ^ x[1]};
        }
    };
    template <half_type T>
    struct layout_config<T, 32, 16, use::B>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 8 ^ x[1], x[0] % 16};
        }
    };
    template <>
    struct layout_config<float, 16, 16, use::Acc>
    {
        __device__ static constexpr auto map(std::array<int, 2> const& x) -> std::array<int, 2>
        {
            return {x[0] / 16 * 4 ^ x[1], x[0] % 16};
        }
    };

    template <typename T>
    __device__ static auto make_buffer(T* global_ptr, int global_size) -> buffer_t
    {
        constexpr std::int32_t data_format = 1 << 15;
        return __builtin_amdgcn_make_buffer_rsrc(const_cast<std::remove_const_t<T>*>(global_ptr),
                                                 0,
                                                 global_size * sizeof(T),
                                                 data_format);
    }

    template <int BytesPerLane>
    struct buffer_load_lds
    {
        __device__ static void load(buffer_t buffer, void* lds_ptr, int v_offset, int s_offset)
        {
            if constexpr(BytesPerLane == 16)
            {
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    buffer, lds_ptr, 16, v_offset, s_offset, 0, 0);
            }
            else if constexpr(BytesPerLane == 4)
            {
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    buffer, lds_ptr, 4, v_offset, s_offset, 0, 0);
            }
            else
            {
                static_assert(true, "BytesPerLane must be 4 or 16");
            }
        }
    };

    template <typename T>
    struct ds_load_b128
    {
        static constexpr int num_items = 128 / (sizeof(T) * 8);
        using vec_t                    = __attribute__((ext_vector_type(num_items))) T;
        __device__ static auto map(int lane, int item) -> std::array<int, 2>
        {
            return {lane, item};
        }
        __device__ static void load(T* lds_ptr, T* dest)
        {
            *reinterpret_cast<vec_t*>(dest) = *reinterpret_cast<vec_t*>(lds_ptr);
        }
    };

    template <typename T>
    struct ds_read_b64_tr_b16
    {
        static constexpr int num_items = 64 / (sizeof(T) * 8);
        using vec_t                    = __attribute__((ext_vector_type(num_items))) T;
        __device__ static auto map(int lane, int item) -> std::array<int, 2>
        {
            const auto item0 = item / num_items * num_items;
            item             = item % num_items;
            return {lane % 4 * 4 ^ lane / 16 * 16 ^ item, lane / 4 % 4 ^ item0};
        }
        __device__ static void load(T* lds_ptr, T* dest)
        {
            if constexpr(std::is_same_v<T, _Float16>)
            {
                *reinterpret_cast<vec_t*>(dest) =
                    __builtin_amdgcn_ds_read_tr16_b64_v4f16(reinterpret_cast<vec_t*>(lds_ptr));
            }
            else if constexpr(std::is_same_v<T, __bf16>)
            {
                *reinterpret_cast<vec_t*>(dest) =
                    __builtin_amdgcn_ds_read_tr16_b64_v4bf16(reinterpret_cast<vec_t*>(lds_ptr));
            }
            else if constexpr(std::is_same_v<T, short>)
            {
                *reinterpret_cast<vec_t*>(dest) =
                    __builtin_amdgcn_ds_read_tr16_b64_v4i16(reinterpret_cast<vec_t*>(lds_ptr));
            }
            else
            {
                static_assert(true, "Unsupported ds_read_b64_tr_b16");
            }
        }
    };
};

} // namespace bunnies
