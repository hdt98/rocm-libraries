#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <array>
#include <concepts>
#include <type_traits>

namespace bunnies
{

using uint32x4 = __attribute__((ext_vector_type(4))) uint32_t;
using uint32x8 = __attribute__((ext_vector_type(8))) uint32_t;
using floatx4  = __attribute__((ext_vector_type(4))) float;
using floatx8  = __attribute__((ext_vector_type(8))) float;
using fp16x8   = __attribute__((ext_vector_type(8))) _Float16;
using bf16x8   = __attribute__((ext_vector_type(8))) __bf16;
using fp16x16  = __attribute__((ext_vector_type(16))) _Float16;
using bf16x16  = __attribute__((ext_vector_type(16))) __bf16;

enum class use
{
    A,
    B,
    Acc
};

enum class wmma_flag : uint32_t
{
    A_reuse = 1,
    B_reuse = 2
};

constexpr auto test(uint32_t flags, wmma_flag flag) -> bool
{
    return flags & static_cast<int>(flag);
}

template <typename T>
concept half_type = std::same_as<T, _Float16> || std::same_as<T, __bf16>;

constexpr auto ilog2(int x) -> int
{
    int l = 0;
    while(x > 1)
    {
        ++l;
        x >>= 1;
    }
    return l;
}

///////////////////////////
////// Architecture ///////
///////////////////////////

template <typename T>
concept device_arch = requires(T t)
{
    {T::wave_size};
};

template <typename T, int Rows, int Cols, use Use, typename LayoutConfig>
struct layout
{
    static constexpr int num_items = 128 / (8 * sizeof(T));
    using type                     = T;
    using vec_t                    = __attribute__((ext_vector_type(num_items))) T;
    static constexpr int rows      = Rows;
    static constexpr int cols      = Cols;
    static constexpr use use_      = Use;

    constexpr auto map(std::array<int, 2>& x) -> std::array<int, 2> { return LayoutConfig::map(x); }
};

///////////////////////////
////// Register tile //////
///////////////////////////

template <typename T, int Rows, int Cols, int RowBlocks, int ColBlocks, use Use, device_arch Arch>
struct reg_tile
{
    using arch                      = Arch;
    using type                      = T;
    using layout                    = arch::template layout_config<T, Rows, Cols, Use>;
    static constexpr int rows       = Rows;
    static constexpr int cols       = Cols;
    static constexpr int row_blocks = RowBlocks;
    static constexpr int col_blocks = ColBlocks;
    static constexpr use use_       = Use;
    static constexpr int num_items  = Rows * Cols / arch::wave_size;

    using vec_t = __attribute__((ext_vector_type(num_items))) T;
    std::array<std::array<vec_t, col_blocks>, row_blocks> data;
};

///////////////////////////
///////// Memref //////////
///////////////////////////

template <typename IdxT = int>
struct slice
{
    IdxT offset = 0, size = 0;
};

namespace detail
{
template <typename IdxT>
__device__ auto offset(IdxT i)
{
    return i;
}
template <typename IdxT>
__device__ auto offset(slice<IdxT> i)
{
    return i.offset;
}
template <typename IdxT>
__device__ auto size(IdxT i)
{
    return 0;
}
template <typename IdxT>
__device__ auto size(slice<IdxT> i)
{
    return i.size;
}
} // namespace detail

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
struct tensor_view
{
    static constexpr int dim = Dim;
    using tuple_t            = std::array<IdxT, Dim>;

    OffsetT offset;
    std::array<IdxT, Dim> shape, stride;

    template <typename... I>
    __device__ auto delta(I&&... idx) const -> IdxT
    {
        static_assert(sizeof...(I) == Dim);
        static_assert((std::is_same_v<std::decay_t<I>, IdxT> && ...));

        IdxT p                        = 0;
        std::array<IdxT, Dim> offsets = {idx...};
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            p += offsets[i] * stride[i];
        }
        return p;
    }
    template <typename... I>
    __device__ auto operator()(I&&... idx) const -> OffsetT
    {
        return offset + delta(std::forward<I>(idx)...);
    }

    template <typename... I>
    __device__ auto subview(I&&... idx_or_slice) const
    {
        static_assert(sizeof...(I) == Dim);
        static_assert(((std::is_same_v<std::decay_t<I>, IdxT> ||
                        std::is_same_v<std::decay_t<I>, slice<IdxT>>) &&
                       ...));

        constexpr int SubDim = (static_cast<int>(std::is_same_v<I, slice<IdxT>>) + ...);

        std::array<IdxT, Dim> offsets  = {detail::offset(idx_or_slice)...};
        std::array<bool, Dim> is_slice = {std::is_same_v<I, slice<IdxT>>...};

        OffsetT suboffset = offset;
        std::array<IdxT, SubDim> subshape, substride;
        int j = 0;
#pragma unroll
        for(int i = 0; i < Dim; ++i)
        {
            suboffset += offsets[i] * stride[i];
            if(is_slice[i])
            {
                subshape[j]  = shape[i];
                substride[j] = stride[i];
                ++j;
            }
        }
        return tensor_view<SubDim, IdxT, OffsetT>(suboffset, subshape, substride);
    }
};

template <int Dim, typename T, typename IdxT = int>
using memref = tensor_view<Dim, IdxT, T*>;

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto
make_view(OffsetT offset, std::array<IdxT, Dim> const& shape, std::array<IdxT, Dim> const& stride)
{
    return tensor_view<Dim, IdxT, OffsetT>{offset, shape, stride};
}

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto make_view_col_major(std::array<IdxT, Dim> const& shape)
{
    std::array<IdxT, Dim> stride;
    stride[0] = 1;
    for(int mode = 0; mode < Dim - 1; ++mode)
    {
        stride[mode + 1] = stride[mode] * shape[mode];
    }
    return tensor_view<Dim, IdxT, OffsetT>{0, shape, stride};
}

template <int Dim, typename IdxT = int, typename OffsetT = IdxT>
__device__ auto make_view_row_major(std::array<IdxT, Dim> const& shape)
{
    std::array<IdxT, Dim> stride;
    stride[Dim - 1] = 1;
    for(int mode = Dim - 1; mode > 0; --mode)
    {
        stride[mode - 1] = stride[mode] * shape[mode];
    }
    return tensor_view<Dim, IdxT, OffsetT>{0, shape, stride};
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto
make_memref(T* ptr, std::array<IdxT, Dim> const& shape, std::array<IdxT, Dim> const& stride)
{
    return make_view<Dim, IdxT, T*>(ptr, shape, stride);
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto make_memref_col_major(std::array<IdxT, Dim> const& shape)
{
    return make_view_col_major<Dim, IdxT, T*>(shape);
}

template <int Dim, typename T, typename IdxT = int>
__device__ auto make_memref_row_major(std::array<IdxT, Dim> const& shape)
{
    return make_view_row_major<Dim, IdxT, T*>(shape);
}

///////////////////////////
///////// Actions /////////
///////////////////////////
//
template <device_arch Arch>
__device__ __forceinline__ auto lane_id() -> int
{
    return threadIdx.x % Arch::wave_size;
}

template <device_arch Arch>
__device__ __forceinline__ auto wave_id() -> int
{
    return __builtin_amdgcn_readfirstlane(threadIdx.x / Arch::wave_size);
}

template <typename Arch,
          int Rows,
          int Cols,
          int BytesPerLane,
          int NumWaves,
          bool UseRoundLaneMap = false>
struct buffer_load_to_lds_config
{
    using arch                              = Arch;
    static constexpr int rows               = Rows;
    static constexpr int cols               = Cols;
    static constexpr int bytes_per_lane     = BytesPerLane;
    static constexpr int num_waves          = NumWaves;
    static constexpr int use_round_lane_map = UseRoundLaneMap;
};
template <typename Cfg, typename T, typename SwizzleInv>
__device__ __forceinline__ void buffer_load_to_lds(int wave,
                                                   typename Cfg::arch::buffer_t global_buffer,
                                                   tensor_view<2> const& global_view,
                                                   std::array<int, 2> global_offset,
                                                   SwizzleInv&& swizzle_inv,
                                                   T* lds_ptr,
                                                   int lds_offset)
{
    using arch      = Cfg::arch;
    using load_inst = typename arch::template buffer_load_lds<Cfg::bytes_per_lane>;
    const auto lane = lane_id<arch>();
    const int num_rounds =
        Cfg::rows * Cfg::cols * sizeof(T) / (arch::wave_size * Cfg::bytes_per_lane);
    const int num_rounds_per_wave = num_rounds / Cfg::num_waves;
    const int lane_stride         = Cfg::bytes_per_lane / sizeof(T);
    const int round_stride        = arch::wave_size * lane_stride;
    const auto s_offset           = global_view(global_offset[0], global_offset[1]) * sizeof(T);
#pragma unroll
    for(int i = 0; i < num_rounds_per_wave; ++i)
    {
        const int round = i + num_rounds_per_wave * wave;
        void* lds_dest  = lds_ptr + lds_offset + round * round_stride;
        if constexpr(Cfg::use_round_lane_map)
        {
            const auto [mm, kk] = swizzle_inv(round, lane);
            const auto v_offset = global_view.delta(mm, kk) * sizeof(T);
            load_inst::load(global_buffer, lds_dest, v_offset, s_offset);
        }
        else
        {
            const auto [mm, kk] = swizzle_inv(round * round_stride + lane * lane_stride);
            const auto v_offset = global_view.delta(mm, kk) * sizeof(T);
            load_inst::load(global_buffer, lds_dest, v_offset, s_offset);
        }
    }
}

enum class matrix_order
{
    row_major,
    col_major
};
template <typename Arch, int Rows, int Cols, int BytesPerLane, matrix_order Order, int NumWaves>
struct global_async_config
{
    using arch                          = Arch;
    static constexpr int rows           = Rows;
    static constexpr int cols           = Cols;
    static constexpr int bytes_per_lane = BytesPerLane;
    static constexpr matrix_order order = Order;
    static constexpr int num_waves      = NumWaves;
};
template <typename Cfg, typename T, typename Swizzle>
__device__ __forceinline__ void global_load_async_to_lds(int wave,
                                                         T* global_ptr,
                                                         tensor_view<2> const& global_view,
                                                         std::array<int, 2> global_offset,
                                                         Swizzle&& swizzle,
                                                         T* lds_ptr)
{
    using arch      = Cfg::arch;
    using load_inst = typename arch::template global_load_async_to_lds<Cfg::bytes_per_lane>;
    const auto lane = lane_id<arch>();
    const int num_rounds =
        Cfg::rows * Cfg::cols * sizeof(T) / (arch::wave_size * Cfg::bytes_per_lane);
    const int num_rounds_per_wave = num_rounds / Cfg::num_waves;
    const int lane_stride         = Cfg::bytes_per_lane / sizeof(T);
    const int round_stride        = arch::wave_size * lane_stride;
    const auto s_offset           = global_view(global_offset[0], global_offset[1]);
#pragma unroll
    for(int i = 0; i < num_rounds_per_wave; ++i)
    {
        const int round  = i + num_rounds_per_wave * wave;
        const int offset = round * round_stride + lane * lane_stride;
        const auto mm =
            Cfg::order == matrix_order::col_major ? offset % Cfg::rows : offset / Cfg::cols;
        const auto kk =
            Cfg::order == matrix_order::col_major ? offset / Cfg::rows : offset % Cfg::cols;
        const int v_offset = global_view.delta(mm, kk);
        load_inst::load(global_ptr + v_offset + s_offset, lds_ptr + swizzle(mm, kk));
    }
}

template <typename Cfg, typename T, typename Swizzle>
__device__ __forceinline__ void global_store_async_from_lds(int wave,
                                                            T* global_ptr,
                                                            tensor_view<2> const& global_view,
                                                            std::array<int, 2> global_offset,
                                                            Swizzle&& swizzle,
                                                            T* lds_ptr)
{
    using arch       = Cfg::arch;
    using store_inst = typename arch::template global_store_async_from_lds<Cfg::bytes_per_lane>;
    const auto lane  = lane_id<arch>();
    const int num_rounds =
        Cfg::rows * Cfg::cols * sizeof(T) / (arch::wave_size * Cfg::bytes_per_lane);
    const int num_rounds_per_wave = num_rounds / Cfg::num_waves;
    const int lane_stride         = Cfg::bytes_per_lane / sizeof(T);
    const int round_stride        = arch::wave_size * lane_stride;
    const auto s_offset           = global_view(global_offset[0], global_offset[1]);
#pragma unroll
    for(int i = 0; i < num_rounds_per_wave; ++i)
    {
        const int round  = i + num_rounds_per_wave * wave;
        const int offset = round * round_stride + lane * lane_stride;
        const auto mm =
            Cfg::order == matrix_order::col_major ? offset % Cfg::rows : offset / Cfg::cols;
        const auto kk =
            Cfg::order == matrix_order::col_major ? offset / Cfg::rows : offset % Cfg::cols;
        const int v_offset = global_view.delta(mm, kk);
        store_inst::store(global_ptr + v_offset + s_offset, lds_ptr + swizzle(mm, kk));
    }
}

template <template <typename> typename LoadInst, typename RegTile, typename Map>
__device__ void lds_load(RegTile& rt, typename RegTile::type* lds_base, Map&& map)
{
    using arch                        = typename RegTile::arch;
    using type                        = typename RegTile::type;
    constexpr int num_items           = RegTile::rows * RegTile::cols / arch::wave_size;
    constexpr int num_items_per_round = LoadInst<type>::num_items;
    constexpr int num_rounds          = num_items / num_items_per_round;
    const int lane                    = lane_id<arch>();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
#pragma unroll
            for(int rnd = 0; rnd < num_rounds; ++rnd)
            {
                const auto laneitem = LoadInst<type>::map(lane, rnd * num_items_per_round);
                const auto coord    = RegTile::layout::map(laneitem);
                const int offset    = map(mb, nb, coord[0], coord[1]);
                LoadInst<type>::load(lds_base + offset,
                                     reinterpret_cast<type*>(&rt.data[mb][nb]) +
                                         rnd * num_items_per_round);
            }
        }
    }
}

template <template <typename> typename StoreInst, typename RegTile, typename Map>
__device__ void lds_store(RegTile& rt, typename RegTile::type* lds_base, Map&& map)
{
    using arch                        = typename RegTile::arch;
    using type                        = typename RegTile::type;
    constexpr int num_items           = RegTile::rows * RegTile::cols / arch::wave_size;
    constexpr int num_items_per_round = StoreInst<type>::num_items;
    constexpr int num_rounds          = num_items / num_items_per_round;
    const int lane                    = lane_id<arch>();
#pragma unroll
    for(int mb = 0; mb < RegTile::row_blocks; ++mb)
    {
#pragma unroll
        for(int nb = 0; nb < RegTile::col_blocks; ++nb)
        {
#pragma unroll
            for(int rnd = 0; rnd < num_rounds; ++rnd)
            {
                const auto laneitem = StoreInst<type>::map(lane, rnd * num_items_per_round);
                const auto coord    = RegTile::layout::map(laneitem);
                const int offset    = map(mb, nb, coord[0], coord[1]);
                StoreInst<type>::store(lds_base + offset,
                                       reinterpret_cast<type*>(&rt.data[mb][nb]) +
                                           rnd * num_items_per_round);
            }
        }
    }
}

template <typename D, typename A, typename B, typename C>
__device__ void mma(D& d, A& a, B& b, C& c)
{
    static_assert(std::is_same_v<typename A::arch, typename B::arch>);
    static_assert(std::is_same_v<typename B::arch, typename C::arch>);
    static_assert(std::is_same_v<typename C::arch, typename D::arch>);
    static_assert(D::use_ == use::Acc);
    static_assert(A::use_ == use::A);
    static_assert(B::use_ == use::B);
    static_assert(C::use_ == use::Acc);
    static_assert(C::rows == D::rows && C::row_blocks == D::row_blocks);
    static_assert(C::cols == D::cols && C::col_blocks == D::col_blocks);
    static_assert(C::rows == A::rows && C::row_blocks == A::row_blocks);
    static_assert(C::cols == B::cols && C::col_blocks == B::col_blocks);
    static_assert(A::cols == B::rows && A::col_blocks == B::row_blocks);

    using arch = typename A::arch;

#pragma unroll
    for(int nb = 0; nb < C::col_blocks; ++nb)
    {
#pragma unroll
        for(int kb = 0; kb < A::col_blocks; ++kb)
        {
            arch::template mma<>::wmma(d.data[0][nb], a.data[0][kb], b.data[kb][nb], c.data[0][nb]);
#pragma unroll
            for(int mb = 1; mb < C::row_blocks; ++mb)
            {
                constexpr uint32_t flags = static_cast<uint32_t>(wmma_flag::A_reuse);
                arch::template mma<flags>::wmma(
                    d.data[mb][nb], a.data[mb][kb], b.data[kb][nb], c.data[mb][nb]);
            }
        }
    }
}

} // namespace bunnies
