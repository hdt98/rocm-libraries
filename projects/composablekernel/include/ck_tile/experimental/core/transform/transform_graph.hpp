// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief Value-based coordinate transform graph and its public traversal API.
///
/// `TransformGraph` is a structural NTTP describing a coordinate mapping as
/// a LIFO stack of `CoordinateTransform`s with slot-based routing. Used as
/// a non-type template parameter (`template<TransformGraph G>`), it
/// guarantees compile-time constant folding of every transform parameter,
/// magic-division constant, and routing index.
///
/// The graph delegates dispatch to `detail::TransformImpl<Type>` via static
/// polymorphism. It carries no transform-specific knowledge itself.
///
/// ## Construction–traversal model
///
/// The transforms array is a LIFO stack:
///   - **Construction** pushes transforms bottom-up (base first, user-facing last).
///   - **Traversal** pops transforms top-down (user-facing first, base last).
///
/// There is only one traversal direction: user coordinates in, memory offset
/// out. Reverse traversal (`reverseMap`) walks the same wiring backwards.
///
/// ## Worked example: GEMM LDS descriptor (M=128, K=64)
///
/// Goal: map user coordinates (M=5, K=19) to a memory offset.
///
/// The physical LDS tensor is 3D: (K/8, M, K%8) with strides
/// ((M+1)*8, 8, 1). The user sees 2D: (M, K), where K merges K/8 and K%8.
///
/// ```cpp
/// constexpr auto desc = make_tensor_descriptor(
///     dims(K_DIV8, M, K_MOD8), dims((M+1)*8, 8, 1));
/// constexpr auto g = make_transform_graph(
///     outputs(0),
///     transform(desc,                    read(1, 2, 3), write(0)),
///     transform(make_pass_through(M),    read(4),       write(2)),
///     transform(make_merge(K_DIV8, K_MOD8), read(5),    write(1, 3)),
///     inputs(4, 5));
/// constexpr index_t off = calculateOffset<g>(static_array<index_t, 2>{5, 19});
/// // off == 5*8 + (19/8)*1032 + (19%8)*1
/// ```

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/experimental/core/transform/coordinate_transform.hpp"
#include "ck_tile/experimental/core/transform/transform_impl.hpp"

#include <type_traits>

namespace ck_tile::core::transform {

inline constexpr index_t MAX_TRANSFORMS = 20; ///< Max transforms per graph
inline constexpr index_t MAX_SLOTS =
    70; ///< Max working array slots (1 + MAX_TENSOR_DIMS for base Embed)

namespace detail {

/** @brief A directed acyclic graph of coordinate transforms.
 *
 *  General coordinate mapping: N input dimensions → M output dimensions.
 *  Used as an NTTP: `template<TransformGraph G>` to guarantee compile-time
 *  constant folding of all transform parameters.
 *
 *  Common case: tensor layout descriptor where `ndim_output == 1` (memory
 *  offset). The graph is general and can map between any two coordinate
 *  spaces.
 *
 *  Internally routes indices through a working array of slots. Users never
 *  see slot assignments — they are computed by factory functions.
 *
 *  @invariant Unused trailing slots in `transforms`, `t_input_slots`,
 *             `t_output_slots`, `input_slots`, and `output_slots` are
 *             zero-initialized by `detail::canonicalize` so that two graphs
 *             with identical meaningful content produce identical NTTP
 *             values.
 *  @note Internal routing fields (`t_input_slots`, `t_output_slots`,
 *        `input_slots`, `output_slots`, `num_slots`) must be public for
 *        structural NTTP eligibility. They are assigned by factory
 *        functions; users should not set them directly.
 */
struct TransformGraph
{
    // Transform stack (LIFO). Pushed during construction (base first,
    // user-facing last); popped during traversal (user-facing first,
    // base last). Index 0 = base (e.g., Embed); index num_transforms - 1 =
    // top (e.g., Merge).
    static_array<CoordinateTransform, MAX_TRANSFORMS> transforms{};
    index_t num_transforms = 0;

    // Per-transform slot routing — assigned by factory functions.
    static_array<DimIds, MAX_TRANSFORMS> t_input_slots{};
    static_array<DimIds, MAX_TRANSFORMS> t_output_slots{};
    index_t num_slots = 0;

    // Graph interface.
    index_t ndim_input  = 0; ///< Number of input coordinate dimensions
    index_t ndim_output = 0; ///< Number of output coordinate dimensions
    static_array<index_t, MAX_TENSOR_DIMS> input_slots{};  ///< Slots receiving input coords
    static_array<index_t, MAX_TENSOR_DIMS> output_slots{}; ///< Slots holding output coords

    constexpr bool operator==(const TransformGraph&) const = default;
};

// NTTP-eligibility canaries — fire at compile time on any regression.
static_assert(std::is_aggregate_v<TransformGraph>);
static_assert(std::is_trivially_copyable_v<TransformGraph>);
namespace {
constexpr TransformGraph _transform_graph_nttp_canary{};
static_assert(_transform_graph_nttp_canary == _transform_graph_nttp_canary);
} // namespace

static_assert(sizeof(TransformGraph) < 65536,
              "TransformGraph NTTP size canary — consider capacity templating if this fires");

// Forward fold-expression helpers (definitions in transform_graph_impl.hpp).
template <TransformGraph G, index_t I>
CK_TILE_HOST_DEVICE constexpr void applySingleTransform(static_array<index_t, G.num_slots>& slots);

template <TransformGraph G, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void applyAllTransforms(static_array<index_t, G.num_slots>& slots,
                                                      sequence<Is...>);

// inputDimLength dispatch helpers.
template <TransformGraph G, index_t T>
CK_TILE_HOST_DEVICE constexpr index_t tryInputLengthAt(index_t slot);

template <TransformGraph G, index_t... Ts>
CK_TILE_HOST_DEVICE constexpr index_t inputDimLengthDispatch(index_t slot, sequence<Ts...>);

// Reverse fold-expression helpers.
template <TransformGraph G, index_t I>
CK_TILE_HOST_DEVICE constexpr void
reverseApplySingleTransform(static_array<index_t, G.num_slots>& slots);

template <TransformGraph G, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void
reverseApplyAllTransforms(static_array<index_t, G.num_slots>& slots, sequence<Is...>);

} // namespace detail

// ============================================================================
// Public traversal API — templated on a TransformGraph NTTP.
// ============================================================================

/** @brief Map an input coordinate to an output coordinate.
 *
 *  Pops transforms from the top of the stack (user-facing) to the base
 *  (memory-facing) using a fold expression for guaranteed compile-time
 *  dispatch.
 *
 *  @tparam G          The transform graph (NTTP — all values are compile-time constants).
 *  @param[out] output Output coordinate buffer (size >= G.ndim_output).
 *  @param[in]  input  Input coordinate buffer  (size >= G.ndim_input).
 *  @complexity O(N) over N transforms; fold expression unrolled at compile time.
 */
template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr void map(index_t* output, const index_t* input);

/** @brief Compute a memory offset for graphs with `ndim_output == 1`.
 *
 *  Type-safe overload that validates dimensions at compile time.
 *
 *  @tparam G     The transform graph (NTTP).
 *  @tparam N     Number of input dimensions (deduced from array size).
 *  @param input  Input coordinate.
 *  @return Memory offset corresponding to `input`.
 *  @pre   `N == G.ndim_input` and `G.ndim_output == 1` (compile-time enforced).
 */
template <detail::TransformGraph G, index_t N>
CK_TILE_HOST_DEVICE constexpr index_t calculateOffset(const static_array<index_t, N>& input);

/** @brief Query the length of the i-th input dimension.
 *
 *  Uses a compile-time fold expression to dispatch to the correct
 *  TransformImpl specialization (runtime loops cannot be used because
 *  TransformImpl<Type> requires a constant expression).
 *
 *  @tparam G  The transform graph (NTTP).
 *  @param i   Input dimension index (0-based).
 *  @return Length of the i-th input dimension.
 */
template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr index_t inputDimLength(index_t i);

/** @brief Compile-time predicate: are all transforms in `G` bijective?
 *
 *  Useful as a guard before calling `reverseMap` / `reverseCalculateOffset`.
 *
 *  @tparam G  The transform graph (NTTP).
 *  @return `true` iff every transform in the graph is bijective.
 */
template <detail::TransformGraph G>
constexpr bool isGraphBijective();

/** @brief Reverse map: output coordinate → input coordinate.
 *
 *  Traverses the graph from base to top, applying `reverseMapIndices` on
 *  each transform. The inverse of `map<G>()`. All transforms must be
 *  bijective; this is checked at compile time.
 *
 *  @tparam G          The transform graph (NTTP).
 *  @param[out] input  Recovered input coordinate (size >= G.ndim_input).
 *  @param[in]  output Output coordinate to invert (size >= G.ndim_output).
 *  @pre All transforms in `G` are bijective.
 */
template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr void reverseMap(index_t* input, const index_t* output);

/** @brief Recover user coordinates from a memory offset.
 *
 *  @tparam G      The transform graph (NTTP).
 *  @param offset  Memory offset to invert.
 *  @return Input coordinate that maps to this offset.
 *  @pre   `G.ndim_output == 1` (compile-time enforced).
 */
template <detail::TransformGraph G>
CK_TILE_HOST_DEVICE constexpr static_array<index_t, G.ndim_input>
reverseCalculateOffset(index_t offset);

} // namespace ck_tile::core::transform

#include "ck_tile/experimental/core/transform/transform_graph_impl.hpp"
