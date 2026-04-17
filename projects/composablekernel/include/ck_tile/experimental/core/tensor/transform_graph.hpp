// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/** @file transform_graph.hpp
 *  @brief Value-based coordinate transform graph and NTTP-based free functions.
 *
 *  A TransformGraph maps user-facing coordinates to a memory offset through a
 *  stack of coordinate transforms. It is used as a C++20 structural NTTP
 *  (template<TransformGraph G>) to guarantee compile-time constant folding of
 *  all transform parameters, routing, and magic division constants.
 *
 *  The graph knows NOTHING about individual transform algorithms ---
 *  it delegates to TransformImpl<Type> via static dispatch.
 *
 *
 *  TRANSFORM STACK --- CONSTRUCTION AND TRAVERSAL
 *  ===============================================
 *
 *  The transforms array is a LIFO stack:
 *
 *    - CONSTRUCTION pushes transforms bottom-up (base first, user-facing last).
 *    - TRAVERSAL pops transforms top-down (user-facing first, base last).
 *
 *  There is only one traversal direction: user coordinates in, memory offset
 *  out. The graph is never traversed in the other direction.
 *
 *
 *  WORKED EXAMPLE: GEMM LDS Descriptor (M=128, K=64)
 *  ===================================================
 *
 *  Goal: Map user coordinates (M=5, K=19) to a memory offset.
 *
 *  The physical LDS tensor is 3D: (K/8, M, 8) with strides ((M+1)*8, 8, 1).
 *  The user sees 2D: (M, K), where K merges the K/8 and K_mod8 dimensions.
 *
 *
 *  STEP 1: CONSTRUCTION --- make_strided_graph
 *  --------------------------------------------
 *
 *    constexpr auto g0 = make_strided_graph(
 *        {K/8=8, M=128, K_mod8=8},        // lengths
 *        {1032,  8,     1});              // strides = ((M+1)*8, 8, 1)
 *
 *  This pushes one Embed transform and assigns 4 slots:
 *
 *    slots:  [0]       [1]       [2]       [3]
 *            offset    K/8       M         K_mod8
 *
 *    Stack:
 *      [0] Embed   reads {1,2,3}   writes {0}   strides=(1032,8,1)
 *
 *    Endpoints:
 *      upper_slots  = {1, 2, 3}    (dim 0=K/8, dim 1=M, dim 2=K_mod8)
 *      lower_slots = {0}          (memory offset)
 *      ndim_upper   = 3
 *
 *  At this point the user would need to provide 3 coordinates (K/8, M, K_mod8).
 *  We want to reduce this to 2 coordinates (M, K) by merging K/8 and K_mod8.
 *
 *
 *  STEP 2: CONSTRUCTION --- applyTransforms
 *  ------------------------------------------
 *
 *  Recall the initial graph g0 has these input dimensions:
 *
 *    g0 input dims:   index 0 = K/8      (slot[1])
 *                     index 1 = M        (slot[2])
 *                     index 2 = K_mod8   (slot[3])
 *                            ^
 *                            these are the indices that input_dims refers to
 *
 *    constexpr auto g = applyTransforms(
 *        g0,                                          // old graph
 *        static_array<CoordinateTransform, 2>{        // 2 new transforms
 *            make_pass_through(128),                  //   [0] PassThrough
 *            make_merge(static_array<index_t, 2>{8, 8})}, // [1] Merge
 *        static_array<DimIds, 2>{                     // input_dims:
 *            dims(1),                                 //   PT replaces old index 1
 *            dims(0, 2)},                             //   Merge replaces old 0 & 2
 *        static_array<DimIds, 2>{                     // output_dims:
 *            dims(0),                                 //   PT becomes new dim 0
 *            dims(1)});                               //   Merge becomes new dim 1
 *
 *  input_dims and output_dims are arrays of DIMENSION INDICES:
 *
 *    input_dims[i] = indices into the OLD graph's input dim array.
 *                    Tells which old dims this transform REPLACES.
 *                    The transform's outputs get wired to those old slots.
 *
 *      input_dims[0] = dims(1)   --> PassThrough replaces old index 1 (M).
 *                                    Old index 1 lives at slot[2].
 *                                    So PassThrough writes to slot[2].
 *
 *      input_dims[1] = dims(0,2) --> Merge replaces old indices 0 and 2
 *                                    (K/8 and K_mod8).
 *                                    Old index 0 = slot[1], index 2 = slot[3].
 *                                    So Merge writes to slot[1] and slot[3].
 *
 *    output_dims[i] = indices into the NEW graph's input dim array.
 *                     Tells which new user-facing dim this transform CREATES.
 *                     A fresh slot is allocated; the user's coordinate enters there.
 *
 *      output_dims[0] = dims(0) --> PassThrough creates new dim index 0.
 *                                   Allocate fresh slot[4].
 *                                   The user's 1st coordinate (M) enters slot[4].
 *                                   PassThrough reads from slot[4].
 *
 *      output_dims[1] = dims(1) --> Merge creates new dim index 1.
 *                                   Allocate fresh slot[5].
 *                                   The user's 2nd coordinate (K) enters slot[5].
 *                                   Merge reads from slot[5].
 *
 *  After processing PassThrough:
 *
 *    slots:  [0]       [1]       [2]       [3]       [4]
 *            offset    K/8       M         K_mod8    M_user
 *                                ^                   ^
 *                                |  PassThrough      |
 *                                +--- writes here    +--- reads here
 *
 *  After processing Merge:
 *
 *    slots:  [0]       [1]       [2]       [3]       [4]       [5]
 *            offset    K_div8    M         K_mod8    M_user    K_user
 *                      ^                   ^                   ^
 *                      |       Merge       |                   |
 *                      +--- writes here    +--- writes here    +--- reads here
 *
 *
 *  RESULTING WIRING DIAGRAM
 *  -------------------------
 *
 *  The complete slot wiring after construction. During traversal, data
 *  flows top-down through these connections:
 *
 *    User provides:   dim 0 = M                dim 1 = K
 *                         |                        |
 *                         v                        v
 *    upper_slots:      slot[4]                  slot[5]
 *                         |                        |
 *                   .-----'                  .-----'
 *                   |                        |
 *                   |  [1] PassThrough       |  [2] Merge
 *                   |   reads:  {4}          |   reads:  {5}
 *                   |   writes: {2}          |   writes: {1, 3}
 *                   |                        |
 *                   '-----.            .-----'------.
 *                         |            |            |
 *                         v            v            v
 *    base slots:       slot[2]      slot[1]      slot[3]
 *                        (M)        (K_div8)     (K_mod8)
 *                         |            |            |
 *                   .-----'------------'------------'
 *                   |
 *                   |  [0] Embed
 *                   |   reads:  {1, 2, 3}
 *                   |   writes: {0}
 *                   |
 *                   '-----.
 *                         |
 *                         v
 *    lower_slots:     slot[0]
 *                    memory offset
 *
 *
 *  STEP 3: TRAVERSAL --- calculateOffset<g>({M=5, K=19})
 *  --------------------------------------------------------
 *
 *  Data flows top-down through the wiring built above:
 *
 *    Pop [2] Merge:        read slot[5] = 19
 *                          19 / 8 = 2 remainder 3
 *                          write slot[1] = 2, slot[3] = 3
 *
 *    Pop [1] PassThrough:  read slot[4] = 5
 *                          write slot[2] = 5
 *
 *    Pop [0] Embed:        read slot[1]=2, slot[2]=5, slot[3]=3
 *                          2*1032 + 5*8 + 3*1 = 2107
 *                          write slot[0] = 2107
 *
 *  Slot state after each step:
 *
 *                        [0]     [1]     [2]     [3]     [4]     [5]
 *                       offset  K_div8    M     K_mod8  M_user  K_user
 *                       ------  ------  ------  ------  ------  ------
 *    After user input:    _       _       _       _       5      19
 *    After pop Merge:     _       2       _       3       5      19
 *    After pop PassThru:  _       2       5       3       5      19
 *    After pop Embed:    2107     2       5       3       5      19
 *                         ^
 *                         output = 2107
 *
 *
 *  ndim_upper AND ndim_lower
 *  ==========================
 *
 *  These fields describe the traversal direction (top-down) for each transform:
 *
 *    ndim_upper  = dimensions received from above (from the user, or from
 *                  the transform above in the stack)
 *    ndim_lower = dimensions sent below (toward memory, feeding the
 *                  transform below in the stack)
 *
 *   .-------------------.---------------------------------------------------.
 *   | Transform         | ndim_upper (from above)  | ndim_lower (to below) |
 *   |-------------------|--------------------------|------------------------|
 *   | Merge (N->1)      | 1 (one merged value)     | N (N components)       |
 *   | Unmerge (1->N)    | N (N component values)   | 1 (one flat value)     |
 *   | PassThrough       | 1                        | 1                      |
 *   | Embed             | N (N dim indices)        | 1 (memory offset)      |
 *   | Pad               | 1                        | 1                      |
 *   '-------------------'--------------------------'------------------------'
 *
 *  Note: Merge is defined as "combine N dims into 1", but during traversal
 *  it must UNDO itself --- decomposing 1 merged value back into N components.
 *  So ndim_upper=1 and ndim_lower=N, which is the inverse of the definition.
 *  This is true for every transform: traversal applies the inverse because
 *  it walks the stack top-down while the stack was built bottom-up.
 */

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/static_array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/experimental/core/tensor/coordinate_transform.hpp"
#include "ck_tile/experimental/core/tensor/transform_impl.hpp"

namespace ck_tile {

inline constexpr index_t MAX_TRANSFORMS = 5; ///< Max transforms per graph
inline constexpr index_t MAX_SLOTS =
    70; ///< Max working array slots (1 + MAX_TENSOR_DIMS for base Embed)

namespace detail {

/** @brief A directed acyclic graph of coordinate transforms.
 *
 *  General coordinate mapping: N input dimensions -> M output dimensions.
 *  Used as an NTTP: template<TransformGraph G> to guarantee compile-time
 *  constant folding of all transform parameters.
 *
 *  Common use case: tensor layout descriptor where ndim_lower=1 (memory offset).
 *  The graph is general and can map between any two coordinate spaces.
 *
 *  Internally routes indices through a working array of slots. The user
 *  never sees slot assignments -- they are computed by factory functions.
 *
 *  Construction:
 *    constexpr auto g = make_strided_graph(lengths, strides);
 *    constexpr auto g2 = applyTransforms(g, transforms, input_dims, output_dims);
 *
 *  Usage (as NTTP for guaranteed constant folding):
 *    constexpr auto g = ...;
 *    auto offset = calculateOffset<g>(static_array<index_t, 2>{m, k});
 *
 *  @note Internal routing fields (t_upper_slots, t_lower_slots, upper_slots,
 *        lower_slots, num_slots) must be public for structural NTTP requirements.
 *        They are assigned by factory functions. Users should not set them directly.
 */
struct TransformGraph
{
    // --- Transform stack (LIFO) ---
    // Transforms are pushed during construction (base first, user-facing last)
    // and popped during traversal (user-facing first, base last).
    // Index 0 = base (e.g., Embed), index num_transforms-1 = top (e.g., Merge).
    static_array<CoordinateTransform, MAX_TRANSFORMS> transforms{};
    index_t num_transforms = 0;

    // --- Internal routing (assigned by factory functions, not user-facing) ---
    // Per-transform: which working-array slots each transform reads from / writes to.
    static_array<DimIds, MAX_TRANSFORMS> t_upper_slots{};
    static_array<DimIds, MAX_TRANSFORMS> t_lower_slots{};
    index_t num_slots = 0;

    // --- Input / output dimensions ---
    index_t ndim_upper = 0; ///< Number of input coordinate dimensions
    index_t ndim_lower = 0; ///< Number of output coordinate dimensions
    static_array<index_t, MAX_TENSOR_DIMS> upper_slots{}; ///< Which slots receive input coords
    static_array<index_t, MAX_TENSOR_DIMS> lower_slots{}; ///< Which slots hold output coords

    constexpr bool operator==(const TransformGraph&) const = default;
};

static_assert(sizeof(TransformGraph) < 16384,
              "TransformGraph NTTP size canary — consider capacity templating if this fires");

// ============================================================================
// Free functions templated on TransformGraph NTTP
// ============================================================================

/** @brief Apply a single transform (called via fold expression, not directly by user).
 *
 *  @tparam G  The transform graph (NTTP)
 *  @tparam I  Fold index (0 = top of stack, num_transforms-1 = base)
 *  @param slots  Working array of index values
 */
template <TransformGraph G, index_t I>
CK_TILE_HOST_DEVICE constexpr void applySingleTransform(static_array<index_t, G.num_slots>& slots)
{
    constexpr index_t t      = G.num_transforms - 1 - I; // pop order (top of stack first)
    constexpr auto transform = G.transforms[t];

    // Gather inputs from working array
    static_array<index_t, MAX_TENSOR_DIMS> t_upper{};
    for(index_t d = 0; d < transform.ndim_upper; ++d)
    {
        t_upper[d] = slots[G.t_upper_slots[t][d]];
    }

    // Static dispatch — TransformImpl<Type> selected at compile time.
    // No switch, no dead branches, no runtime dispatch.
    static_array<index_t, MAX_TENSOR_DIMS> t_lower{};
    TransformImpl<transform.type>::mapIndices(transform, t_lower.elems, t_upper.elems);

    // Scatter outputs back to working array
    for(index_t d = 0; d < transform.ndim_lower; ++d)
    {
        slots[G.t_lower_slots[t][d]] = t_lower[d];
    }
}

/** @brief Pop all transforms from top to base via fold expression (internal dispatch).
 *
 *  @tparam G   The transform graph (NTTP)
 *  @tparam Is  Index sequence 0..num_transforms-1
 *  @param slots  Working array of index values
 */
template <TransformGraph G, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void applyAllTransforms(static_array<index_t, G.num_slots>& slots,
                                                      sequence<Is...>)
{
    (applySingleTransform<G, Is>(slots), ...);
}

/** @brief Map an input coordinate to an output coordinate.
 *
 *  Core operation of the transform graph. Pops transforms from the top of
 *  the stack (user-facing) to the base (memory-facing) using a fold
 *  expression for guaranteed compile-time dispatch.
 *
 *  @tparam G       The transform graph (NTTP — all values are compile-time constants)
 *  @param[out] output  Output coordinate buffer (size >= G.ndim_lower)
 *  @param[in]  input   Input coordinate buffer (size >= G.ndim_upper)
 */
template <TransformGraph G>
CK_TILE_HOST_DEVICE constexpr void map(index_t* lower, const index_t* upper)
{
    static_array<index_t, G.num_slots> slots{};

    // Place input coordinate into assigned slots
    for(index_t i = 0; i < G.ndim_upper; ++i)
    {
        slots[G.upper_slots[i]] = upper[i];
    }

    // Pop transforms from top of stack to base (user-facing → memory).
    // Fold expression guarantees compile-time dispatch.
    applyAllTransforms<G>(slots, make_index_sequence<G.num_transforms>{});

    // Extract lower coordinate from assigned slots
    for(index_t i = 0; i < G.ndim_lower; ++i)
    {
        lower[i] = slots[G.lower_slots[i]];
    }
}

/** @brief Convenience: compute memory offset when ndim_lower == 1.
 *
 *  Type-safe overload that takes a static_array and validates dimensions
 *  at compile time.
 *
 *  @tparam G  The transform graph (NTTP)
 *  @tparam N  Number of input dimensions (deduced from array size)
 *  @param input  Input coordinate
 *  @return The single output value (memory offset)
 */
template <TransformGraph G, index_t N>
CK_TILE_HOST_DEVICE constexpr index_t calculateOffset(const static_array<index_t, N>& input)
{
    static_assert(N == G.ndim_upper,
                  "calculateOffset: input coordinate size must match graph's ndim_upper");
    static_assert(G.ndim_lower == 1,
                  "calculateOffset: requires ndim_lower == 1 (use map() for N->M)");
    index_t output;
    map<G>(&output, input.elems);
    return output;
}

/// Check a single transform (compile-time index) for a matching slot.
template <TransformGraph G, index_t T>
CK_TILE_HOST_DEVICE constexpr index_t tryUpperLengthAt(index_t slot)
{
    constexpr auto transform = G.transforms[T];
    for(index_t d = 0; d < transform.ndim_upper; ++d)
    {
        if(G.t_upper_slots[T][d] == slot)
        {
            return TransformImpl<transform.type>::upperLength(transform, d);
        }
    }
    return -1; // not found at this transform
}

/// Fold over all transforms to find the one matching a slot.
template <TransformGraph G, index_t... Ts>
CK_TILE_HOST_DEVICE constexpr index_t upperDimLengthDispatch(index_t slot, sequence<Ts...>)
{
    index_t result = -1;
    // Check transforms in reverse order; keep first valid result
    auto check = [&](index_t candidate) {
        if(result == -1 && candidate != -1)
        {
            result = candidate;
        }
    };
    (check(tryUpperLengthAt<G, G.num_transforms - 1 - Ts>(slot)), ...);
    return result;
}

/** @brief Query the length of the i-th input dimension.
 *
 *  Uses compile-time fold expression to dispatch to the correct
 *  TransformImpl specialization (runtime loops cannot be used
 *  because TransformImpl<Type> requires a constant expression).
 *
 *  @tparam G  The transform graph (NTTP)
 *  @param i   Input dimension index (0-based)
 *  @return Length of the i-th input dimension
 */
template <TransformGraph G>
CK_TILE_HOST_DEVICE constexpr index_t upperDimLength(index_t i)
{
    index_t slot = G.upper_slots[i];
    return detail::upperDimLengthDispatch<G>(slot, make_index_sequence<G.num_transforms>{});
}

// ============================================================================
// Reverse traversal: output → input (memory offset → user coordinates)
// ============================================================================

/** @brief Apply one transform in reverse (called via fold expression).
 *
 *  Gathers from OUTPUT slots, calls reverseMapIndices, scatters to INPUT slots.
 *  Iterates in FORWARD array order (base first, user-facing last).
 *
 *  @tparam G  The transform graph (NTTP)
 *  @tparam I  Fold index (0 = base, num_transforms-1 = top)
 */
template <TransformGraph G, index_t I>
CK_TILE_HOST_DEVICE constexpr void
reverseApplySingleTransform(static_array<index_t, G.num_slots>& slots)
{
    constexpr index_t t      = I; // forward array order (base → top)
    constexpr auto transform = G.transforms[t];

    // Gather from output slots (we're going backwards through the wiring)
    static_array<index_t, MAX_TENSOR_DIMS> t_lower{};
    for(index_t d = 0; d < transform.ndim_lower; ++d)
    {
        t_lower[d] = slots[G.t_lower_slots[t][d]];
    }

    // Reverse dispatch — each TransformImpl provides reverseMapIndices
    static_array<index_t, MAX_TENSOR_DIMS> t_upper{};
    TransformImpl<transform.type>::reverseMapIndices(transform, t_upper.elems, t_lower.elems);

    // Scatter to input slots
    for(index_t d = 0; d < transform.ndim_upper; ++d)
    {
        slots[G.t_upper_slots[t][d]] = t_upper[d];
    }
}

/** @brief Apply all transforms in reverse via fold expression. */
template <TransformGraph G, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void
reverseApplyAllTransforms(static_array<index_t, G.num_slots>& slots, sequence<Is...>)
{
    (reverseApplySingleTransform<G, Is>(slots), ...);
}

/** @brief Reverse map: output coordinate → input coordinate.
 *
 *  Traverses the graph from base to top, applying reverseMapIndices
 *  on each transform. The reverse of map<G>().
 *
 *  Requires all transforms to be bijective (checked at compile time).
 *
 *  @tparam G       The transform graph (NTTP)
 *  @param[out] input   Recovered input coordinate (size >= G.ndim_upper)
 *  @param[in]  output  Output coordinate to reverse (size >= G.ndim_lower)
 */
template <TransformGraph G>
CK_TILE_HOST_DEVICE constexpr void reverseMap(index_t* upper, const index_t* lower)
{
    // Verify all transforms are bijective at compile time
    static_assert(
        []() constexpr {
            for(index_t t = 0; t < G.num_transforms; ++t)
                if(!G.transforms[t].is_bijective)
                    return false;
            return true;
        }(),
        "reverseMap: all transforms must be bijective");
    static_array<index_t, G.num_slots> slots{};

    // Place lower coordinate into lower slots
    for(index_t i = 0; i < G.ndim_lower; ++i)
    {
        slots[G.lower_slots[i]] = lower[i];
    }

    // Apply transforms in forward array order (base → top), each reversed
    reverseApplyAllTransforms<G>(slots, make_index_sequence<G.num_transforms>{});

    // Extract upper coordinate from upper slots
    for(index_t i = 0; i < G.ndim_upper; ++i)
    {
        upper[i] = slots[G.upper_slots[i]];
    }
}

/** @brief Convenience: recover user coordinates from a memory offset.
 *
 *  @tparam G  The transform graph (NTTP)
 *  @tparam N  Number of input dimensions (must match G.ndim_upper)
 *  @param offset  The memory offset to reverse
 *  @return The input coordinate that maps to this offset
 */
template <TransformGraph G>
CK_TILE_HOST_DEVICE constexpr static_array<index_t, G.ndim_upper>
reverseCalculateOffset(index_t offset)
{
    static_assert(G.ndim_lower == 1, "reverseCalculateOffset: requires ndim_lower == 1");
    static_array<index_t, G.ndim_upper> result{};
    reverseMap<G>(result.elems, &offset);
    return result;
}

} // namespace detail
} // namespace ck_tile
