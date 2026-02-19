// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HPP_

#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/various.hpp"
#include "config_types.hpp"

#include "../functional.hpp"
#include "../intrinsics.hpp"
#include "../types.hpp"

#include "../block/block_load.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/reverse_iterator.hpp"
#include "detail/device_segmented_radix_sort.hpp"
#include "device_partition.hpp"
#include "device_segmented_radix_sort_config.hpp"

/// \addtogroup devicemodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

struct Partitioner
{
    bool three_way_partitioning;

    Partitioner(bool three_way_part) : three_way_partitioning(three_way_part) {}

    template<typename InputIterator,
             typename FirstOutputIterator,
             typename SecondOutputIterator,
             typename UnselectedOutputIterator,
             typename SelectedCountOutputIterator,
             typename FirstUnaryPredicate,
             typename SecondUnaryPredicate>
    hipError_t operator()(void*                       temporary_storage,
                          size_t&                     storage_size,
                          InputIterator               input,
                          FirstOutputIterator         output_first_part,
                          SecondOutputIterator        output_second_part,
                          UnselectedOutputIterator    output_unselected,
                          SelectedCountOutputIterator selected_count_output,
                          const size_t                size,
                          FirstUnaryPredicate         select_first_part_op,
                          SecondUnaryPredicate        select_second_part_op,
                          const hipStream_t           stream,
                          const bool                  debug_synchronous)
    {
        using input_type = typename std::iterator_traits<InputIterator>::value_type;

        if(three_way_partitioning)
        {
            constexpr auto params = partition_config_params_base<input_type, true>();
            using config          = select_config<params.kernel_config.block_size,
                                                  params.kernel_config.items_per_thread>;
            return partition_three_way<config>(temporary_storage,
                                               storage_size,
                                               input,
                                               output_first_part,
                                               output_second_part,
                                               output_unselected,
                                               selected_count_output,
                                               size,
                                               select_first_part_op,
                                               select_second_part_op,
                                               stream,
                                               debug_synchronous);
        }
        else
        {
            constexpr auto params = partition_config_params_base<input_type, false>();
            using config          = select_config<params.kernel_config.block_size,
                                                  params.kernel_config.items_per_thread>;
            return partition<config>(temporary_storage,
                                     storage_size,
                                     input,
                                     output_first_part,
                                     selected_count_output,
                                     size,
                                     select_first_part_op,
                                     stream,
                                     debug_synchronous);
        }
    }
};

template<class Config,
         bool Descending,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetIterator>
inline hipError_t segmented_radix_sort_impl(
    void*                                                           temporary_storage,
    size_t&                                                         storage_size,
    KeysInputIterator                                               keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_tmp,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_tmp,
    ValuesOutputIterator                                            values_output,
    unsigned int                                                    size,
    bool&                                                           is_result_in_output,
    unsigned int                                                    num_segments,
    OffsetIterator                                                  begin_offsets,
    OffsetIterator                                                  end_offsets,
    unsigned int                                                    begin_bit,
    unsigned int                                                    end_bit,
    hipStream_t                                                     stream,
    bool                                                            debug_synchronous)
{
    using key_type               = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type             = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using segment_index_type     = unsigned int;
    using segment_index_iterator = counting_iterator<segment_index_type>;

    static_assert(
        std::is_same<key_type,
                     typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type");
    static_assert(
        std::is_same<value_type,
                     typename std::iterator_traits<ValuesOutputIterator>::value_type>::value,
        "ValuesInputIterator and ValuesOutputIterator must have the same value_type");

    using Selector = segmented_radix_sort_config_selector<key_type, value_type>;

    const target current_target(stream);

    const auto params = get_config<Selector>(Config{}, current_target);

    static constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    // Get partition boundary conditions.
    const unsigned int max_small_segment_length = params.warp_sort_config.items_per_thread_small
                                                  * params.warp_sort_config.logical_warp_size_small;
    const unsigned int max_medium_segment_length
        = params.warp_sort_config.items_per_thread_medium
          * params.warp_sort_config.logical_warp_size_medium;

    // Define partitioning selectors.
    const auto large_segment_selector = [=](const unsigned int segment_index) mutable -> bool
    {
        const unsigned int segment_length
            = end_offsets[segment_index] - begin_offsets[segment_index];
        return segment_length > max_medium_segment_length;
    };
    const auto medium_segment_selector = [=](const unsigned int segment_index) mutable -> bool
    {
        const unsigned int segment_length
            = end_offsets[segment_index] - begin_offsets[segment_index];
        return segment_length > max_small_segment_length;
    };

    // Define various algorithm parameters.
    const bool         with_double_buffer = keys_tmp != nullptr;
    const unsigned int bits               = end_bit - begin_bit;
    const unsigned int iterations         = ::rocprim::detail::ceiling_div(bits, params.radix_bits);
    const bool         to_output          = with_double_buffer || (iterations - 1) % 2 == 0;
    is_result_in_output                   = (iterations % 2 == 0) != to_output;

    // Check if we can use can use partition.
    const bool partitioning_allowed = params.warp_sort_config.partitioning_allowed;
    const bool do_partitioning
        = partitioning_allowed && num_segments >= params.warp_sort_config.partitioning_threshold;

    // When we do use partition, check if we should use 3-way partition.
    const bool   three_way_partitioning      = max_small_segment_length < max_medium_segment_length;
    const size_t medium_segment_indices_size = three_way_partitioning ? num_segments : 0;

    // Define the buffer size of the segment counters for the partioning results.
    constexpr size_t segment_count_output_size = 2;
    constexpr size_t segment_count_output_bytes
        = segment_count_output_size * sizeof(segment_index_type);

    // Large and small segments are in the same buffer, expanding from both ends.
    //   I.e.: [large... ...small]
    // We therefore have to reverse the small segment iterator.
    segment_index_type* large_segment_indices_output{};
    auto                small_segment_indices_output
        = make_reverse_iterator(large_segment_indices_output + num_segments);

    key_type*           keys_tmp_storage;
    value_type*         values_tmp_storage;
    segment_index_type* medium_segment_indices_output{};
    segment_index_type* segment_count_output{};
    size_t              partition_storage_size{};
    void*               partition_temporary_storage{};

    // Compute temporary storage requirement for partition.
    Partitioner partitioner(three_way_partitioning);
    ROCPRIM_RETURN_ON_ERROR(partitioner(nullptr,
                                        partition_storage_size,
                                        segment_index_iterator{},
                                        large_segment_indices_output,
                                        medium_segment_indices_output,
                                        small_segment_indices_output,
                                        segment_count_output,
                                        num_segments,
                                        large_segment_selector,
                                        medium_segment_selector,
                                        stream,
                                        debug_synchronous));

    // Declare temporary storage buffers.
    ROCPRIM_RETURN_ON_ERROR(detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            // These are required by both partitioning and by sorting.
            detail::temp_storage::ptr_aligned_array(&large_segment_indices_output, num_segments),
            detail::temp_storage::ptr_aligned_array(&medium_segment_indices_output,
                                                    medium_segment_indices_size),
            detail::temp_storage::ptr_aligned_array(&segment_count_output,
                                                    segment_count_output_size),
            detail::temp_storage::make_union_partition(
                // Partition temporary storage only needed by partitioning.
                detail::temp_storage::make_partition(&partition_temporary_storage,
                                                     partition_storage_size),
                // Keys/values temporary storage only needed by sorting.
                detail::temp_storage::make_linear_partition(
                    detail::temp_storage::ptr_aligned_array(&keys_tmp_storage,
                                                            !with_double_buffer ? size : 0),
                    detail::temp_storage::ptr_aligned_array(
                        &values_tmp_storage,
                        !with_double_buffer && with_values ? size : 0))))));

    if(num_segments == 0u || temporary_storage == nullptr)
    {
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "begin_bit " << begin_bit << '\n';
        std::cout << "end_bit " << end_bit << '\n';
        std::cout << "bits " << bits << '\n';
        std::cout << "segments " << num_segments << '\n';
        std::cout << "storage_size " << storage_size << '\n';
        std::cout << "iterations " << iterations << '\n';
        std::cout << "do_partitioning " << do_partitioning << '\n';
        std::cout << "params.kernel_config.block_size: " << params.kernel_config.block_size << '\n';
        std::cout << "params.kernel_config.items_per_thread: "
                  << params.kernel_config.items_per_thread << '\n';
        ROCPRIM_RETURN_ON_ERROR(hipStreamSynchronize(stream));
    }

    if(!with_double_buffer)
    {
        keys_tmp   = keys_tmp_storage;
        values_tmp = values_tmp_storage;
    }
    small_segment_indices_output
        = make_reverse_iterator(large_segment_indices_output + num_segments);

    bool is_capturing = false;
    ROCPRIM_RETURN_ON_ERROR(detail::is_graph_capture(stream, is_capturing));

    std::chrono::steady_clock::time_point start;
    if(do_partitioning)
    {
        // This branch does an analysis step using partition before
        // sorting. We do more initial work to more efficiently utilize
        // blocks in case of many small segments.

        // Our kernels are not fully aware if we're using partition
        // or not. We thereful have to ensure that the output is set
        // properly beforehand. We can do this easily with a copy.
        if(is_capturing)
        {
            ROCPRIM_RETURN_ON_ERROR(
                hipMemsetAsync(segment_count_output, 0, 2 * sizeof(segment_index_type), stream));
        }

        ROCPRIM_RETURN_ON_ERROR(partitioner(partition_temporary_storage,
                                            partition_storage_size,
                                            segment_index_iterator{},
                                            large_segment_indices_output,
                                            medium_segment_indices_output,
                                            small_segment_indices_output,
                                            segment_count_output,
                                            num_segments,
                                            large_segment_selector,
                                            medium_segment_selector,
                                            stream,
                                            debug_synchronous));
        std::vector<segment_index_type> segment_counts(segment_count_output_size,
                                                       segment_index_type{});

        // Now we need to optionally compute the number of blocks
        // we need to launch. This is trivial to compute once we
        // have the segment counts.
        //
        // We can ignore this on graph based launches, since we'd
        // have to ferry the arguments through device memory.
        dim3 large_segment_blocks  = 0;
        dim3 medium_segment_blocks = 0;
        dim3 small_segment_blocks  = 0;

        segment_index_type large_segment_count  = 0;
        segment_index_type medium_segment_count = 0;
        segment_index_type small_segment_count  = 0;

        // Optionally copy device to host to get segment and
        // block counts.
        // * If we're using debug, we need the data on host.
        // * If we're NOT capturing, we can optimize our launches
        //   too by using the data on host.
        if(!is_capturing)
        {
            ROCPRIM_RETURN_ON_ERROR(detail::memcpy_and_sync(segment_counts.data(),
                                                            segment_count_output,
                                                            segment_count_output_bytes,
                                                            hipMemcpyDeviceToHost,
                                                            stream));
            large_segment_count  = segment_counts[0];
            medium_segment_count = three_way_partitioning ? segment_counts[1] : 0;
            small_segment_count  = num_segments - large_segment_count - medium_segment_count;

            const auto medium_warps_per_block = params.warp_sort_config.block_size_medium
                                                / params.warp_sort_config.logical_warp_size_medium;

            const auto small_warps_per_block = params.warp_sort_config.block_size_small
                                               / params.warp_sort_config.logical_warp_size_small;

            // We sort large segments as one segment per block. Smaller segments we
            // sort as one segment per warp.
            large_segment_blocks  = large_segment_count;
            medium_segment_blocks = ceiling_div(medium_segment_count, medium_warps_per_block);
            small_segment_blocks  = ceiling_div(small_segment_count, small_warps_per_block);
        }

        // Define individual kernel launches.
        //
        // When we have a host-defined grid size (i.e. not a grapgh capture),
        // we need to pass this grid size, otherwise use maximum active blocks.
        // We implement this using 'launch_maybe_bounded_plan' and call this for
        // each of our kernel launches.
        const auto launch_maybe_bounded_plan
            = [&](auto plan, auto is_unknown_grid_size, auto grid_size, auto block_size)
        {
            return is_unknown_grid_size ? plan.launch_with_max_active_blocks(block_size, 0, stream)
                                        : plan.launch(grid_size, block_size, 0, stream);
        };

        const auto launch_large_segments_kernel = [&](auto is_unknown_grid_size)
        {
            auto kernel = [=](auto arch_config)
            {
                sort_block_sized_segments<decltype(arch_config), Descending, is_unknown_grid_size>(
                    segment_count_output,
                    keys_input,
                    keys_tmp,
                    keys_output,
                    values_input,
                    values_tmp,
                    values_output,
                    to_output,
                    large_segment_indices_output,
                    begin_offsets,
                    end_offsets,
                    iterations,
                    begin_bit,
                    end_bit);
            };
            const auto plan = make_launch_plan<Config, Selector>(current_target, kernel);
            return launch_maybe_bounded_plan(plan,
                                             is_unknown_grid_size,
                                             large_segment_blocks,
                                             params.kernel_config.block_size);
        };

        const auto launch_medium_segments_kernel = [&](auto is_unknown_grid_size)
        {
            auto kernel = [=](auto target_config)
            {
                sort_warp_sized_segments<decltype(target_config),
                                         detail::medium_segment_config_extractor,
                                         Descending,
                                         is_unknown_grid_size>(segment_count_output,
                                                               keys_input,
                                                               keys_tmp,
                                                               keys_output,
                                                               values_input,
                                                               values_tmp,
                                                               values_output,
                                                               is_result_in_output,
                                                               medium_segment_indices_output,
                                                               begin_offsets,
                                                               end_offsets,
                                                               begin_bit,
                                                               end_bit,
                                                               num_segments,
                                                               medium_segment_count);
            };
            const auto plan
                = make_launch_plan<Config,
                                   Selector,
                                   segmented_radix_sort_warp_sort_medium_config_static_selector>(
                    current_target,
                    kernel);
            return launch_maybe_bounded_plan(plan,
                                             is_unknown_grid_size,
                                             medium_segment_blocks,
                                             params.warp_sort_config.block_size_medium);
        };

        const auto launch_small_segments_kernel = [&](auto is_unknown_grid_size)
        {
            auto kernel = [=](auto arch_config)
            {
                sort_warp_sized_segments<decltype(arch_config),
                                         detail::small_segment_config_extractor,
                                         Descending,
                                         is_unknown_grid_size>(segment_count_output,
                                                               keys_input,
                                                               keys_tmp,
                                                               keys_output,
                                                               values_input,
                                                               values_tmp,
                                                               values_output,
                                                               is_result_in_output,
                                                               small_segment_indices_output,
                                                               begin_offsets,
                                                               end_offsets,
                                                               begin_bit,
                                                               end_bit,
                                                               num_segments,
                                                               small_segment_count);
            };
            const auto plan
                = make_launch_plan<Config,
                                   Selector,
                                   segmented_radix_sort_warp_sort_small_config_static_selector>(
                    current_target,
                    kernel);
            return launch_maybe_bounded_plan(plan,
                                             is_unknown_grid_size,
                                             small_segment_blocks,
                                             params.warp_sort_config.block_size_medium);
        };

        // Launch previously defined kernels.
        //
        // We're using std::visit here to generate kernel variations, one
        // where we pass host launch grid size according to the segment count
        // and one where we just use the maximal active blocks.
        //
        // The reason for this is that we cannot setup host launches on graph
        // captures. The extra overhead from in-block tiling causes non-graph
        // launches to regress in performance. The solution is to just have
        // seperate kernels for these two situations.
        const auto is_unknown_grid_size
            = rocprim::detail::constexpr_value_variant<bool, false, true>::create(is_capturing);
        ROCPRIM_RETURN_ON_ERROR(std::visit(
            [&](auto is_unknown_grid_size)
            {
                // Sort large segments using block-level algorithm.
                if(is_unknown_grid_size || large_segment_count > 0)
                {
                    ROCPRIM_DETAIL_DEBUG_SYNC_START(start);
                    ROCPRIM_RETURN_ON_ERROR(launch_large_segments_kernel(is_unknown_grid_size));
                    ROCPRIM_DETAIL_DEBUG_SYNC_END("segmented_sort:large_segments",
                                                  large_segment_count,
                                                  start);
                }

                // When using 3-way partitioning, we may have medium segments.
                // Sort medium segments using warp sort.
                if(three_way_partitioning && (is_unknown_grid_size || medium_segment_count > 0))
                {
                    ROCPRIM_DETAIL_DEBUG_SYNC_START(start);
                    ROCPRIM_RETURN_ON_ERROR(launch_medium_segments_kernel(is_unknown_grid_size));
                    ROCPRIM_DETAIL_DEBUG_SYNC_END("segmented_sort:medium_segments",
                                                  medium_segment_count,
                                                  start);
                }

                // Sort small segments using warp sort.
                if(is_unknown_grid_size || small_segment_count > 0)
                {
                    ROCPRIM_DETAIL_DEBUG_SYNC_START(start);
                    ROCPRIM_RETURN_ON_ERROR(launch_small_segments_kernel(is_unknown_grid_size));
                    ROCPRIM_DETAIL_DEBUG_SYNC_END("segmented_sort:small_segments",
                                                  small_segment_count,
                                                  start);
                }

                return hipSuccess;
            },
            is_unknown_grid_size));
    }
    else
    {
        // This branch does not use partitioning.
        //
        // We can sort everything in a single kernel launch!
        const auto launch_single_pass_kernel = [&]()
        {
            auto segmented_sort_kernel = [=](auto target_config)
            {
                segmented_sort<decltype(target_config), Descending>(keys_input,
                                                                  keys_tmp,
                                                                  keys_output,
                                                                  values_input,
                                                                  values_tmp,
                                                                  values_output,
                                                                  to_output,
                                                                  begin_offsets,
                                                                  end_offsets,
                                                                  iterations,
                                                                  begin_bit,
                                                                  end_bit);
            };
            return execute_launch_plan<Config, Selector>(current_target,
                                                         segmented_sort_kernel,
                                                         dim3(num_segments),
                                                         dim3(params.kernel_config.block_size),
                                                         0,
                                                         stream);
        };

        ROCPRIM_DETAIL_DEBUG_SYNC_START(start);
        ROCPRIM_RETURN_ON_ERROR(launch_single_pass_kernel());
        ROCPRIM_DETAIL_DEBUG_SYNC_END("segmented_sort", num_segments, start);
    }
    return hipSuccess;
}

} // end namespace detail

/// \brief Parallel ascending radix sort primitive for device level.
///
/// \p segmented_radix_sort_keys function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of keys. Function sorts input keys in ascending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input and \p keys_output must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_keys is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam KeysInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first element in the range to sort.
/// \param [out] keys_output pointer to the first element in the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;      // e.g., 8
/// float * input;          // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// float * output;         // empty array of 8 elements
/// unsigned int segments;  // e.g., 3
/// int * offsets;          // e.g. [0, 2, 3, 8]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size,
///     segments, offsets, offsets + 1
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size,
///     segments, offsets, offsets + 1
/// );
/// // keys_output: [0.3, 0.6, 0.65, 0.08, 0.2, 0.4, 0.7, 1]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class OffsetIterator,
         class Key = typename std::iterator_traits<KeysInputIterator>::value_type>
inline hipError_t segmented_radix_sort_keys(void*              temporary_storage,
                                            size_t&            storage_size,
                                            KeysInputIterator  keys_input,
                                            KeysOutputIterator keys_output,
                                            unsigned int       size,
                                            unsigned int       segments,
                                            OffsetIterator     begin_offsets,
                                            OffsetIterator     end_offsets,
                                            unsigned int       begin_bit         = 0,
                                            unsigned int       end_bit           = 8 * sizeof(Key),
                                            hipStream_t        stream            = 0,
                                            bool               debug_synchronous = false)
{
    empty_type* values = nullptr;
    bool        ignored;
    return detail::segmented_radix_sort_impl<Config, false>(temporary_storage,
                                                            storage_size,
                                                            keys_input,
                                                            nullptr,
                                                            keys_output,
                                                            values,
                                                            nullptr,
                                                            values,
                                                            size,
                                                            ignored,
                                                            segments,
                                                            begin_offsets,
                                                            end_offsets,
                                                            begin_bit,
                                                            end_bit,
                                                            stream,
                                                            debug_synchronous);
}

/// \brief Parallel descending radix sort primitive for device level.
///
/// \p segmented_radix_sort_keys_desc function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of keys. Function sorts input keys in descending order.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input and \p keys_output must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_keys_desc is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam KeysInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first element in the range to sort.
/// \param [out] keys_output pointer to the first element in the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;      // e.g., 8
/// int * input;            // e.g., [6, 3, 5, 4, 2, 8, 1, 7]
/// int * output;           // empty array of 8 elements
/// unsigned int segments;  // e.g., 3
/// int * offsets;          // e.g. [0, 2, 3, 8]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size,
///     segments, offsets, offsets + 1
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, output, input_size,
///     segments, offsets, offsets + 1
/// );
/// // keys_output: [6, 3, 5, 8, 7, 4, 2, 1]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class OffsetIterator,
         class Key = typename std::iterator_traits<KeysInputIterator>::value_type>
inline hipError_t segmented_radix_sort_keys_desc(void*              temporary_storage,
                                                 size_t&            storage_size,
                                                 KeysInputIterator  keys_input,
                                                 KeysOutputIterator keys_output,
                                                 unsigned int       size,
                                                 unsigned int       segments,
                                                 OffsetIterator     begin_offsets,
                                                 OffsetIterator     end_offsets,
                                                 unsigned int       begin_bit = 0,
                                                 unsigned int       end_bit   = 8 * sizeof(Key),
                                                 hipStream_t        stream    = 0,
                                                 bool               debug_synchronous = false)
{
    empty_type* values = nullptr;
    bool        ignored;
    return detail::segmented_radix_sort_impl<Config, true>(temporary_storage,
                                                           storage_size,
                                                           keys_input,
                                                           nullptr,
                                                           keys_output,
                                                           values,
                                                           nullptr,
                                                           values,
                                                           size,
                                                           ignored,
                                                           segments,
                                                           begin_offsets,
                                                           end_offsets,
                                                           begin_bit,
                                                           end_bit,
                                                           stream,
                                                           debug_synchronous);
}

/// \brief Parallel ascending radix sort-by-key primitive for device level.
///
/// \p segmented_radix_sort_pairs function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of (key, value) pairs. Function sorts input pairs in ascending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_pairs is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam KeysInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first element in the range to sort.
/// \param [out] keys_output pointer to the first element in the output range.
/// \param [in] values_input pointer to the first element in the range to sort.
/// \param [out] values_output pointer to the first element in the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_output; // empty array of 8 elements
/// double * values_output;     // empty array of 8 elements
/// unsigned int segments;      // e.g., 3
/// int * offsets;              // e.g. [0, 2, 3, 8]
///
/// // Keys are in range [0; 8], so we can limit compared bit to bits on indexes
/// // 0, 1, 2, 3, and 4. In order to do this begin_bit is set to 0 and end_bit
/// // is set to 5.
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output, input_size,
///     segments, offsets, offsets + 1,
///     0, 5
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output, input_size,
///     segments, offsets, offsets + 1,
///     0, 5
/// );
/// // keys_output:   [3,  6,  5,  1,  1, 4, 7,  8]
/// // values_output: [2, -5, -4, -1, -2, 3, 7, -8]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetIterator,
         class Key = typename std::iterator_traits<KeysInputIterator>::value_type>
inline hipError_t segmented_radix_sort_pairs(void*                temporary_storage,
                                             size_t&              storage_size,
                                             KeysInputIterator    keys_input,
                                             KeysOutputIterator   keys_output,
                                             ValuesInputIterator  values_input,
                                             ValuesOutputIterator values_output,
                                             unsigned int         size,
                                             unsigned int         segments,
                                             OffsetIterator       begin_offsets,
                                             OffsetIterator       end_offsets,
                                             unsigned int         begin_bit = 0,
                                             unsigned int         end_bit   = 8 * sizeof(Key),
                                             hipStream_t          stream    = 0,
                                             bool                 debug_synchronous = false)
{
    bool ignored;
    return detail::segmented_radix_sort_impl<Config, false>(temporary_storage,
                                                            storage_size,
                                                            keys_input,
                                                            nullptr,
                                                            keys_output,
                                                            values_input,
                                                            nullptr,
                                                            values_output,
                                                            size,
                                                            ignored,
                                                            segments,
                                                            begin_offsets,
                                                            end_offsets,
                                                            begin_bit,
                                                            end_bit,
                                                            stream,
                                                            debug_synchronous);
}

/// \brief Parallel descending radix sort-by-key primitive for device level.
///
/// \p segmented_radix_sort_pairs_desc function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of (key, value) pairs. Function sorts input pairs in descending order of keys.
///
/// \par Overview
/// * The contents of the inputs are not altered by the sorting function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * \p Key type (a \p value_type of \p KeysInputIterator and \p KeysOutputIterator) must be
/// an arithmetic type (that is, an integral type or a floating-point type).
/// * Ranges specified by \p keys_input, \p keys_output, \p values_input and \p values_output must
/// have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_pairs_desc is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
/// (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam KeysInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam KeysOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam ValuesInputIterator random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam ValuesOutputIterator random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input pointer to the first element in the range to sort.
/// \param [out] keys_output pointer to the first element in the output range.
/// \param [in] values_input pointer to the first element in the range to sort.
/// \param [out] values_output pointer to the first element in the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed where input keys are
/// represented by an array of integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// int * keys_input;        // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;   // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// int * keys_output;       // empty array of 8 elements
/// double * values_output;  // empty array of 8 elements
/// unsigned int segments;   // e.g., 3
/// int * offsets;           // e.g. [0, 2, 3, 8]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size,
///     segments, offsets, offsets + 1
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, values_input, values_output,
///     input_size,
///     segments, offsets, offsets + 1
/// );
/// // keys_output:   [ 6, 3,  5,  8, 7, 4,  1,  1]
/// // values_output: [-5, 2, -4, -8, 7, 3, -1, -2]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class OffsetIterator,
         class Key = typename std::iterator_traits<KeysInputIterator>::value_type>
inline hipError_t segmented_radix_sort_pairs_desc(void*                temporary_storage,
                                                  size_t&              storage_size,
                                                  KeysInputIterator    keys_input,
                                                  KeysOutputIterator   keys_output,
                                                  ValuesInputIterator  values_input,
                                                  ValuesOutputIterator values_output,
                                                  unsigned int         size,
                                                  unsigned int         segments,
                                                  OffsetIterator       begin_offsets,
                                                  OffsetIterator       end_offsets,
                                                  unsigned int         begin_bit = 0,
                                                  unsigned int         end_bit   = 8 * sizeof(Key),
                                                  hipStream_t          stream    = 0,
                                                  bool                 debug_synchronous = false)
{
    bool ignored;
    return detail::segmented_radix_sort_impl<Config, true>(temporary_storage,
                                                           storage_size,
                                                           keys_input,
                                                           nullptr,
                                                           keys_output,
                                                           values_input,
                                                           nullptr,
                                                           values_output,
                                                           size,
                                                           ignored,
                                                           segments,
                                                           begin_offsets,
                                                           end_offsets,
                                                           begin_bit,
                                                           end_bit,
                                                           stream,
                                                           debug_synchronous);
}

/// \brief Parallel ascending radix sort primitive for device level.
///
/// \p segmented_radix_sort_keys function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of keys. Function sorts input keys in ascending order.
///
/// \par Overview
/// * The contents of both buffers of \p keys may be altered by the sorting function.
/// * \p current() of \p keys is used as the input.
/// * The function will update \p current() of \p keys to point to the buffer
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_keys is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam Key key type. Must be an integral type or a floating-point type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed on an array of
/// \p float values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// float * input;           // e.g., [0.6, 0.3, 0.65, 0.4, 0.2, 0.08, 1, 0.7]
/// float * tmp;             // empty array of 8 elements
/// unsigned int segments;   // e.g., 3
/// int * offsets;           // e.g. [0, 2, 3, 8]
/// // Create double-buffer
/// rocprim::double_buffer<float> keys(input, tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size,
///     segments, offsets, offsets + 1
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_keys(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size,
///     segments, offsets, offsets + 1
/// );
/// // keys.current(): [0.3, 0.6, 0.65, 0.08, 0.2, 0.4, 0.7, 1]
/// \endcode
/// \endparblock
template<class Config = default_config, class Key, class OffsetIterator>
inline hipError_t segmented_radix_sort_keys(void*               temporary_storage,
                                            size_t&             storage_size,
                                            double_buffer<Key>& keys,
                                            unsigned int        size,
                                            unsigned int        segments,
                                            OffsetIterator      begin_offsets,
                                            OffsetIterator      end_offsets,
                                            unsigned int        begin_bit         = 0,
                                            unsigned int        end_bit           = 8 * sizeof(Key),
                                            hipStream_t         stream            = 0,
                                            bool                debug_synchronous = false)
{
    empty_type* values = nullptr;
    bool        is_result_in_output;
    hipError_t  error = detail::segmented_radix_sort_impl<Config, false>(temporary_storage,
                                                                        storage_size,
                                                                        keys.current(),
                                                                        keys.current(),
                                                                        keys.alternate(),
                                                                        values,
                                                                        values,
                                                                        values,
                                                                        size,
                                                                        is_result_in_output,
                                                                        segments,
                                                                        begin_offsets,
                                                                        end_offsets,
                                                                        begin_bit,
                                                                        end_bit,
                                                                        stream,
                                                                        debug_synchronous);
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

/// \brief Parallel descending radix sort primitive for device level.
///
/// \p segmented_radix_sort_keys_desc function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of keys. Function sorts input keys in descending order.
///
/// \par Overview
/// * The contents of both buffers of \p keys may be altered by the sorting function.
/// * \p current() of \p keys is used as the input.
/// * The function will update \p current() of \p keys to point to the buffer
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_keys_desc is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam Key key type. Must be an integral type or a floating-point type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// int * input;             // e.g., [6, 3, 5, 4, 2, 8, 1, 7]
/// int * tmp;               // empty array of 8 elements
/// unsigned int segments;   // e.g., 3
/// int * offsets;           // e.g. [0, 2, 3, 8]
/// // Create double-buffer
/// rocprim::double_buffer<int> keys(input, tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size,
///     segments, offsets, offsets + 1
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_keys_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, input_size,
///     segments, offsets, offsets + 1
/// );
/// // keys.current(): [6, 3, 5, 8, 7, 4, 2, 1]
/// \endcode
/// \endparblock
template<class Config = default_config, class Key, class OffsetIterator>
inline hipError_t segmented_radix_sort_keys_desc(void*               temporary_storage,
                                                 size_t&             storage_size,
                                                 double_buffer<Key>& keys,
                                                 unsigned int        size,
                                                 unsigned int        segments,
                                                 OffsetIterator      begin_offsets,
                                                 OffsetIterator      end_offsets,
                                                 unsigned int        begin_bit = 0,
                                                 unsigned int        end_bit   = 8 * sizeof(Key),
                                                 hipStream_t         stream    = 0,
                                                 bool                debug_synchronous = false)
{
    empty_type* values = nullptr;
    bool        is_result_in_output;
    hipError_t  error = detail::segmented_radix_sort_impl<Config, true>(temporary_storage,
                                                                       storage_size,
                                                                       keys.current(),
                                                                       keys.current(),
                                                                       keys.alternate(),
                                                                       values,
                                                                       values,
                                                                       values,
                                                                       size,
                                                                       is_result_in_output,
                                                                       segments,
                                                                       begin_offsets,
                                                                       end_offsets,
                                                                       begin_bit,
                                                                       end_bit,
                                                                       stream,
                                                                       debug_synchronous);
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

/// \brief Parallel ascending radix sort-by-key primitive for device level.
///
/// \p segmented_radix_sort_pairs function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of (key, value) pairs. Function sorts input pairs in ascending order of keys.
///
/// \par Overview
/// * The contents of both buffers of \p keys and \p values may be altered by the sorting function.
/// * \p current() of \p keys and \p values are used as the input.
/// * The function will update \p current() of \p keys and \p values to point to buffers
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_pairs is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
/// (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam Key key type. Must be an integral type or a floating-point type.
/// \tparam Value value type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in,out] values reference to the double-buffer of values, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level ascending radix sort is performed where input keys are
/// represented by an array of unsigned integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// unsigned int * keys_input;  // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;      // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// unsigned int * keys_tmp;    // empty array of 8 elements
/// double*  values_tmp;        // empty array of 8 elements
/// unsigned int segments;      // e.g., 3
/// int * offsets;              // e.g. [0, 2, 3, 8]
/// // Create double-buffers
/// rocprim::double_buffer<unsigned int> keys(keys_input, keys_tmp);
/// rocprim::double_buffer<double> values(values_input, values_tmp);
///
/// // Keys are in range [0; 8], so we can limit compared bit to bits on indexes
/// // 0, 1, 2, 3, and 4. In order to do this begin_bit is set to 0 and end_bit
/// // is set to 5.
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     segments, offsets, offsets + 1
///     0, 5
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_pairs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     segments, offsets, offsets + 1
///     0, 5
/// );
/// // keys.current():   [3,  6,  5,  1,  1, 4, 7,  8]
/// // values.current(): [2, -5, -4, -1, -2, 3, 7, -8]
/// \endcode
/// \endparblock
template<class Config = default_config, class Key, class Value, class OffsetIterator>
inline hipError_t segmented_radix_sort_pairs(void*                 temporary_storage,
                                             size_t&               storage_size,
                                             double_buffer<Key>&   keys,
                                             double_buffer<Value>& values,
                                             unsigned int          size,
                                             unsigned int          segments,
                                             OffsetIterator        begin_offsets,
                                             OffsetIterator        end_offsets,
                                             unsigned int          begin_bit = 0,
                                             unsigned int          end_bit   = 8 * sizeof(Key),
                                             hipStream_t           stream    = 0,
                                             bool                  debug_synchronous = false)
{
    bool       is_result_in_output;
    hipError_t error = detail::segmented_radix_sort_impl<Config, false>(temporary_storage,
                                                                        storage_size,
                                                                        keys.current(),
                                                                        keys.current(),
                                                                        keys.alternate(),
                                                                        values.current(),
                                                                        values.current(),
                                                                        values.alternate(),
                                                                        size,
                                                                        is_result_in_output,
                                                                        segments,
                                                                        begin_offsets,
                                                                        end_offsets,
                                                                        begin_bit,
                                                                        end_bit,
                                                                        stream,
                                                                        debug_synchronous);
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

/// \brief Parallel descending radix sort-by-key primitive for device level.
///
/// \p segmented_radix_sort_pairs_desc function performs a device-wide radix sort across multiple,
/// non-overlapping sequences of (key, value) pairs. Function sorts input pairs in descending order of keys.
///
/// \par Overview
/// * The contents of both buffers of \p keys and \p values may be altered by the sorting function.
/// * \p current() of \p keys and \p values are used as the input.
/// * The function will update \p current() of \p keys and \p values to point to buffers
/// that contains the output range.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * The function requires small \p temporary_storage as it does not need
/// a temporary buffer of \p size elements.
/// * \p Key type must be an arithmetic type (that is, an integral type or a floating-point
/// type).
/// * Buffers of \p keys must have at least \p size elements.
/// * Ranges specified by \p begin_offsets and \p end_offsets must have
/// at least \p segments elements. They may use the same sequence <tt>offsets</tt> of at least
/// <tt>segments + 1</tt> elements: <tt>offsets</tt> for \p begin_offsets and
/// <tt>offsets + 1</tt> for \p end_offsets.
/// * If \p Key is an integer type and the range of keys is known in advance, the performance
/// can be improved by setting \p begin_bit and \p end_bit, for example if all keys are in range
/// [100, 10000], <tt>begin_bit = 0</tt> and <tt>end_bit = 14</tt> will cover the whole range.
///
/// \par Stability
/// \p segmented_radix_sort_pairs_desc is \b stable: it preserves the relative ordering of equivalent keys.
/// That is, given two keys \p a and \p b and a binary boolean operation \p op such that:
///   * \p a precedes \p b in the input keys, and
///   * op(a, b) and op(b, a) are both false,
/// then it is \b guaranteed that \p a will precede \p b as well in the output (ordered) keys.
/// (ordered) keys.
///
/// \tparam Config [optional] Configuration of the primitive, must be `default_config` or `segmented_radix_sort_config`.
/// \tparam Key key type. Must be an integral type or a floating-point type.
/// \tparam Value value type.
/// \tparam OffsetIterator random-access iterator type of segment offsets. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size reference to a size (in bytes) of \p temporary_storage.
/// \param [in,out] keys reference to the double-buffer of keys, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in,out] values reference to the double-buffer of values, its \p current()
/// contains the input range and will be updated to point to the output range.
/// \param [in] size number of element in the input range.
/// \param [in] segments number of segments in the input range.
/// \param [in] begin_offsets iterator to the first element in the range of beginning offsets.
/// \param [in] end_offsets iterator to the first element in the range of ending offsets.
/// \param [in] begin_bit [optional] index of the first (least significant) bit used in
/// key comparison. Must be in range <tt>[0; 8 * sizeof(Key))</tt>. Default value: \p 0.
/// Non-default value not supported for floating-point key-types.
/// \param [in] end_bit [optional] past-the-end index (most significant) bit used in
/// key comparison. Must be in range <tt>(begin_bit; 8 * sizeof(Key)]</tt>. Default
/// value: \p <tt>8 * sizeof(Key)</tt>. Non-default value not supported for floating-point key-types.
/// \param [in] stream [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level descending radix sort is performed where input keys are
/// represented by an array of integers and input values by an array of <tt>double</tt>s.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and tmp (declare pointers, allocate device memory etc.)
/// size_t input_size;       // e.g., 8
/// int * keys_input;        // e.g., [ 6, 3,  5, 4,  1,  8,  1, 7]
/// double * values_input;   // e.g., [-5, 2, -4, 3, -1, -8, -2, 7]
/// int * keys_tmp;          // empty array of 8 elements
/// double * values_tmp;     // empty array of 8 elements
/// unsigned int segments;   // e.g., 3
/// int * offsets;           // e.g. [0, 2, 3, 8]
/// // Create double-buffers
/// rocprim::double_buffer<int> keys(keys_input, keys_tmp);
/// rocprim::double_buffer<double> values(values_input, values_tmp);
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::segmented_radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     segments, offsets, offsets + 1
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform sort
/// rocprim::segmented_radix_sort_pairs_desc(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys, values, input_size,
///     segments, offsets, offsets + 1
/// );
/// // keys.current():   [ 6, 3,  5,  8, 7, 4,  1,  1]
/// // values.current(): [-5, 2, -4, -8, 7, 3, -1, -2]
/// \endcode
/// \endparblock
template<class Config = default_config, class Key, class Value, class OffsetIterator>
inline hipError_t segmented_radix_sort_pairs_desc(void*                 temporary_storage,
                                                  size_t&               storage_size,
                                                  double_buffer<Key>&   keys,
                                                  double_buffer<Value>& values,
                                                  unsigned int          size,
                                                  unsigned int          segments,
                                                  OffsetIterator        begin_offsets,
                                                  OffsetIterator        end_offsets,
                                                  unsigned int          begin_bit = 0,
                                                  unsigned int          end_bit   = 8 * sizeof(Key),
                                                  hipStream_t           stream    = 0,
                                                  bool                  debug_synchronous = false)
{
    bool       is_result_in_output;
    hipError_t error = detail::segmented_radix_sort_impl<Config, true>(temporary_storage,
                                                                       storage_size,
                                                                       keys.current(),
                                                                       keys.current(),
                                                                       keys.alternate(),
                                                                       values.current(),
                                                                       values.current(),
                                                                       values.alternate(),
                                                                       size,
                                                                       is_result_in_output,
                                                                       segments,
                                                                       begin_offsets,
                                                                       end_offsets,
                                                                       begin_bit,
                                                                       end_bit,
                                                                       stream,
                                                                       debug_synchronous);
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group devicemodule

#endif // ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HPP_
