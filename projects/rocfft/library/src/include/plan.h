// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef PLAN_H
#define PLAN_H

#include <array>
#include <complex>
#include <cstring>
#include <list>
#include <vector>

#include "../../../shared/array_predicate.h"
#include "data_layout.h"
#include "function_pool.h"
#include "load_store_ops.h"
#include "rocfft_mpi.h"
#include "tree_node.h"

// Calculate the maximum pow number with the given base number
template <int base>
constexpr size_t PowMax()
{
    size_t u = base;
    while(u < std::numeric_limits<size_t>::max() / base)
    {
        u = u * base;
    }
    return u;
}

// types of grid layouts for global transpositions
enum class grid_layout
{
    invalid = 0,
    slab    = 1,
    pencil  = 2,
    brick   = 3
};
using transpose_type = std::pair<grid_layout, grid_layout>;

// Generic function to check is pow of a given base number or not
template <int base>
static inline bool IsPow(size_t u)
{
    constexpr size_t max = PowMax<base>(); //Practically, we could save this by using 3486784401
    return (u > 0 && max % u == 0);
}

struct rocfft_brick_t
{
    // no default constructor
    rocfft_brick_t() = delete;
    // default move and copy constructors
    rocfft_brick_t(const rocfft_brick_t&) = default;
    rocfft_brick_t& operator=(const rocfft_brick_t&) = default;
    rocfft_brick_t(rocfft_brick_t&&)                 = default;
    rocfft_brick_t& operator=(rocfft_brick_t&&) = default;

    // all vectors here are column-major, with same size as FFT
    // rank + 1 (batch axis last)
    rocfft_brick_t(const std::vector<size_t>& field_lower,
                   const std::vector<size_t>& field_upper,
                   const std::vector<size_t>& brick_stride,
                   const rocfft_location_t&   location)
        : layout(field_lower, field_upper, brick_stride)
        , location(location)
    {
    }

    // Data layout of the brick
    data_layout_t layout;
    // Location of the brick
    rocfft_location_t location;

    bool operator==(const rocfft_brick_t& other) const
    {
        return layout == other.layout && location == other.location;
    }

    std::string str() const;
};

struct rocfft_field_t
{
    std::vector<rocfft_brick_t> bricks;

    /**
     * @brief Finalize all the bricks for a given full-range data set
     * that `full_range_layout` captures:
     * 
     * - sort field's bricks by increasing rank (stable sort);
     * 
     * - set the `is_partial` flags for all axes of all the bricks' layouts,
     * in light of the given full-range layout.
     * 
     * @param[in] full_range_layout A full (i.e. non-partial) data layout
     * capturing the range of logical indices that this field's bricks are
     * expected to cover.
     * 
     * @note The in-buffer strides of `full_range_layout` are irrelevant for
     * this function; only the range of logical indices along all axes matters.
     * 
     * @throw An `std::invalid_argument` is thrown if `full_range_layout` is
     * partial (i.e., not full) or if some bricks are not dimensionally
     * consistent with it (different number of length or batch axes).
     */
    void finalize_bricks_for(const data_layout_t& full_range_layout);

    /**
     * @brief Verify that the field is valid and finalized for a given
     * full-range data set that `full_range_layout` captures. An
     * `std::runtime_error` is thrown if
     * 
     * - the bricks are not sorted by increasing rank;
     * 
     * - two different bricks' layouts are found to have overlapping ranges of
     * logical indices;
     * 
     * - some brick layout is not logically contained in the given full-data layout;
     * 
     * - some `is_partial` flag is incorrectly set in brick layout's axes;
     * 
     * - the union of logical index ranges covered by all the bricks' layouts
     * does not match the logical index range of the given full-data layout.
     * 
     * @param[in] full_range_layout A full (i.e. non-partial) data layout
     * capturing the range of logical indices that this field's bricks are
     * supposed to cover.
     * 
     * @note The in-buffer strides of `full_range_layout` are irrelevant for
     * this function; only the range of logical indices along all axes matters. 
     *
     * @throw An `std::invalid_argument` is thrown if `full_range_layout` is
     * partial (i.e., not full) or if some bricks are not dimensionally
     * consistent with it (different number of length or batch axes). 
     */
    void throw_if_invalid_for(const data_layout_t& full_range_layout) const;

    inline decltype(bricks)::const_iterator first_brick_on_rank(int rank) const
    {
        return std::find_if(bricks.begin(), bricks.end(), [&rank](const auto& brick) {
            return brick.location.comm_rank == rank;
        });
    }

    bool has_dimensionally_consistent_bricks_for(const data_layout_t& full_range_layout) const;
};

struct rocfft_plan_description_t
{
    rocfft_array_type inArrayType  = rocfft_array_type_unset;
    rocfft_array_type outArrayType = rocfft_array_type_unset;

    data_layout_t input_layout;
    data_layout_t output_layout;

    std::array<size_t, 2> inOffset  = {0, 0};
    std::array<size_t, 2> outOffset = {0, 0};

    std::vector<rocfft_field_t> inFields;
    std::vector<rocfft_field_t> outFields;

    // Multi-process communicator info:
    rocfft_comm_type comm_type = rocfft_comm_none;
#ifdef ROCFFT_MPI_ENABLE
    MPI_Comm_wrapper_t mpi_comm;
#endif

    LoadOps  loadOps;
    StoreOps storeOps;

    rocfft_plan_description_t()  = default;
    ~rocfft_plan_description_t() = default;

    /**
     * @return The number of length dimensions.
     */
    size_t rank() const;
    /**
     * @return The batch size.
     */
    size_t batch() const;

    // Get the local communication rank
    int get_local_comm_rank() const;
    // Get number of ranks in the local communicator
    int get_local_comm_size() const;
    // returns the current rocfft_location_t (process rank + current device ID)
    // seen by this object
    rocfft_location_t get_current_location() const;

    /**
     * @brief Finalize the description by
     * 
     * - assigning default values for the description members that haven't been
     * explicitly set yet;
     * 
     * - finalizing bricks for all member fields (see `rocfft_field_t::finalize_bricks_for`);
     * 
     * - removing trivial unit-span axes from all layouts (including bricks' if any)
     * and sorting length axes by increasing strides if that can be done consistently
     * across all of them;
     * 
     * @param[in] dft_type user-provided type of transform for the owning plan.
     * @param[in] placement user-provided placement of transform results for the owning plan.
     * @param[in] user_lengths user-provided lengths of the transform for the owning plan.
     * @param[in] len_rank user-provided number of length dimensions for the owning plan.
     * @param[in] number_of_transforms user-provided batch size for the owning plan.
     */
    void finalize_for(rocfft_transform_type   dft_type,
                      rocfft_result_placement placement,
                      const size_t*           user_lengths,
                      const size_t            len_rank,
                      const size_t            number_of_transforms);

    /**
     * @brief Throw an `std::runtime_error` or a `rocfft_status` value (i.e., to be
     * escalated back to the end user as is), if the description is found invalid
     * or inconsistent for the intended plan operations. Series of verifications:
     * 
     * - No planar data type if not consistent with single-device operations on the
     *   current location;
     * 
     * - Validation of input and output array types (rocfft_status_invalid_array_type thrown);
     * 
     * - No planar hermitian type for in-place real transforms (rocfft_status_invalid_array_type thrown);
     * 
     * - Consistency for input and output data layouts;
     * 
     * - Input and output fields must be set for multi-process usage;
     * 
     * - Validity of input and output fields, if any (see `rocfft_field_t::throw_if_invalid_for`);
     * 
     * - Matching locations for all buffers expected to be used as input and output
     *   (if intended in-place);
     * 
     * - Actual in-place data layout requirements if consistent with single-device
     *   operations on the current location (if intended in-place);
     * 
     * @param[in] dft_type intended type of transform for the owning plan.
     * @param[in] placement intended placement for the owning plan.
     */
    void throw_if_inconsistent_or_invalid_for(rocfft_transform_type   dft_type,
                                              rocfft_result_placement placement) const;

    /**
     * @brief Verify if the description has undistributed input/output data (for all
     * possible fields) and, if so, if all said data shares the same location.
     * 
     * @tparam io input (resp. output) data sets are considered for specialization
     * value `io_data_label::INPUT` (resp. `io_data_label::OUTPUT`)
     * @return An `std::optional<rocfft_location_t>` object which has a value
     * set iff all ipnut (resp. output) data sets are undistributed and shares
     * the (returned) location. 
     */
    template <io_data_label io>
    std::optional<rocfft_location_t> expected_undistributed_location_for() const;

    /**
     * @return `true` if the description is consistent with single-device
     * operations on the current location.
     */
    bool has_undistributed_io_on_current_location() const;

    /**
     * @brief Read accessor implementation helper for the input (resp. output)
     * data layout that must take precedence. If a lone input (resp. output)
     * field with a lone brick is used, this returns a (const) reference to that
     * brick's layout (regardless of the brick's location). Otherwise, a (const)
     * reference to `input_layout` (resp. `output_lauyout`) is returned.
     * 
     * @tparam io input (resp. output) data sets are considered for specialization
     * value `io_data_label::INPUT` (resp. `io_data_label::OUTPUT`)
     * 
     * @throw An `std::logic_error` is thrown if this description involves more
     * than one brick.
     */
    template <io_data_label io>
    const data_layout_t& undistributed_layout() const;

    // Count the number of pointers required for either input or output
    // - planar data requires two pointers, real + complex require one.
    // But if fields are declared then the number of pointers is the
    // number of bricks in the fields.
    static size_t count_pointers(const std::vector<rocfft_field_t>& fields,
                                 rocfft_array_type                  arrayType,
                                 int                                comm_rank)
    {
        if(fields.empty())
            return array_type_is_planar(arrayType) ? 2 : 1;
        size_t fieldPtrs = 0;
        for(auto& f : fields)
        {
            fieldPtrs += std::count_if(
                f.bricks.begin(), f.bricks.end(), [comm_rank](const rocfft_brick_t& b) {
                    return b.location.comm_rank == comm_rank;
                });
        }
        return fieldPtrs;
    }

    // returns true if a field has bricks such that any rank has
    // bricks on more than one device
    static bool multiple_devices_in_rank(const rocfft_field_t& field);

private:
    template <io_data_label io>
    bool bricks_are_dimensionally_consistent() const;
};

struct rocfft_plan_t
{
    rocfft_result_placement placement     = rocfft_placement_inplace;
    rocfft_transform_type   transformType = rocfft_transform_type_complex_forward;
    rocfft_precision        precision     = rocfft_precision_single;

    rocfft_plan_description_t desc;

    rocfft_plan_t() = default;

    // Add a multi-plan item for execution.  Returns the index of the
    // new item in the overall multi-GPU plan.  Also provide a
    // vector of indexes of other items that must complete before this
    // item can run.
    size_t AddMultiPlanItem(std::unique_ptr<MultiPlanItem>&& item,
                            const std::vector<size_t>&       antecedents);

    // Add a new antecedent for an existing item index
    void AddAntecedent(size_t itemIdx, size_t antecedentIdx);

    // Execute the multi-GPU plan.
    void Execute(void* in_buffer[], void* out_buffer[], rocfft_execution_info info);

    size_t WorkBufBytes() const;

    // Insert core execPlan into multi-item plan, surrounding it with
    // sufficient items to gather/scatter to/from a single device if
    // the plan needs it.  Gathering all the data to a single device is
    // suboptimal but is a first step towards proper multi-device
    // logic.
    void GatherScatterSingleDevicePlan(std::unique_ptr<ExecPlan>&& execPlan);

    // Construct an optimized multi-device plan for the FFT
    // parameters in *this.  Returns false if:
    // - multiple devices are not requested for this FFT, or
    // - we have no particular optimization for this FFT and we'll need
    //   to fall back to a single-device plan
    bool BuildOptMultiDevicePlan();

    // check log level, log the topologically sorted plan if plan
    // logging is enabled
    void LogSortedPlan(const std::vector<size_t>& sortedIdx) const;

    // log field layout at plan level
    static void LogFields(const char* description, const std::vector<rocfft_field_t>& fields);

    // During plan creation, InternalTempBuffer remembers how much
    // space will be needed but doesn't allocate.  Allocate the buffers
    // after the space requirements are finalized.
    void AllocateInternalTempBuffers();

private:
    // Multi-node or multi-GPU plan is built up from a vector of plan
    // items.  Items can launch kernels on a device, or move
    // data between devices.
    std::vector<std::unique_ptr<MultiPlanItem>> multiPlan;

    // Communicate bricks on all ranks to all other ranks
    rocfft_status allgather_brick_params_mpi(rocfft_plan& plan);

    // Adjacency list describing dependencies between multiPlan items.
    // Size of this vector == multiPlan.size().
    //
    // The size_t's at multiPlanAntecedents[i] are the indexes in
    // multiPlan that need to complete before multiPlan[i] can run
    // (i.e. its antecedents).
    std::vector<std::vector<size_t>> multiPlanAntecedents;

    // Return a stack of multiPlan indexes that are in topological
    // order.  Traverse this vector in reverse order to follow the
    // sorting.
    std::vector<size_t> MultiPlanTopologicalSort() const;

    // Recursive utility function to do depth-first search.  tracks
    // visited indexes as it goes along.
    void TopologicalSortDFS(size_t               idx,
                            std::vector<bool>&   visited,
                            std::vector<size_t>& sorted) const;

    // Temp buffers allocated during plan creation for multi-device
    // plans are remembered here.  Mapped per-location.  Individual
    // plan items can have void*'s that point to these buffers.
    std::multimap<rocfft_location_t, std::shared_ptr<InternalTempBuffer>> tempBuffers;

    /**
     * @brief Create the plan items required to gather the input data buffer(s) of a
     * multi-device transform into the input buffer of the (single-device) execution
     * plan (abiding by that execution plan's input data layout).
     * 
     * @param[in] execution_plan Single-device execution plan.
     * @param[in] input_bricks bricks describing the data (layout and location) in
     * all execution-time input buffers of the multi-device transform.
     * @param[in] antecedents indices of the plan items that must complete before
     * the gather steps may be initiated.
     * @return An `std::vector<size_t>` of indices of the created plan items.
     * When these items complete, the execution plan's input buffer is set for
     * computating the desired transform.
     */
    std::vector<size_t> CreateInputGatheringItems(const ExecPlan&                    execution_plan,
                                                  const std::vector<rocfft_brick_t>& input_bricks,
                                                  const std::vector<size_t>& antecedents = {});

    /**
     * @brief Create the plan items required to scatter the output buffer of the
     * (single-device) execution plan (observing that plan's output data layout)
     * into the output data buffer(s) of a multi-device transform.
     * 
     * @param[in] execution_plan single-device execution plan.
     * @param[in] output_bricks bricks describing the data (layout and location) in
     * all execution-time output buffers of the multi-device transform.
     * @param[in] antecedents indices of the plan items that must complete before
     * the scatter steps may be initiated. 
     * @return An `std::vector<size_t>` of indices of the created plan items.
     * When these items complete, the user's output buffers are set with the
     * corresponding portions of the transform's results.
     */
    std::vector<size_t>
        CreateOutputScatteringItems(const ExecPlan&                    execution_plan,
                                    const std::vector<rocfft_brick_t>& output_bricks,
                                    const std::vector<size_t>&         antecedents);

    // Transpose the input field to the output field by adding work items
    // to the plan.  Antecedents are provided as a vector of item
    // indexes, one per brick.  Final work item per brick (that future
    // per-brick operations can depend on) is returned in outputItems.
    //
    // transposeNumber identifies this particular transpose in the
    // plan, for debugging.
    void GlobalTranspose(size_t                     elem_size,
                         const rocfft_field_t&      inField,
                         const rocfft_field_t&      outField,
                         std::vector<BufferPtr>&    input,
                         std::vector<BufferPtr>&    output,
                         const std::vector<size_t>& inputAntecedents,
                         std::vector<size_t>&       outputItems,
                         size_t                     transposeNumber);

    // default global all-to-all transpose
    void GlobalTransposeA2A(size_t                     elem_size,
                            const rocfft_field_t&      inField,
                            const rocfft_field_t&      outField,
                            std::vector<BufferPtr>&    input,
                            std::vector<BufferPtr>&    output,
                            const std::vector<size_t>& inputAntecedents,
                            std::vector<size_t>&       outputItems,
                            const std::string&         itemGroup);

    // fallback case for global transpose that uses point-to-point
    // communications, for when all-to-all isn't possible.
    void GlobalTransposeP2P(size_t                     elem_size,
                            const rocfft_field_t&      inField,
                            const rocfft_field_t&      outField,
                            std::vector<BufferPtr>&    input,
                            std::vector<BufferPtr>&    output,
                            const std::vector<size_t>& inputAntecedents,
                            std::vector<size_t>&       outputItems,
                            const std::string&         itemGroup);

    // Transform (complex-complex FFT) a whole field along specified
    // dimensions.  Input and output ptrs are provided as a vector of
    // BufferPtrs, one per brick in the field.
    //
    // Input antecedents, if provided, are the last items from the
    // previous global operation (e.g. a global transpose).  Operations
    // in this transform will depend on those antecedents that touch the
    // same buffers.
    //
    // Work items are added to the plan.  Final work item per brick (that
    // future per-brick operations can depend on) is returned in
    // outputItems.
    void C2CField(const rocfft_field_t&          field,
                  const std::vector<size_t>&     fftDims,
                  std::vector<BufferPtr>&        input,
                  std::vector<BufferPtr>&        output,
                  const std::optional<LoadOps>&  loadOps,
                  const std::optional<StoreOps>& storeOps,
                  const std::vector<size_t>&     inputAntecedents,
                  std::vector<size_t>&           outputItems);
};

bool PlanPowX(ExecPlan& execPlan);
bool GetTuningKernelInfo(ExecPlan& execPlan);
void RuntimeCompilePlan(ExecPlan& execPlan);

#endif // PLAN_H
