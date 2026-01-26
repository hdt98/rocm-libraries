/* **************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "rocblas_utility.hpp"

#define ROCSOLVER_INIT_DEVICE_WORKSPACE_IMPL(dwptr, work_items)                                     \
    do                                                                                              \
    {                                                                                               \
        /* */                                                                                       \
        /* Initialize workspace if empty */                                                         \
        /* */                                                                                       \
        if(dwptr == nullptr)                                                                        \
        {                                                                                           \
            /* Allocate device memory */                                                            \
            auto [dev_workspace_ptr, status] = rocsolver_device_workspace::Get(handle, work_items); \
                                                                                                    \
            /* Early return with `status` if this is a memory query */                              \
            if(rocblas_is_device_memory_size_query(handle))                                         \
            {                                                                                       \
                return status;                                                                      \
            }                                                                                       \
                                                                                                    \
            /* Early return with `status` in case of error */                                       \
            if(!dev_workspace_ptr || (status != rocblas_status_success))                            \
            {                                                                                       \
                return status;                                                                      \
            }                                                                                       \
                                                                                                    \
            /* Memory has been correctly allocated and is ready for use */                          \
            dwptr = dev_workspace_ptr;                                                              \
        }                                                                                           \
    } while(0);

#define ROCSOLVER_INIT_DEVICE_WORKSPACE(dwptr, ...) \
    ROCSOLVER_INIT_DEVICE_WORKSPACE_IMPL(dwptr, (__VA_ARGS__))

ROCSOLVER_BEGIN_NAMESPACE

namespace detail
{
class work_items_impl
{
public:
    work_items_impl() = default;

    work_items_impl(std::size_t M)
        : N(M)
    {
        items_sizes_ = std::vector<std::size_t>(N, static_cast<std::size_t>(0));
        items_names_ = std::vector<std::vector<std::string>>(N, std::vector<std::string>());
    }

    virtual ~work_items_impl() = default;

    work_items_impl(const work_items_impl&) = default;
    work_items_impl& operator=(const work_items_impl&) = default;

    work_items_impl(work_items_impl&&) = default;
    work_items_impl& operator=(work_items_impl&&) = default;

    std::size_t num_work_items() const noexcept
    {
        return N;
    }

    bool item(std::size_t id, std::string name, std::size_t size)
    {
        std::vector<std::string> names = {name};
        return item(id, names, size);
    }

    bool item(std::size_t id, std::vector<std::string> names, std::size_t size)
    {
        if(id >= N)
        {
            throw(std::domain_error("Error at work_items_impl::item(): id >= N"));
            return false;
        }

        if((size == 0) && (names.size() == 1) && (names.front().empty()))
        {
            return false;
        }

        for(auto& name : names)
        {
            map_names_to_ids_[name] = id;
        }
        items_sizes_[id] = size;
        items_names_[id] = names;

        return true;
    }

    /* bool merge_item(std::size_t id, std::vector<std::string> names, std::size_t size) */
    /* { */
    /*     if(id >= N) */
    /*     { */
    /*         throw(std::domain_error("Error at rocsolver_work_items_n::merge_item(): id >= N")); */
    /*         return false; */
    /*     } */

    /*     for(auto& name : names) */
    /*     { */
    /*         map_names_to_ids_[name] = id; */
    /*     } */
    /*     items_sizes_[id] = std::max(size, items_sizes_[id]); */
    /*     items_names_[id].insert(items_names_[id].end(), names.begin(), names.end()); */

    /*     return true; */
    /* } */

    std::size_t id(std::string item_name) const
    {
        auto item_id = map_names_to_ids_[item_name];
        return item_id;
    }

    std::size_t size(size_t item_id) const
    {
        return items_sizes_[item_id];
    }

    std::size_t size(std::string item_name) const
    {
        auto item_id = id(item_name);
        return items_sizes_[item_id];
    }

    std::vector<std::string> names(std::size_t item_id) const
    {
        return items_names_[item_id];
    }

    template <std::size_t M>
    auto tuple_of_items_sizes() const
    {
        return generate_tuple<M>([&](std::size_t i) { return items_sizes_[i]; });
    }

    ///
    /// For debugging
    ///

    //
    // Prints internal information for debugging purposes.
    //
    // \return std::string with debug information.
    //
    auto print_debug_str(std::string s) -> std::string
    {
        std::ostringstream os;
        os << s;
        return print_debug(os).str();
    }

    //
    // Prints internal information for debugging purposes.
    //
    // \param os: reference to a variable of a type that derives from
    // std::ostream, in which debug information is meant to be appended to.
    //
    // \return *reference* to input parameter `os`, for convenience.
    //
    // See `print_debug_str` for an example of usage.
    //
    template <typename K = std::ostringstream,
              typename = typename std::enable_if<std::is_base_of_v<std::ostream, K>>::type>
    [[maybe_unused]] auto print_debug(K& os) -> K&
    {
        std::size_t total{0};
        os << ">>>\n";
        /* os << ":: :: work_items_impl::print_debug()\n\n" << std::flush; */
        for(std::size_t i = 0; i < N; ++i)
        {
            os << ":: Item = " << i;
            os << "; Size  = " << items_sizes_[i];
            os << "; Names  = {";
            auto& names = items_names_[i];
            for(std::size_t j = 0; j < names.size(); ++j)
            {
                os << names[j];
                if(j < names.size() - 1)
                {
                    os << ", ";
                }
            }
            os << "}" << std::endl;

            total += items_sizes_[i];
        }
        os << ":: Total size = " << total << std::endl;
        os << "<<<\n" << std::flush;

        return os;
    }

protected:
    mutable std::size_t N{0};
    mutable std::vector<std::size_t> items_sizes_{};
    mutable std::vector<std::vector<std::string>> items_names_{};
    mutable std::unordered_map<std::string, std::size_t> map_names_to_ids_{};
    mutable std::string tag_start = "^";
    mutable std::string tag_end = "$";

    bool tag_prefix_str(const std::string& tag, const std::string& str) const
    {
        std::string tag_ = tag_start + tag + tag_end;
        auto [iter_tag_, _] = std::mismatch(tag_.begin(), tag_.end(), str.begin());
        if(iter_tag_ == tag_.end())
        {
            return true;
        }

        return false;
    }

    bool unique_names(const std::vector<std::string>& names)
    {
        return true;
    }

    template <typename Lambda, std::size_t... Indices>
    auto generate_tuple_impl(Lambda l, std::integer_sequence<std::size_t, Indices...>) const
    {
        return std::make_tuple(l(Indices)...);
    }

    template <std::size_t M, typename Lambda>
    auto generate_tuple(Lambda l) const
    {
        return generate_tuple_impl(l, std::make_integer_sequence<std::size_t, M>{});
    }
};
} /// namespace detail

using work_items_list_t = std::vector<std::pair<std::string, std::size_t>>;

template <std::size_t N>
class rocsolver_work_items_n
{
public:
    using work_items_impl = detail::work_items_impl;

    rocsolver_work_items_n()
        : impl_(N)
    {
    }

    ~rocsolver_work_items_n() = default;

    rocsolver_work_items_n(const rocsolver_work_items_n&) = default;
    rocsolver_work_items_n& operator=(const rocsolver_work_items_n&) = default;

    rocsolver_work_items_n(rocsolver_work_items_n&&) = default;
    rocsolver_work_items_n& operator=(rocsolver_work_items_n&&) = default;

    constexpr std::size_t num_work_items() const noexcept
    {
        return N;
    }

    bool item(std::size_t id, std::string name, std::size_t size)
    {
        return impl_.item(id, name, size);
    }

    bool item(std::size_t id, std::vector<std::string> names, std::size_t size)
    {
        return impl_.item(id, names, size);
    }

    std::size_t id(std::string item_name) const
    {
        return impl_.id(item_name);
    }

    std::size_t size(size_t item_id) const
    {
        return impl_.size(item_id);
    }

    std::size_t size(std::string item_name) const
    {
        return impl_.size(item_name);
    }

    std::vector<std::string> names(std::size_t item_id) const
    {
        return impl_.names(item_id);
    }

    work_items_impl& impl() const
    {
        return impl_;
    }

private:
    mutable work_items_impl impl_{N};
};

template <std::size_t N>
auto create_work_items_sorted_by_size(const work_items_list_t& list)
{
    rocsolver_work_items_n<N> w_out;
    if(list.size() > N)
    {
        throw(std::domain_error("Error at create_work_items_sorted_by_size(): list.size() > N"));
        return w_out;
    }

    std::multimap<std::size_t, std::string, std::greater<std::size_t>> mmap;
    for(auto& [name, size] : list)
    {
        mmap.insert(std::pair{size, name});
    }

    std::size_t id{0};
    for(auto iter = mmap.begin(); iter != mmap.end(); ++iter)
    {
        auto& [size, name] = *iter;
        w_out.item(id, name, size);
        ++id;
    }

    return w_out;
}

template <std::size_t N>
auto create_work_items(const work_items_list_t& list)
{ 
    return create_work_items_sorted_by_size<N>(list);

    rocsolver_work_items_n<N> w_out;
    if(list.size() > N)
    {
        throw(std::domain_error("Error at create_work_items(): list.size() > N"));
        return w_out;
    }

    std::size_t id{0};
    for(auto& [name, size] : list)
    {
        w_out.item(id, name, size);
        ++id;
    }

    return w_out;
}

template <std::size_t N1, std::size_t N2>
auto merge(const rocsolver_work_items_n<N1>& w1, const rocsolver_work_items_n<N2>& w2)
{
    constexpr std::size_t N = std::max({N1, N2});
    rocsolver_work_items_n<N> w_out;

    auto merge_item = [&](std::size_t id, std::vector<std::string> names_, std::size_t size_) {
        if(id >= N)
        {
            throw(std::domain_error("Error at merge::merge_item(): id >= N"));
        }

        std::size_t size = std::max(size_, w_out.size(id));
        auto names = w_out.names(id);
        names.insert(names.end(), names_.begin(), names_.end());

        w_out.item(id, names, size);
    };

    auto item_not_empty = [&](std::vector<std::string> names_, std::size_t size_) {
        for(auto& name : names_)
        {
            if(!name.empty())
            {
                return true;
            }
        }

        return size_ > 0;
    };

    for(std::size_t i = 0; i < N1; ++i)
    {
        if(item_not_empty(w1.names(i), w1.size(i)))
        {
            merge_item(i, w1.names(i), w1.size(i));
        }
    }

    for(std::size_t i = 0; i < N2; ++i)
    {
        if(item_not_empty(w2.names(i), w2.size(i)))
        {
            merge_item(i, w2.names(i), w2.size(i));
        }
    }

    return w_out;
}

template <std::size_t N1, std::size_t N2>
auto join(const rocsolver_work_items_n<N1>& w1, const rocsolver_work_items_n<N2>& w2)
{
    constexpr std::size_t N = N1 + N2;
    rocsolver_work_items_n<N> w_out;

    for(std::size_t i = 0; i < N1; ++i)
    {
        w_out.item(i, w1.names(i), w1.size(i));
    }

    for(std::size_t i = 0; i < N2; ++i)
    {
        w_out.item(i + N1, w2.names(i), w2.size(i));
    }

    return w_out;
}

template <std::size_t N1, std::size_t N2 = 0>
auto cond(bool condition,
          const rocsolver_work_items_n<N1>& w1,
          const rocsolver_work_items_n<N2>& w2 = rocsolver_work_items_n<N2>())
{
    constexpr std::size_t N = std::max({N1, N2});
    rocsolver_work_items_n<N> w_out;

    return condition ? merge(w1, w_out) : merge(w2, w_out);
}

template <std::size_t N1, std::size_t N2, std::size_t N3>
auto merge(const rocsolver_work_items_n<N1>& w1, const rocsolver_work_items_n<N2>& w2, const rocsolver_work_items_n<N3>& w3)
{
    return merge(w1, merge(w2, w3));
}

template <std::size_t N1, std::size_t N2, std::size_t N3, std::size_t N4>
auto merge(const rocsolver_work_items_n<N1>& w1, const rocsolver_work_items_n<N2>& w2, const rocsolver_work_items_n<N3>& w3, const rocsolver_work_items_n<N4>& w4)
{
    return merge(w1, merge(w2, w3, w4));
}

class rocsolver_device_workspace
{
public:
    using Handle = rocblas_handle;
    using Status = rocblas_status;
    using Ptr = std::shared_ptr<rocsolver_device_workspace>;
    using work_items_impl = detail::work_items_impl;

    template <std::size_t N>
    static auto Get(Handle handle,
                    const rocsolver_work_items_n<N>& work_items) -> std::pair<Ptr, Status>
    {
        Ptr ptr{nullptr};
        Status status = rocblas_status_success;
        auto sizes = work_items.impl().template tuple_of_items_sizes<N>();

        if(rocblas_is_device_memory_size_query(handle))
        {
            status = std::apply(
                [&](auto&&... args) {
                    return rocblas_set_optimal_device_memory_size(handle, std::size_t(args)...);
                },
                sizes);
            return std::make_pair(nullptr, status);
        }

        ptr = Ptr(new rocsolver_device_workspace(handle, work_items.impl()));
        ptr->work_ = std::make_shared<rocblas_device_malloc>(std::apply(
            [&](auto&&... args) { return rocblas_device_malloc(handle, std::size_t(args)...); },
            sizes));

        if(!ptr->work_)
        {
            status = rocblas_status_memory_error;
            return std::make_pair(nullptr, status);
        }

        return std::make_pair(std::move(ptr), status);
    }

    ~rocsolver_device_workspace() = default;

    rocsolver_device_workspace(const rocsolver_device_workspace&) = delete;
    rocsolver_device_workspace& operator=(const rocsolver_device_workspace&) = delete;

    rocsolver_device_workspace(rocsolver_device_workspace&&) = default;
    rocsolver_device_workspace& operator=(rocsolver_device_workspace&&) = default;

    std::size_t id(const std::string& item_name) const
    {
        return work_items_.id(item_name);
    }

    std::size_t size(size_t item_id) const
    {
        return work_items_.size(item_id);
    }

    std::size_t size(const std::string& item_name) const
    {
        return work_items_.size(item_name);
    }

    std::vector<std::string> names(std::size_t item_id) const
    {
        return work_items_.names(item_id);
    }

    void* work(std::size_t item_id)
    {
        return (*work_)[item_id];
    }

    void* work(const std::string& item_name)
    {
        auto item_id = id(item_name);
        return work(item_id);
    }

    ///
    /// For debugging
    ///

    //
    // Prints internal information for debugging purposes.
    //
    // \return std::string with debug information.
    //
    auto print_debug_str(std::string s = {}) -> std::string
    {
        std::ostringstream os;
        os << s;
        return work_items_.print_debug(os).str();
    }

private:
    Handle handle_{nullptr};
    work_items_impl work_items_{};
    std::shared_ptr<rocblas_device_malloc> work_{nullptr};

    rocsolver_device_workspace(Handle h, work_items_impl w)
        : handle_(h)
        , work_items_(std::move(w))
    {
    }
};

using rocsolver_device_workspace_ptr_t = rocsolver_device_workspace::Ptr;

//
// DRAFT methods
//
// Those are meant to exist on the respective roclapack and rocauxiliary `.hpp`
// files; will be moved when finalized.
//

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
auto rocsolver_geqrf_getWorkItems(rocblas_handle handle,
                                  const I m,
                                  const I n,
                                  U /* A */,
                                  const rocblas_stride shiftA,
                                  const I lda,
                                  const rocblas_stride strideA,
                                  T* /* ipiv */,
                                  const rocblas_stride strideP,
                                  const I batch_count)
{
    //
    // Get sizes using legacy `_getMemorySize` method
    //
    // Size for constants in rocblas calls
    size_t size_scalars;
    // Size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr, size_workArr;
    // Extra requirements for calling GEQR2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // Extra requirements for calling GEQR2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_geqrf_getMemorySize<BATCHED, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                              &size_Abyx_norms_trfact, &size_diag_tmptr,
                                              &size_workArr);

    //
    // Create a list of work items with previously computed sizes and return it
    //
    work_items_list_t wlist = {{"geqrf_scalars", size_scalars},
                               {"geqrf_work_workArr", size_work_workArr},
                               {"geqrf_workArr", size_workArr},
                               {"geqrf_Abyx_norms_trfact", size_Abyx_norms_trfact},
                               {"geqrf_diag_tmptr", size_diag_tmptr}};

    // Note: Number of elements in `wlist` must be known at compile time
    // if we are to use rocBLAS' methods for device memory management
    constexpr std::size_t N{5};

    auto work_items = create_work_items<N>(wlist);

    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U, typename DevWorkPtr>
rocblas_status rocsolver_geqrf_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const I batch_count,
                                        DevWorkPtr dwptr)
{
    /* // */
    /* // Initialize workspace if empty (an empty workspace means that this is a top level function call) */
    /* // */
    /* // Note: This can be abstracted in a macro */
    /* // */
    /* if(dwptr == nullptr) */
    /* { */
    /*     // Get work items, i.e., names and sizes corresponding to all memory */
    /*     // buffers that the template method will use */
    /*     auto work_items = rocsolver_geqrf_getWorkItems<BATCHED, STRIDED, T, I, U>( */
    /*         handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count); */

    /*     // Allocate device memory */
    /*     auto [dev_workspace_ptr, status] = rocsolver_device_workspace::Get(handle, work_items); */

    /*     // Early return with `status` if this is a memory query */
    /*     if(rocblas_is_device_memory_size_query(handle)) */
    /*     { */
    /*         return status; */
    /*     } */

    /*     // Early return with `status` in case of error */
    /*     if(!dev_workspace_ptr || (status != rocblas_status_success)) */
    /*     { */
    /*         return status; */
    /*     } */

    /*     // Memory has been correctly allocated and is ready for use */
    /*     dwptr = dev_workspace_ptr; */
    /* } */

    //
    // Initialize workspace if empty (an empty workspace means that this is a top level function call)
    //
    ROCSOLVER_INIT_DEVICE_WORKSPACE(
        dwptr,
        rocsolver_geqrf_getWorkItems<BATCHED, STRIDED, T, I, U>(
            handle, m, n, A, shiftA, lda, strideA, ipiv, strideP, batch_count));

    //
    // Get pointers to buffers in device workspace pointer `dwptr`
    //
    // Note: Work items names must match the names defined in the `_getWorkItems` method
    //
    T* scalars = (T*)dwptr->work("geqrf_scalars");
    void* work_workArr = dwptr->work("geqrf_work_workArr");
    T* Abyx_norms_trfact = (T*)dwptr->work("geqrf_Abyx_norms_trfact");
    T* diag_tmptr = (T*)dwptr->work("geqrf_diag_tmptr");
    T** workArr = (T**)dwptr->work("geqrf_workArr");

    //
    // Initialize memory if required
    //
    if(dwptr->size("geqrf_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    //
    // Call legacy method
    //
    return rocsolver_geqrf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      strideP, batch_count, scalars, work_workArr,
                                                      Abyx_norms_trfact, diag_tmptr, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_gelqf_getWorkItems(rocblas_handle handle,
                                  const rocblas_int m,
                                  const rocblas_int n,
                                  U /* A */,
                                  const rocblas_int shiftA,
                                  const rocblas_int lda,
                                  const rocblas_stride strideA,
                                  T* /* ipiv */,
                                  const rocblas_stride strideP,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr, size_workArr;
    // extra requirements for calling GEQL2 and to store temporary triangular factor
    size_t size_Abyx_norms_trfact;
    // extra requirements for calling GEQL2 and LARFB
    size_t size_diag_tmptr;
    rocsolver_gelqf_getMemorySize<BATCHED, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                              &size_Abyx_norms_trfact, &size_diag_tmptr,
                                              &size_workArr);

    constexpr std::size_t N{5};
    work_items_list_t wlist = {{"gelqf_scalars", size_scalars},
                               {"gelqf_work_workArr", size_work_workArr},
                               {"gelqf_workArr", size_workArr},
                               {"gelqf_Abyx_norms_trfact", size_Abyx_norms_trfact},
                               {"gelqf_diag_tmptr", size_diag_tmptr}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
rocblas_status rocsolver_gelqf_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count,
                                        DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("gelqf_scalars");
    void* work_workArr = dwptr->work("gelqf_work_workArr");
    T* Abyx_norms_trfact = (T*)dwptr->work("gelqf_Abyx_norms_trfact");
    T* diag_tmptr = (T*)dwptr->work("gelqf_diag_tmptr");
    T** workArr = (T**)dwptr->work("gelqf_workArr");

    if(dwptr->size("gelqf_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_gelqf_template<BATCHED, STRIDED>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      strideP, batch_count, scalars, work_workArr,
                                                      Abyx_norms_trfact, diag_tmptr, workArr);
}

template <typename T, typename S, typename W1, typename W2, typename W3>
auto rocsolver_bdsqr_getWorkItems(rocblas_handle handle,
                                  const rocblas_fill uplo,
                                  const rocblas_int n,
                                  const rocblas_int nv,
                                  const rocblas_int nu,
                                  const rocblas_int nc,
                                  S* /* D */,
                                  const rocblas_stride strideD,
                                  S* /* E */,
                                  const rocblas_stride strideE,
                                  W1 /* V */,
                                  const rocblas_int shiftV,
                                  const rocblas_int ldv,
                                  const rocblas_stride strideV,
                                  W2 /* U */,
                                  const rocblas_int shiftU,
                                  const rocblas_int ldu,
                                  const rocblas_stride strideU,
                                  W3 /* C */,
                                  const rocblas_int shiftC,
                                  const rocblas_int ldc,
                                  const rocblas_stride strideC,
                                  rocblas_int* info,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size of re-usable workspace
    size_t size_splits_map, size_work, size_completed;
    rocsolver_bdsqr_getMemorySize<S>(n, nv, nu, nc, batch_count, &size_splits_map, &size_work,
                                     &size_completed);

    constexpr std::size_t N{3};
    work_items_list_t wlist = {{"bdsqr_splits_map", size_splits_map},
                               {"bdsqr_work", size_work},
                               {"bdsqr_completed", size_completed}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <typename T, typename S, typename W1, typename W2, typename W3, typename DevWorkPtr>
rocblas_status rocsolver_bdsqr_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nv,
                                        const rocblas_int nu,
                                        const rocblas_int nc,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        W1 V,
                                        const rocblas_int shiftV,
                                        const rocblas_int ldv,
                                        const rocblas_stride strideV,
                                        W2 U,
                                        const rocblas_int shiftU,
                                        const rocblas_int ldu,
                                        const rocblas_stride strideU,
                                        W3 C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    void* splits_map = dwptr->work("bdsqr_splits_map");
    void* work = dwptr->work("bdsqr_splits_map");
    void* completed = dwptr->work("bdsqr_completed");

    return rocsolver_bdsqr_template<T>(handle, uplo, n, nv, nu, nc, D, strideD, E, strideE, V,
                                       shiftV, ldv, strideV, U, shiftU, ldu, strideU, C, shiftC,
                                       ldc, strideC, info, batch_count, (rocblas_int*)splits_map,
                                       (S*)work, (rocblas_int*)completed);
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
auto rocsolver_gebrd_getWorkItems(
    rocblas_handle handle,
    const rocblas_int m,
    const rocblas_int n,
    U A,
    const rocblas_int shiftA,
    const rocblas_int lda,
    const rocblas_stride strideA,
    S* D,
    const rocblas_stride strideD,
    S* E,
    const rocblas_stride strideE,
    T* tauq,
    const rocblas_stride strideQ,
    T* taup,
    const rocblas_stride strideP,
    /* T* X, */
    /*                                         const rocblas_int shiftX, */
    /*                                         const rocblas_int ldx, */
    /*                                         const rocblas_stride strideX, */
    /*                                         T* Y, */
    /*                                         const rocblas_int shiftY, */
    /*                                         const rocblas_int ldy, */
    /*                                         const rocblas_stride strideY, */
    const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_work_workArr;
    // extra requirements for calling GEDB2 and LABRD
    size_t size_Abyx_norms;
    // size for temporary resulting orthogonal matrices when calling LABRD
    size_t size_X;
    size_t size_Y;
    rocsolver_gebrd_getMemorySize<false, T>(m, n, batch_count, &size_scalars, &size_work_workArr,
                                            &size_Abyx_norms, &size_X, &size_Y);

    constexpr std::size_t N{5};
    work_items_list_t wlist = {{"gebrd_scalars", size_scalars},
                               {"gebrd_work_workArr", size_work_workArr},
                               {"gebrd_Abyx_norms", size_Abyx_norms},
                               {"gebrd_X", size_X},
                               {"gebrd_Y", size_Y}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename U, typename DevWorkPtr>
rocblas_status rocsolver_gebrd_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        T* tauq,
                                        const rocblas_stride strideQ,
                                        T* taup,
                                        const rocblas_stride strideP,
                                        /* T* X, */
                                        /* const rocblas_int shiftX, */
                                        /* const rocblas_int ldx, */
                                        /* const rocblas_stride strideX, */
                                        /* T* Y, */
                                        /* const rocblas_int shiftY, */
                                        /* const rocblas_int ldy, */
                                        /* const rocblas_stride strideY, */
                                        const rocblas_int batch_count,
                                        DevWorkPtr dwptr)
{
    ROCSOLVER_INIT_DEVICE_WORKSPACE(
        dwptr,
        rocsolver_gerbd_getWorkItems(handle, m, n, A, shiftA, lda, strideA, D, strideD, E, strideE,
                                     tauq, strideQ, taup, strideP, batch_count));

    void* scalars = dwptr->work("gebrd_scalars");
    void* work_workArr = dwptr->work("gebrd_work_workArr");
    void* Abyx_norms = dwptr->work("gebrd_Abyx_norms");
    void* X = dwptr->work("gebrd_X");
    void* Y = dwptr->work("gebrd_Y");

    if(dwptr->size("gebrd_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    rocblas_int shiftX{0}, ldx{0}, shiftY{0}, ldy{0};
    rocblas_stride strideX{0}, strideY{0};

    if(BATCHED)
    {
        // working with unshifted arrays
        shiftX = 0;
        shiftY = 0;

        // batched execution
        strideX = m * GEBRD_BLOCKSIZE;
        strideY = n * GEBRD_BLOCKSIZE;
    }
    else if(STRIDED)
    {
        // working with unshifted arrays
        shiftX = 0;
        shiftY = 0;

        // strided_batched execution
        strideX = m * GEBRD_BLOCKSIZE;
        strideY = n * GEBRD_BLOCKSIZE;
    }

    return rocsolver_gebrd_template<BATCHED, STRIDED, T>(
        handle, m, n, A, shiftA, lda, strideA, D, strideD, E, strideE, tauq, strideQ, taup, strideP,
        (T*)X, shiftX, m, strideX, (T*)Y, shiftY, n, strideY, batch_count, (T*)scalars,
        work_workArr, (T*)Abyx_norms);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_orgbr_ungbr_getWorkItems(rocblas_handle handle,
                                              const rocblas_storev storev,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U /* A */,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* /* ipiv */,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORG2R/UNG2R and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orgbr_ungbr_getMemorySize<false, T>(storev, m, n, k, batch_count, &size_scalars,
                                                  &size_work, &size_Abyx_tmptr, &size_trfact,
                                                  &size_workArr);

    constexpr std::size_t N{5};
    work_items_list_t wlist = {{"orgbr_ungbr_scalars", size_scalars},
                               {"orgbr_ungbr_workArr", size_workArr},
                               {"orgbr_ungbr_work", size_work},
                               {"orgbr_ungbr_Abyx_tmptr", size_Abyx_tmptr},
                               {"orgbr_ungbr_trfact", size_trfact}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
rocblas_status rocsolver_orgbr_ungbr_template(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("orgbr_ungbr_scalars");
    T* work = (T*)dwptr->work("orgbr_ungbr_work");
    T* Abyx_tmptr = (T*)dwptr->work("orgbr_ungbr_Abyx_tmptr");
    T* trfact = (T*)dwptr->work("orgbr_ungbr_trfact");
    T** workArr = (T**)dwptr->work("orgbr_ungbr_workArr");

    if(dwptr->size("orgbr_ungbr_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_orgbr_ungbr_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                            ipiv, strideP, batch_count, scalars,
                                                            work, Abyx_tmptr, trfact, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U, bool COMPLEX = rocblas_is_complex<T>>
auto rocsolver_ormbr_unmbr_getWorkItems(rocblas_handle handle,
                                        const rocblas_storev storev,
                                        const rocblas_side side,
                                        const rocblas_operation trans,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U /* A */,
                                        /* const rocblas_int shiftA, */
                                        const rocblas_int lda,
                                        /* const rocblas_stride strideA, */
                                        T* /* ipiv */,
                                        /* const rocblas_stride strideP, */
                                        U /* C */,
                                        /* const rocblas_int shiftC, */
                                        const rocblas_int ldc,
                                        /* const rocblas_stride strideC, */
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // requirements for calling ORMQR/UNMQR or ORMLQ/UNMLQ
    size_t size_scalars;
    size_t size_AbyxORwork, size_diagORtmptr;
    size_t size_trfact;
    size_t size_workArr;
    rocsolver_ormbr_unmbr_getMemorySize<false, T>(storev, side, m, n, k, batch_count, &size_scalars,
                                                  &size_AbyxORwork, &size_diagORtmptr, &size_trfact,
                                                  &size_workArr);

    constexpr std::size_t N{5};
    work_items_list_t wlist = {{"ormbr_unmbr_scalars", size_scalars},
                               {"ormbr_unmbr_AbyxORwork", size_AbyxORwork},
                               {"ormbr_unmbr_diagORtmptr", size_diagORtmptr},
                               {"ormbr_unmbr_trfact", size_trfact},
                               {"ormbr_unmbr_workArr", size_workArr}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename U,
          bool COMPLEX = rocblas_is_complex<T>,
          typename DevWorkPtr = rocsolver_device_workspace_ptr_t>
auto rocsolver_ormbr_unmbr_template(rocblas_handle handle,
                                    const rocblas_storev storev,
                                    const rocblas_side side,
                                    const rocblas_operation trans,
                                    const rocblas_int m,
                                    const rocblas_int n,
                                    const rocblas_int k,
                                    U A,
                                    rocblas_int shiftA,
                                    const rocblas_int lda,
                                    rocblas_stride strideA,
                                    T* ipiv,
                                    rocblas_stride strideP,
                                    U C,
                                    rocblas_int shiftC,
                                    const rocblas_int ldc,
                                    rocblas_stride strideC,
                                    rocblas_int batch_count,
                                    DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("ormbr_unmbr_scalars");
    T* AbyxORwork = (T*)dwptr->work("ormbr_unmbr_AbyxORwork");
    T* diagORtmptr = (T*)dwptr->work("ormbr_unmbr_diagORtmptr");
    T* trfact = (T*)dwptr->work("ormbr_unmbr_trfact");
    T** workArr = (T**)dwptr->work("ormbr_unmbr_workArr");

    if(dwptr->size("ormbr_unmbr_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    // working with unshifted arrays
    shiftA = 0;
    shiftC = 0;

    // normal (non-batched non-strided) execution
    strideA = 0;
    strideP = 0;
    strideC = 0;
    batch_count = 1;

    return rocsolver_ormbr_unmbr_template<BATCHED, STRIDED, T>(
        handle, storev, side, trans, m, n, k, A, shiftA, lda, strideA, ipiv, strideP, C, shiftC, ldc,
        strideC, batch_count, (T*)scalars, (T*)AbyxORwork, (T*)diagORtmptr, (T*)trfact, (T**)workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_orgqr_ungqr_getWorkItems(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORG2R/UNG2R and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orgqr_ungqr_getMemorySize<BATCHED, T>(m, n, k, batch_count, &size_scalars, &size_work,
                                                    &size_Abyx_tmptr, &size_trfact, &size_workArr);

    constexpr std::size_t N{5};
    work_items_list_t wlist = {{"orgqr_ungqr_scalars", size_scalars},
                               {"orgqr_ungqr_workArr", size_workArr},
                               {"orgqr_ungqr_work", size_work},
                               {"orgqr_ungqr_Abyx_tmptr", size_Abyx_tmptr},
                               {"orgqr_ungqr_trfact", size_trfact}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
rocblas_status rocsolver_orgqr_ungqr_template(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("orgqr_ungqr_scalars");
    T* work = (T*)dwptr->work("orgqr_ungqr_work");
    T* Abyx_tmptr = (T*)dwptr->work("orgqr_ungqr_Abyx_tmptr");
    T* trfact = (T*)dwptr->work("orgqr_ungqr_trfact");
    T** workArr = (T**)dwptr->work("orgqr_ungqr_workArr");

    if(dwptr->size("orgqr_ungqr_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_orgqr_ungqr_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                            ipiv, strideP, batch_count, scalars,
                                                            work, Abyx_tmptr, trfact, workArr);
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
auto rocsolver_orglq_unglq_getWorkItems(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        const rocblas_int k,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* ipiv,
                                        const rocblas_stride strideP,
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases)
    size_t size_workArr;
    // size of re-usable workspace
    size_t size_work;
    // extra requirements for calling ORGL2/UNGL2 and LARFB
    size_t size_Abyx_tmptr;
    // size of temporary array for triangular factor
    size_t size_trfact;
    rocsolver_orglq_unglq_getMemorySize<BATCHED, T>(m, n, k, batch_count, &size_scalars, &size_work,
                                                    &size_Abyx_tmptr, &size_trfact, &size_workArr);

    constexpr std::size_t N{5};
    work_items_list_t wlist = {{"orglq_unglq_scalars", size_scalars},
                               {"orglq_unglq_workArr", size_workArr},
                               {"orglq_unglq_work", size_work},
                               {"orglq_unglq_Abyx_tmptr", size_Abyx_tmptr},
                               {"orglq_unglq_trfact", size_trfact}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename U, typename DevWorkPtr>
rocblas_status rocsolver_orglq_unglq_template(rocblas_handle handle,
                                              const rocblas_int m,
                                              const rocblas_int n,
                                              const rocblas_int k,
                                              U A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              T* ipiv,
                                              const rocblas_stride strideP,
                                              const rocblas_int batch_count,
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("orglq_unglq_scalars");
    T* work = (T*)dwptr->work("orglq_unglq_work");
    T* Abyx_tmptr = (T*)dwptr->work("orglq_unglq_Abyx_tmptr");
    T* trfact = (T*)dwptr->work("orglq_unglq_trfact");
    T** workArr = (T**)dwptr->work("orglq_unglq_workArr");

    if(dwptr->size("orglq_unglq_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_orglq_unglq_template<BATCHED, STRIDED>(handle, m, n, k, A, shiftA, lda, strideA,
                                                            ipiv, strideP, batch_count, scalars,
                                                            work, Abyx_tmptr, trfact, workArr);
}

// CURRENTLY NOT IN USE
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
auto rocsolver_stedc_getWorkItems(rocblas_handle handle,
                                  const rocblas_evect evect,
                                  const rocblas_int n,
                                  S* /* D */,
                                  const rocblas_int shiftD,
                                  const rocblas_stride strideD,
                                  S* /* E */,
                                  const rocblas_int shiftE,
                                  const rocblas_stride strideE,
                                  U /* C */,
                                  const rocblas_int shiftC,
                                  const rocblas_int ldc,
                                  const rocblas_stride strideC,
                                  rocblas_int* /* info */,
                                  const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for lasrt stack/stedc workspace
    size_t size_work_stack;
    // size for temporary computations
    size_t size_tempvect, size_tempgemm;
    // size for pointers to workspace (batched case)
    size_t size_workArr;
    // size for vector with positions of split blocks
    size_t size_splits_map;
    // size for temporary diagonal and z vectors.
    size_t size_tmpz;
    rocsolver_stedc_getMemorySize<BATCHED, T, S>(evect, n, batch_count, &size_work_stack,
                                                 &size_tempvect, &size_tempgemm, &size_tmpz,
                                                 &size_splits_map, &size_workArr);

    constexpr std::size_t N{6}; // num of work items
    work_items_list_t wlist
        = {{"stedc_work_stack", size_work_stack}, {"stedc_tempvect", size_tempvect},
           {"stedc_tempgemm", size_tempgemm},     {"stedc_workArr", size_workArr},
           {"stedc_splits_map", size_splits_map}, {"stedc_tmpz", size_tmpz}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W>
auto rocsolver_syevd_heevd_getWorkItems(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        W A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        S* D,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_stride strideE,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspaces
    size_t size_work1;
    size_t size_work2;
    size_t size_work3;
    size_t size_tmptau_W;
    // extra space for call stedc
    size_t size_splits, size_tmpz;
    // size of array of pointers (only for batched case)
    size_t size_workArr;
    // size for temporary householder scalars
    size_t size_tau;

    rocsolver_syevd_heevd_getMemorySize<BATCHED, T, S>(
        handle, evect, uplo, n, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
        &size_tmpz, &size_splits, &size_tmptau_W, &size_tau, &size_workArr);

    constexpr std::size_t N{9}; // num of work items
    work_items_list_t wlist = {{"syevd_heevd_scalars", size_scalars},
                               {"syevd_heevd_work1", size_work1},
                               {"syevd_heevd_work2", size_work2},
                               {"syevd_heevd_work3", size_work3},
                               {"syevd_heevd_tmptau_W", size_tmptau_W},
                               {"syevd_heevd_splits", size_splits},
                               {"syevd_heevd_tmpz", size_tmpz},
                               {"syevd_heevd_workArr", size_workArr},
                               {"syevd_heevd_tau", size_tau}};

    auto work_items = create_work_items<N>(wlist);
    return work_items;
}

template <bool BATCHED, bool STRIDED, typename T, typename S, typename W, typename DevWorkPtr>
rocblas_status rocsolver_syevd_heevd_template(rocblas_handle handle,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              W A,
                                              const rocblas_int shiftA,
                                              const rocblas_int lda,
                                              const rocblas_stride strideA,
                                              S* D,
                                              const rocblas_stride strideD,
                                              S* E,
                                              const rocblas_stride strideE,
                                              rocblas_int* info,
                                              const rocblas_int batch_count,
                                              DevWorkPtr dwptr)
{
    if(dwptr == nullptr)
    {
        // This auxiliary method is not meant to be called directly
        std::abort();
    }

    T* scalars = (T*)dwptr->work("syevd_heevd_scalars");
    void* work1 = dwptr->work("syevd_heevd_work1");
    void* work2 = dwptr->work("syevd_heevd_work2");
    void* work3 = dwptr->work("syevd_heevd_work3");
    S* tmpz = (S*)dwptr->work("syevd_heevd_tmpz");
    rocblas_int* splits = (rocblas_int*)dwptr->work("syevd_heevd_splits");
    T* tmptau_W = (T*)dwptr->work("syevd_heevd_tmptau_W");
    T* tau = (T*)dwptr->work("syevd_heevd_tau");
    T** workArr = (T**)dwptr->work("syevd_heevd_workArr");

    if(dwptr->size("syevd_heevd_scalars") > 0)
    {
        init_scalars(handle, (T*)scalars);
    }

    return rocsolver_syevd_heevd_template<BATCHED, STRIDED>(
        handle, evect, uplo, n, A, shiftA, lda, strideA, D, strideD, E, strideE, info, batch_count,
        scalars, work1, work2, work3, tmpz, splits, tmptau_W, tau, workArr);
}

ROCSOLVER_END_NAMESPACE
