/* **************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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
            throw(std::domain_error("Error at work_items_impl::item(): item is empty"));
            return false;
        }

        for(auto& name : names)
        {
            if(!name.empty())
            {
                if(map_names_to_ids_.find(name) != map_names_to_ids_.end())
                {
                    throw(std::domain_error("Error at work_items_impl::item(): non unique name"));
                    return false;
                }
                map_names_to_ids_[name] = id;
            }
        }
        items_sizes_[id] = size;
        items_names_[id] = names;

        return true;
    }

    bool merge_item(std::size_t id, std::vector<std::string> names, std::size_t size)
    {
        if(id >= N)
        {
            throw(std::domain_error("Error at rocsolver_work_items_n::merge_item(): id >= N"));
            return false;
        }

        for(auto& name : names)
        {
            auto iter = map_names_to_ids_.find(name);
            if(iter == map_names_to_ids_.end())
            {
                if(!name.empty())
                {
                    map_names_to_ids_[name] = id;
                }
                items_sizes_[id] = std::max(size, items_sizes_[id]);
                items_names_[id].push_back(name);
            }
            else
            {
                auto [_, old_id] = *iter;
                items_sizes_[old_id] = std::max(size, items_sizes_[old_id]);
            }
        }

        return true;
    }

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
    auto print_debug_str(std::string s = {}) -> std::string
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
        os << std::endl;
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

    bool merge_item(std::size_t id, std::vector<std::string> names, std::size_t size)
    {
        return impl_.merge_item(id, names, size);
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

template <std::size_t N1, std::size_t N2>
auto join(const rocsolver_work_items_n<N1>& w1, const rocsolver_work_items_n<N2>& w2)
{
    constexpr std::size_t N = std::max({N1, N2});
    rocsolver_work_items_n<N> w_out;
    std::size_t i1 = 0, i2 = 0;
    [[maybe_unused]] bool success = true;

    auto item_not_empty = [&](const std::vector<std::string>& names_, std::size_t size_) {
        for(auto& name : names_)
        {
            if(!name.empty())
            {
                return true;
            }
        }

        return size_ > 0;
    };

    while(i1 + i2 < N1 + N2)
    {
        if((i1 < N1) && (i2 < N2))
        {
            if(w1.size(i1) >= w2.size(i2))
            {
                if(item_not_empty(w1.names(i1), w1.size(i1)))
                {
                    success &= w_out.merge_item(i1, w1.names(i1), w1.size(i1));
                }
                ++i1;
            }
            else
            {
                if(item_not_empty(w2.names(i2), w2.size(i2)))
                {
                    success &= w_out.merge_item(i2, w2.names(i2), w2.size(i2));
                }
                ++i2;
            }
        }
        else if(i1 < N1)
        {
            if(item_not_empty(w1.names(i1), w1.size(i1)))
            {
                success &= w_out.merge_item(i1, w1.names(i1), w1.size(i1));
            }
            ++i1;
        }
        else
        {
            if(item_not_empty(w2.names(i2), w2.size(i2)))
            {
                success &= w_out.merge_item(i2, w2.names(i2), w2.size(i2));
            }
            ++i2;
        }
    }

    return w_out;
}

template <std::size_t N1, std::size_t N2>
auto add(const rocsolver_work_items_n<N1>& w1, const rocsolver_work_items_n<N2>& w2)
{
    constexpr std::size_t N = N1 + N2;
    rocsolver_work_items_n<N> w_out;
    std::size_t i1 = 0, i2 = 0;
    [[maybe_unused]] bool success = true;

    while(i1 + i2 < N1 + N2)
    {
        if((i1 < N1) && (i2 < N2))
        {
            if(w1.size(i1) >= w2.size(i2))
            {
                success &= w_out.item(i1 + i2, w1.names(i1), w1.size(i1));
                ++i1;
            }
            else
            {
                success &= w_out.item(i1 + i2, w2.names(i2), w2.size(i2));
                ++i2;
            }
        }
        else if(i1 < N1)
        {
            success &= w_out.item(i1 + i2, w1.names(i1), w1.size(i1));
            ++i1;
        }
        else
        {
            success &= w_out.item(i1 + i2, w2.names(i2), w2.size(i2));
            ++i2;
        }
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

    return condition ? join(w1, w_out) : join(w2, w_out);
}

template <std::size_t N1, std::size_t N2>
auto operator|(const rocsolver_work_items_n<N1>& lhs, const rocsolver_work_items_n<N2>& rhs)
{
    return join(lhs, rhs);
}

template <std::size_t N1, std::size_t N2>
auto operator+(const rocsolver_work_items_n<N1>& lhs, const rocsolver_work_items_n<N2>& rhs)
{
    return add(lhs, rhs);
}
} /// namespace detail

class rocsolver_device_workspace
{
public:
    using Handle = rocblas_handle;
    using Status = rocblas_status;
    using Ptr = std::shared_ptr<rocsolver_device_workspace>;
    using work_items_impl = detail::work_items_impl;

    template <std::size_t N>
    static auto Size(Handle handle, const detail::rocsolver_work_items_n<N>& work_items)
        -> std::pair<std::size_t, Status>
    {
        std::size_t size = 0;
        Status status = rocblas_status_success;
        auto sizes = work_items.impl().template tuple_of_items_sizes<N>();

        rocblas_start_device_memory_size_query((Handle)handle);
        status = std::apply(
            [&](auto&&... args) {
                return rocblas_set_optimal_device_memory_size(handle, std::size_t(args)...);
            },
            sizes);
        rocblas_stop_device_memory_size_query((Handle)handle, &size);

        return std::make_pair(size, status);
    }

    template <std::size_t N>
    static auto Get(Handle handle,
                    const detail::rocsolver_work_items_n<N>& work_items) -> std::pair<Ptr, Status>
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
        if(!ptr)
        {
            status = rocblas_status_memory_error;
            return std::make_pair(nullptr, status);
        }

        rocblas_start_device_memory_size_query((Handle)handle);
        auto _ = std::apply(
            [&](auto&&... args) {
                return rocblas_set_optimal_device_memory_size(handle, std::size_t(args)...);
            },
            sizes);
        rocblas_stop_device_memory_size_query((Handle)handle, &(ptr->work_size_));

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

    static auto Set(Handle handle, void* work, std::size_t lwork) -> Status
    {
        if(!work || !lwork)
        {
            return rocblas_status_memory_error;
        }

        return rocblas_set_workspace((Handle)handle, work, lwork);
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

    std::size_t size(const std::string& item_name) const
    {
        return work_items_.size(item_name);
    }

    void* work(const std::string& item_name)
    {
        auto item_id = id(item_name);
        return work(item_id);
    }

    Status set_work(const std::string& item_name, std::int32_t value = 0)
    {
        auto item_id = id(item_name);
        return set_work(item_id, value);
    }

    Status set_workspace(std::int32_t value = 0)
    {
        HIP_CHECK(hipMemset((*work_)[0], value, work_size_));

        return rocblas_status_success;
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

protected:
    std::size_t size(size_t item_id) const
    {
        return work_items_.size(item_id);
    }

    std::vector<std::string> names(std::size_t item_id) const
    {
        return work_items_.names(item_id);
    }

    void* work(std::size_t item_id)
    {
        return (*work_)[item_id];
    }

    Status set_work(std::size_t item_id, std::int32_t value)
    {
        void* ptr = work(item_id);
        std::size_t num_bytes = size(item_id);
        HIP_CHECK(hipMemset(ptr, value, num_bytes));

        return rocblas_status_success;
    }

private:
    Handle handle_{nullptr};
    work_items_impl work_items_{};
    std::shared_ptr<rocblas_device_malloc> work_{nullptr};
    std::size_t work_size_{};

    rocsolver_device_workspace(Handle h, work_items_impl w)
        : handle_(h)
        , work_items_(std::move(w))
    {
    }
};

using rocsolver_device_workspace_ptr_t = rocsolver_device_workspace::Ptr;

inline auto create_work_item(const std::pair<std::string, std::size_t>& pair)
{
    detail::rocsolver_work_items_n<1> w_out;
    auto [name, size] = pair;
    w_out.item(0, name, size);

    return w_out;
}

inline auto empty_work_item()
{
    return detail::rocsolver_work_items_n<0>();
}

ROCSOLVER_END_NAMESPACE
