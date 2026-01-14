/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc.
 * ************************************************************************/
#pragma once
static inline void adjust_for_alignment(size_t& size_work)
{
    constexpr int ialign = 256;

    auto ceil = [](auto n, auto base) { return ((n - 1) / base + 1); };

    size_work = ceil(size_work, ialign) * ialign;
}

static inline void adjust_for_alignment(size_t* p_size_work)
{
    size_t size_work = *p_size_work;

    adjust_for_alignment(size_work);

    *p_size_work = size_work;
}

#ifndef IS_POINTER_BATCHED
#define IS_POINTER_BATCHED(A, T) \
    (std::is_pointer_v<std::remove_cv_t<std::remove_cv_t<std::remove_reference_t<decltype((A)[0])>>>>)
#endif

#ifndef MEM_CHECK
#define MEM_CHECK(pfree)                                        \
    {                                                           \
        bool const is_mem_ok_ = (pfree <= (pwork + size_work)); \
        if(!is_mem_ok_)                                         \
        {                                                       \
            return (rocblas_status_memory_error);               \
        }                                                       \
    }
#endif

#ifndef MEM_CHECK_THROW
#define MEM_CHECK_THROW(pfree)                                  \
    {                                                           \
        bool const is_mem_ok_ = (pfree <= (pwork + size_work)); \
        if(!is_mem_ok_)                                         \
        {                                                       \
            istat = rocblas_status_memory_error;                \
            throw(istat);                                       \
        }                                                       \
    }
#endif
