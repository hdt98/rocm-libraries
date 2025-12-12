/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "kernel_type.hpp"

struct ProblemSize
{
    size_t m;
    size_t n;
    size_t k;
    size_t batchCount;

    auto operator<=>(const ProblemSize& other) const = default;
};

template<>
struct std::hash<ProblemSize>
{
    size_t operator()(const ProblemSize& s) const noexcept
    {
        size_t mHash = std::hash<size_t>{}(s.m);
        size_t nHash = std::hash<size_t>{}(s.n);
        size_t kHash = std::hash<size_t>{}(s.k);
        size_t batchCountHash = std::hash<size_t>{}(s.batchCount);

        return mHash ^ (nHash << 1) ^ (kHash << 2) ^ (batchCountHash << 3);
    }
};

struct SelectedKernel
{
    int solutionIndex;
    size_t workspaceRequired;
};

struct SelectedKernels
{
    std::vector<SelectedKernel> selectedKernels;
    int numRequested;
};

class SelectionCache
{
    public:
    void addKernelSelections(const KernelType& kernelType, const ProblemSize& problemSize, SelectedKernels selections);
    std::optional<SelectedKernels> getKernelSelections(const KernelType& kernelType, const ProblemSize& problemSize);

    private:

    std::unordered_map<KernelType, std::unordered_map<ProblemSize, SelectedKernels>> m_kernelSelections;
};
