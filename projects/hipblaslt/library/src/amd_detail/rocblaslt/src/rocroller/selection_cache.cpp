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

#include "selection_cache.hpp"

void SelectionCache::addKernelSelections(const KernelType& kernelType, const ProblemSize& problemSize, SelectedKernels selections)
{
    auto existingKernelType = m_kernelSelections.find(kernelType);
    if(existingKernelType == m_kernelSelections.end())
    {
        m_kernelSelections[kernelType] = {};
    }

    m_kernelSelections[kernelType][problemSize] = selections;
}

std::optional<SelectedKernels> SelectionCache::getKernelSelections(const KernelType& kernelType, const ProblemSize& problemSize)
{
    auto existingKernelType = m_kernelSelections.find(kernelType);
    if(existingKernelType == m_kernelSelections.end())
    {
        return std::nullopt;
    }

    auto selection = existingKernelType->second.find(problemSize);

    if (selection == existingKernelType->second.end())
        return std::nullopt;
    else
        return selection->second;
}
