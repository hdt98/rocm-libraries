// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#ifndef GUARD_MEMORY_ECOSYSTEM_HPP
#define GUARD_MEMORY_ECOSYSTEM_HPP

#include <gtest/gtest.h>

#ifdef _WIN32
#include <dxgi1_4.h>
#endif
#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

// This addresses a Windows-specific issue on low-memory APUs. MemoryEcosystem does not do
// anything in linux yet. Linux support will be added if/when it is needed.

namespace
{
    inline float GiB(size_t bytes)
    {
        return 0.1 * static_cast<float>((bytes * 10) / 1024 / 1024 / 1024);
    }
}

struct MemoryEcosystemInfo
{
    MemoryEcosystemInfo() {}
#ifdef _WIN32
    MemoryEcosystemInfo(const DXGI_ADAPTER_DESC1& desc) :
        adapter_index(0),
        dedicated_vram(desc.DedicatedVideoMemory),
        shared_ram(desc.SharedSystemMemory),
        dedicated_ram(desc.DedicatedSystemMemory)
    {}
#endif
    MemoryEcosystemInfo(int adapter, size_t ded_vram, size_t shared, size_t ded_ram) :
        adapter_index(adapter),
        dedicated_vram(ded_vram),
        shared_ram(shared),
        dedicated_ram(ded_ram)
    {}

    std::string Description()
    {
        std::ostringstream oss;
        oss << "Adapter " << adapter_index <<
                " with " << GiB(dedicated_vram) << " GiB dedicated and " <<
                GiB(shared_ram) << " GiB shared";
        return oss.str();
    }

    int adapter_index{-1};
    size_t dedicated_vram{0};
    size_t shared_ram{0};
    size_t dedicated_ram{0};
};

struct MemoryEcosystem
{
    static float GiB(size_t bytes)
    {
        return 0.1 * static_cast<float>((bytes * 10) / 1024 / 1024 / 1024);
    }

    static MemoryEcosystemInfo GetMemoryEcosystemInfo(int adapter_index = 0)
    {
#ifndef _WIN32
        return MemoryEcosystemInfo{};
#else
        IDXGIFactory4* pFactory;
        HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&pFactory);
        if (!SUCCEEDED(hr))
        {
            MIOPEN_LOG_E("Unable to create DXGI factory. Error Code: " << hr);
            return MemoryEcosystemInfo{};
        }

        IDXGIAdapter1* pAdapter;
        if(pFactory->EnumAdapters1(adapter_index, &pAdapter) == DXGI_ERROR_NOT_FOUND)
        {
            MIOPEN_LOG_E("Unable to access DXGI for adapter index " << adapter_index);
            return MemoryEcosystemInfo{};
        }

        DXGI_ADAPTER_DESC1 desc;
        hr = pAdapter->GetDesc1(&desc);
        if(!SUCCEEDED(hr))
        {
            MIOPEN_LOG_E("Unable to retrieve details for adapter index " << adapter_index);
            pFactory->Release();
            return MemoryEcosystemInfo{};
        }

        pAdapter->Release();
        pFactory->Release();

        return MemoryEcosystemInfo{desc};
#endif
    }

    static bool AbleToAllocate(const MemoryEcosystemInfo& info, const std::vector<size_t>& vram_blocks, const size_t reserved_bytes = 0)
    {
        if(reserved_bytes > info.shared_ram)
            return false;

        const auto free_shared_vram = info.shared_ram - reserved_bytes;
        size_t used_ded = 0;
        size_t used_shared = 0;

        for(auto block : vram_blocks)
        {
            if(info.dedicated_vram >= used_ded + block)
            {
                used_ded += block;
            }
            else if(free_shared_vram >= used_shared + block)
            {
                used_shared += block;
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    static bool AbleToAllocate(const std::vector<size_t>& vram_blocks, const size_t reserved_bytes = 0)
    {
#ifndef _WIN32
        return true;
#else
        const auto info = MemoryEcosystem::GetMemoryEcosystemInfo(0);
        return AbleToAllocate(info, vram_blocks, reserved_bytes);
#endif
    }

    static bool AbleToAllocate(const MemoryEcosystemInfo& info,
        const std::vector<size_t>& vram_blocks,
        const std::vector<size_t> cpu_blocks)
    {
        auto reserved = std::accumulate(cpu_blocks.begin(), cpu_blocks.end(), 0);
        return AbleToAllocate(info, vram_blocks, reserved);
    }

    static bool AbleToAllocate(const std::vector<size_t>& vram_blocks, const std::vector<size_t> cpu_blocks)
    {
#ifndef _WIN32
        return true;
#else
        const auto info = MemoryEcosystem::GetMemoryEcosystemInfo(0);
        return AbleToAllocate(info, vram_blocks, cpu_blocks);
#endif
    }

    static void SkipTestIfUnableToAllocate(const std::vector<size_t>& gpu_bufs, const std::vector<size_t>& cpu_bufs, bool& skip_flag)
    {
        if(!AbleToAllocate(gpu_bufs, cpu_bufs))
        {
            skip_flag = true;
            GTEST_SKIP() << "Unable to allocate enough Dedicated + Shared VRAM";
        }
    }
};

#endif
