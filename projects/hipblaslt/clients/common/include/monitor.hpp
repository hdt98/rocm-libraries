#pragma once

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <rocm_smi/rocm_smi.h>
#include <hip/hip_runtime.h>

inline void CHECK_ERR(hipError_t status)
{
    if(status != hipSuccess)
    {
        std::cerr << "HIP error: " << hipGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void CHECK_ERR(rsmi_status_t status)
{
    if(status != RSMI_STATUS_SUCCESS)
    {
        const char* err_name = nullptr;
        rsmi_status_string(status, &err_name);
        std::cerr << "ROCm SMI error: status=" << static_cast<int>(status)
                  << " (" << (err_name ? err_name : "unknown") << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct SMI_Monitor
{
    std::chrono::milliseconds period{1};
    std::atomic<bool> running{true};
    std::thread monitor_thread;

    SMI_Monitor() { CHECK_ERR(rsmi_init(0)); }

    ~SMI_Monitor()
    {
        if(monitor_thread.joinable())
            monitor_thread.join();
        CHECK_ERR(rsmi_shut_down());
    }

    uint32_t _get_smi_index(int hip_dev_id)
    {
        char pci_bus_id[64]{};
        CHECK_ERR(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), hip_dev_id));

        unsigned domain = 0, bus = 0, dev = 0, func = 0;
        if(std::sscanf(pci_bus_id, "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4)
            return UINT32_MAX;

        const uint64_t hip_pci_id =
            ((static_cast<uint64_t>(domain) & 0xffffffff) << 32)
            | ((static_cast<uint64_t>(bus) & 0xff) << 8)
            | ((static_cast<uint64_t>(dev) & 0x1f) << 3);

        uint32_t smi_count = 0;
        CHECK_ERR(rsmi_num_monitor_devices(&smi_count));

        for(uint32_t smi_index = 0; smi_index < smi_count; ++smi_index)
        {
            uint64_t rsmi_pci_id = 0;
            CHECK_ERR(rsmi_dev_pci_id_get(smi_index, &rsmi_pci_id));
            if(rsmi_pci_id == hip_pci_id)
                return smi_index;
        }

        return UINT32_MAX;
    }

    static void _monitor_loop(std::atomic<bool>& running,
                              uint32_t smi_index,
                              std::chrono::milliseconds period)
    {        
        const double MhzToMHz = 1;
        uint16_t xcd_count;
        if(rsmi_dev_metrics_xcd_counter_get(smi_index, &xcd_count) != RSMI_STATUS_SUCCESS)
            xcd_count = 1;

        while(running.load(std::memory_order_acquire))
        {
            const auto now = std::chrono::system_clock::now();
            const auto ms_since_epoch =
                std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch())
                    .count();

            rsmi_gpu_metrics_t metrics{};
            const bool metrics_ok =
                rsmi_dev_gpu_metrics_info_get(smi_index, &metrics) == RSMI_STATUS_SUCCESS;

            double power_w = -1.0;
            double mem_mhz = -1.0;
            double avg_clk = 0.0;

            if(metrics_ok)
            {
                if(metrics.current_socket_power)
                    power_w = static_cast<double>(metrics.current_socket_power);
                if(metrics.current_uclk)
                    mem_mhz = static_cast<double>(metrics.current_uclk);
            }

            uint32_t gfx_busy = 0;
            const bool usage_ok =
                rsmi_dev_busy_percent_get(smi_index, &gfx_busy) == RSMI_STATUS_SUCCESS;

            std::cout << "[monitor " << ms_since_epoch << " ms] power=" << power_w
                      << " W, gfx_clk=";

            for(size_t i = 0; i < xcd_count; ++i){
                double xcd_clk = metrics.current_gfxclks[i] * MhzToMHz;
                avg_clk += xcd_clk;
                std::cout << "|" << metrics.current_gfxclks[i] * MhzToMHz;
            }
            std::cout << "|->" << uint32_t(avg_clk / xcd_count);

            std::cout << " MHz, mem_clk=" << mem_mhz << " MHz";

            if(usage_ok)
                std::cout << ", gfx%=" << gfx_busy;

            std::cout << std::endl;
            std::this_thread::sleep_for(period);
        }
    }

    void Start(int hip_dev_id)
    {
        auto smi_index = _get_smi_index(hip_dev_id);
        if(smi_index == UINT32_MAX)
        {
            std::cerr << "ROCm SMI error: unable to map HIP device to SMI index." << std::endl;
            return;
        }
        monitor_thread = std::thread(_monitor_loop, std::ref(running), smi_index, period);
    }

    void Stop()
    {
        running.store(false, std::memory_order_release);
        if(monitor_thread.joinable())
            monitor_thread.join();
    }
};