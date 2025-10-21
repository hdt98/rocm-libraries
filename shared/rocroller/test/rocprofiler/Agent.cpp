// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Agent.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <rocRoller/Utilities/Error.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#define ROCPROFILER_CALL(result, msg)                                                   \
    if(auto ec = (result); ec != ROCPROFILER_STATUS_SUCCESS)                            \
    {                                                                                   \
        rocRoller::Throw<rocRoller::FatalError>(                                        \
            "rocprofiler-sdk error: ", rocprofiler_get_status_string(ec), " :: ", msg); \
    }

/*
Based on
- thread trace decoding: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk/samples/thread_trace
- using dispatch: https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/tests/thread-trace/single_dispatch.cpp
*/

namespace rocRoller
{
    namespace
    {
        rocprofiler_thread_trace_decoder_id_t decoder{};
        rocprofiler_context_id_t              client_ctx{};

        std::map<rocprofiler_dispatch_id_t, profiler::InstructionLatencyMap>
            dispatch_instruction_latencies;
        rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate address_table;

        std::mutex dispatch_shader_mutex; // Ensure dispatch and shader callback are in sync

        void codeobj_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*,
                              void*)
        {
            if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
               || record.operation != ROCPROFILER_CODE_OBJECT_LOAD
               || record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
                return;

            auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(
                record.payload);

            if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
            {
                address_table.addDecoder(
                    data->uri, data->code_object_id, data->load_delta, data->load_size);
            }
            else
            {
                auto* memorybase = reinterpret_cast<const void*>(data->memory_base);
                if(memorybase)
                {
                    rocprofiler_thread_trace_decoder_codeobj_load(decoder,
                                                                  data->code_object_id,
                                                                  data->load_delta,
                                                                  data->load_size,
                                                                  memorybase,
                                                                  data->memory_size);

                    address_table.addDecoder(memorybase,
                                             data->memory_size,
                                             data->code_object_id,
                                             data->load_delta,
                                             data->load_size);
                }
            }
        }

        void shader_data_callback(rocprofiler_agent_id_t  agent,
                                  int64_t                 shader_engine_id,
                                  void*                   data,
                                  size_t                  data_size,
                                  rocprofiler_user_data_t userdata)
        {
            rocprofiler_dispatch_id_t dispatch_id
                = static_cast<rocprofiler_dispatch_id_t>(userdata.value);

            auto parse = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                            void*                                          events,
                            uint64_t                                       num_events,
                            void*                                          userdata) {
                rocprofiler_user_data_t userdata_casted
                    = *static_cast<rocprofiler_user_data_t*>(userdata);

                rocprofiler_dispatch_id_t dispatch_id
                    = static_cast<rocprofiler_dispatch_id_t>(userdata_casted.value);

                if(record_type_id != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE)
                    return;

                for(size_t w = 0; w < num_events; w++)
                {
                    auto* wave = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(events);
                    for(size_t i = 0; i < wave->instructions_size; i++)
                    {
                        auto& inst = wave->instructions_array[i];
                        auto& data = dispatch_instruction_latencies[dispatch_id][inst.pc];
                        data.latency += inst.duration;
                        data.hitcount += 1;
                    }
                }
            };

            rocprofiler_trace_decode(decoder, parse, data, data_size, &userdata);

            for(auto& [pc, data] : dispatch_instruction_latencies[dispatch_id])
            {
                auto inst = address_table.get(pc.code_object_id, pc.address);
                if(inst)
                {
                    data.instruction = inst->inst;
                }
                else
                {
                    data.instruction = "UNKNOWN_INSTRUCTION";
                }
            }

            dispatch_shader_mutex.unlock();
        }

        rocprofiler_thread_trace_control_flags_t
            dispatch_callback(rocprofiler_agent_id_t             agent_id,
                              rocprofiler_queue_id_t             queue_id,
                              rocprofiler_async_correlation_id_t correlation_id,
                              rocprofiler_kernel_id_t            kernel_id,
                              rocprofiler_dispatch_id_t          dispatch_id,
                              void*                              userdata_config,
                              rocprofiler_user_data_t*           userdata_shader)
        {
            dispatch_shader_mutex.lock();
            userdata_shader->value = dispatch_id;
            return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
        }

        rocprofiler_status_t query_agents(rocprofiler_agent_version_t,
                                          const void** agents,
                                          size_t       num_agents,
                                          void*        user_data)
        {
            constexpr uint64_t TARGET_CU   = 1; // CU (gfx9) or WGP (gfx10+)
            constexpr uint64_t SHADER_MASK = 0x1; // Only enable SE=0
            constexpr uint64_t BUFFER_SIZE = 0x10000000; // 256MB

            for(size_t idx = 0; idx < num_agents; idx++)
            {
                const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
                if(agent->type != ROCPROFILER_AGENT_TYPE_GPU)
                    continue;

                auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
                parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, TARGET_CU});
                parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, BUFFER_SIZE});
                parameters.push_back(
                    {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, SHADER_MASK});

                ROCPROFILER_CALL(
                    rocprofiler_configure_dispatch_thread_trace_service(client_ctx,
                                                                        agent->id,
                                                                        parameters.data(),
                                                                        parameters.size(),
                                                                        dispatch_callback,
                                                                        shader_data_callback,
                                                                        user_data),
                    "configure dispatch thread trace");
            }
            return ROCPROFILER_STATUS_SUCCESS;
        }

    } // anonymous namespace

    int tool_init(rocprofiler_client_finalize_t, void*)
    {
        ROCPROFILER_CALL(rocprofiler_thread_trace_decoder_create(&decoder, "/opt/rocm/lib"),
                         "create decoder");

        ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "create context");

        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(client_ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                           nullptr,
                                                           0,
                                                           codeobj_callback,
                                                           nullptr),
            "configure code object tracing");

        ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                            &query_agents,
                                                            sizeof(rocprofiler_agent_t),
                                                            nullptr),
                         "query agents");

        ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "start context");

        return 0;
    }

    void tool_fini(void*)
    {
        rocprofiler_thread_trace_decoder_destroy(decoder);
    }

    namespace profiler
    {
        std::vector<InstructionData> getInstructionData()
        {
            const std::lock_guard<std::mutex> lock(dispatch_shader_mutex);

            std::vector<InstructionData> result;

            // Search backwards for the last dispatch with non-zero length data
            for(auto it = dispatch_instruction_latencies.rbegin();
                it != dispatch_instruction_latencies.rend();
                ++it)
            {
                if(!it->second.empty())
                {
                    result.reserve(it->second.size());

                    for(const auto& [pc, data] : it->second)
                    {
                        result.push_back(data);
                    }

                    return result;
                }
                std::cerr << "had to skip dispatch " << it->first << " with zero data" << std::endl;
            }

            // see Troubleshooting section in
            // https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html#troubleshooting
            std::cerr << "No dispatches with data found, launch more waves" << std::endl;

            return result;
        }
    } // namespace profiler
} // namespace rocRoller

extern "C" rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0)
        return nullptr;

    id->name = "rocRoller rocprofiler";

    static auto cfg
        = rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                              &rocRoller::tool_init,
                                              &rocRoller::tool_fini,
                                              nullptr};

    return &cfg;
}
