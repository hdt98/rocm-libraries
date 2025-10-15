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

#include "agent.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#define ROCPROFILER_CALL(result, msg)                                               \
    if(auto ec = (result); ec != ROCPROFILER_STATUS_SUCCESS)                        \
    {                                                                               \
        std::cerr << "rocprofiler-sdk error: " << rocprofiler_get_status_string(ec) \
                  << " :: " << msg << std::endl;                                    \
        abort();                                                                    \
    }

namespace
{
    rocprofiler_thread_trace_decoder_id_t decoder{};
    rocprofiler_context_id_t              client_ctx{};

    // Use types from the header namespace
    rocroller_profiler::InstructionLatencyMap                       instruction_latencies;
    rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate address_table;
    bool                                                            instructions_resolved = false;

    // Callback for code object loading
    void codeobj_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t*,
                          void*)
    {
        if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
           || record.operation != ROCPROFILER_CODE_OBJECT_LOAD
           || record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
            return;

        auto* data
            = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

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

    // Callback for shader data processing
    void shader_data_callback(
        rocprofiler_agent_id_t, int64_t, void* se_data, size_t data_size, rocprofiler_user_data_t)
    {
        auto parse = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                        void*                                          events,
                        uint64_t                                       num_events,
                        void*) {
            if(record_type_id != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE)
                return;

            for(size_t w = 0; w < num_events; w++)
            {
                auto* wave = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(events);
                for(size_t i = 0; i < wave->instructions_size; i++)
                {
                    auto& inst = wave->instructions_array[i];
                    auto& data = instruction_latencies[inst.pc];
                    data.latency += inst.duration;
                    data.hitcount += 1;
                }
            }
        };

        rocprofiler_trace_decode(decoder, parse, se_data, data_size, nullptr);
    }

    // Dispatch callback - always enable tracing
    rocprofiler_thread_trace_control_flags_t dispatch_callback(rocprofiler_agent_id_t,
                                                               rocprofiler_queue_id_t,
                                                               rocprofiler_async_correlation_id_t,
                                                               rocprofiler_kernel_id_t,
                                                               rocprofiler_dispatch_id_t,
                                                               void*,
                                                               rocprofiler_user_data_t*)
    {
        return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
    }

    // Query and configure GPU agents
    rocprofiler_status_t query_agents(rocprofiler_agent_version_t,
                                      const void** agents,
                                      size_t       num_agents,
                                      void*        user_data)
    {
        for(size_t idx = 0; idx < num_agents; idx++)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
            if(agent->type != ROCPROFILER_AGENT_TYPE_GPU)
                continue;

            ROCPROFILER_CALL(
                rocprofiler_configure_dispatch_thread_trace_service(client_ctx,
                                                                    agent->id,
                                                                    nullptr,
                                                                    0,
                                                                    dispatch_callback,
                                                                    shader_data_callback,
                                                                    user_data),
                "configure thread trace");
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Tool initialization
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

    // Helper function to resolve instruction names
    void resolve_instruction_names()
    {
        if(instructions_resolved)
            return;

        for(auto& [pc, data] : instruction_latencies)
        {
            auto inst = address_table.get(pc.marker_id, pc.addr);
            if(inst)
            {
                data.instruction = inst->inst;
            }
        }
        instructions_resolved = true;
    }

    // Tool finalization - output results
    void tool_fini(void*)
    {
        // Resolve instruction names
        resolve_instruction_names();

        // Output results
        const char*   OUTPUT_FILE = "thread_trace.log";
        std::ofstream file(OUTPUT_FILE);
        std::ostream& output = file.is_open() ? file : std::cout;

        if(file.is_open())
            std::cout << "Writing results to: " << OUTPUT_FILE << std::endl;

        output << "\"Instruction\", \"Total Latency (cycles)\", \"Hit Count\", \"Avg Latency "
                  "(cycles)\""
               << std::endl;
        for(const auto& [pc, data] : instruction_latencies)
        {
            output << "\"" << data.instruction << "\", " << data.latency << ", " << data.hitcount
                   << ", " << (data.hitcount ? (data.latency / data.hitcount) : 0) << std::endl;
        }

        rocprofiler_thread_trace_decoder_destroy(decoder);
    }
}

extern "C" rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if(priority > 0)
        return nullptr;

    id->name = "Simple Instruction Latency Tracer";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    return &cfg;
}

// API implementation
namespace rocroller_profiler
{
    const InstructionLatencyMap& get_instruction_latencies()
    {
        // Ensure instruction names are resolved before returning the data
        resolve_instruction_names();
        return instruction_latencies;
    }
} // namespace rocroller_profiler
