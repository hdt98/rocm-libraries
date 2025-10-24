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
#include <rocRoller/Utilities/Logging.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
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

// Disable if want to use rocprofv3 from CLI
// Otherwise this agent can be stuck waiting for a dispatch that rocprofv3 already consumed
constexpr bool ENABLE_AGENT = true;

namespace rocRoller
{
    namespace
    {
        struct TraceDecodeCallbackUserData
        {
            rocprofiler_dispatch_id_t        dispatch_id;
            bool                             ok;
            profiler::InstructionLatencyMap* instruction_map;
        };

        rocprofiler_thread_trace_decoder_id_t decoder{};
        rocprofiler_context_id_t              client_ctx{};

        std::map<rocprofiler_dispatch_id_t, std::vector<uint8_t>>
            dispatch_instruction_latencies; // Currently only ever increases in size
        rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate address_table;

        // TODO: investiate which locks are really needed
        std::mutex everything_mutex;

        // To wait for expected number of dispatches
        std::condition_variable dispatch_cv;
        std::mutex              dispatch_count_mutex;
        std::atomic<uint64_t>   requested_dispatch_id{0};

        void codeobj_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*,
                              void*)
        {
            std::lock_guard<std::mutex> att_shader_data_lock(everything_mutex);

            if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT
               || record.operation != ROCPROFILER_CODE_OBJECT_LOAD
               || record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
                return;

            auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(
                record.payload);

            Log::info(
                "codeobj_callback: code_object_id {}, requested_dispatch_id {}, storage_type {}",
                data->code_object_id,
                requested_dispatch_id.load(),
                static_cast<std::underlying_type_t<rocprofiler_code_object_storage_type_t>>(
                    data->storage_type));

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

        void trace_decode_callback(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                                   void*                                          events,
                                   uint64_t                                       num_events,
                                   void*                                          userdata_raw)
        {
            /*
            Don't use AssertFatal macro, as that is caught as a rocprofiler error and masks the real issue.

            Based on https://github.com/ROCm/rocm-systems/blob/e9dac39102606ac6f7ab2778f74745e27841fb6c/projects/rocprofiler-sdk/samples/thread_trace/agent.cpp#L199
            
            Handle other record types based on recommendations in https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/thread_trace.html#trace-decoder-info-events
            */

            TraceDecodeCallbackUserData* userdata
                = static_cast<TraceDecodeCallbackUserData*>(userdata_raw);

            if(!userdata->ok)
                return;

            switch(record_type_id)
            {
            case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
                for(size_t w = 0; w < num_events; w++)
                {
                    auto* wave = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(events);
                    for(size_t i = 0; i < wave->instructions_size; i++)
                    {
                        auto& inst = wave->instructions_array[i];

                        Log::info("parse: dispatch_id {}, code_object_id {}, "
                                  "requested_dispatch_id {}",
                                  userdata->dispatch_id,
                                  inst.pc.code_object_id,
                                  requested_dispatch_id.load());

                        auto& data = (*userdata->instruction_map)[inst.pc];
                        data.totalLatency += inst.duration;
                        data.hitcount += 1;
                    }
                }
                break;

            // Handle if interested in data in `rocprofiler_thread_trace_decoder_occupancy_t`
            case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:
            case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP:
                // Ok to ignore both these
                break;

            case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO:
                for(size_t i = 0; i < num_events; i++)
                {
                    auto* info = static_cast<rocprofiler_thread_trace_decoder_info_t*>(events);
                    Log::warn("parse: record info {}",
                              static_cast<
                                  std::underlying_type_t<rocprofiler_thread_trace_decoder_info_t>>(
                                  *info));
                    // Currently only seen this type of info; investigate if others are encountered
                    assert(*info == ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE
                           && "parse: received INFO record that is not STITCH_INCOMPLETE");
                }
                userdata->ok = false;
                userdata->instruction_map->clear();
                break;
            default:
                Log::critical(
                    "parse: unhandled record type {}",
                    static_cast<
                        std::underlying_type_t<rocprofiler_thread_trace_decoder_record_type_t>>(
                        record_type_id));
                assert(!"parse: unhandled record type");
                break;
            }
        }

        void shader_data_callback(rocprofiler_agent_id_t  agent,
                                  int64_t                 shader_engine_id,
                                  void*                   data,
                                  size_t                  data_size,
                                  rocprofiler_user_data_t userdata)
        {
            /*
            Based on to `att_shader_data_callback` for rocprofv3 [0].

            The thread trace sample [1] decodes directly in body of this function, however, the docs say "The rate at which thread trace generates data tends to be higher (GB/s) than the rate at which it can be processed (MB/s). Deferred processing is strongly recommended to avoid performance bottlenecks" [2].

            Therefore, deferring decoding until user requests it.

            [0]: https://github.com/ROCm/rocm-systems/blob/e9dac39102606ac6f7ab2778f74745e27841fb6c/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk-tool/tool.cpp#L1387-L1408
            [1]: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk/samples/thread_trace
            [2]: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/thread_trace.html#processing-thread-trace-data
            */

            std::lock_guard<std::mutex> att_shader_data_lock(everything_mutex);
            rocprofiler_dispatch_id_t   dispatch_id
                = static_cast<rocprofiler_dispatch_id_t>(userdata.value);

            Log::info(
                "shader_data_callback: dispatch_id {}, requested_dispatch_id {}, data_size {}",
                dispatch_id,
                requested_dispatch_id.load(),
                data_size);

            // Store raw data for deferred processing
            std::vector<uint8_t> raw_data(static_cast<uint8_t*>(data),
                                          static_cast<uint8_t*>(data) + data_size);
            dispatch_instruction_latencies[dispatch_id] = std::move(raw_data);

            dispatch_cv.notify_all();
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
            std::lock_guard<std::mutex> att_shader_data_lock(everything_mutex);
            Log::info("dispatch_callback: dispatch_id {}, requested_dispatch_id {}",
                      dispatch_id,
                      requested_dispatch_id.load());

            userdata_shader->value = dispatch_id;
            return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
        }

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

                auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
                parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, 1});
                parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0x1});
                parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, 0x1});
                parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL, 1});
                parameters.push_back(
                    {ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, 0x10000000}); // 256MB

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
        std::optional<std::vector<InstructionProfile>> waitForDispatchData(int n)
        {
            if constexpr(!ENABLE_AGENT)
                return std::nullopt;

            std::optional<std::vector<InstructionProfile>> result;
            requested_dispatch_id += n;
            Log::info("waitForDispatchData: requesting {} more dispatches, now waiting for ID {}",
                      n,
                      requested_dispatch_id.load());

            std::unique_lock<std::mutex> lock(dispatch_count_mutex);

            dispatch_cv.wait(lock, [] {
                std::lock_guard<std::mutex> att_shader_data_lock(everything_mutex);
                return dispatch_instruction_latencies.find(requested_dispatch_id.load())
                       != dispatch_instruction_latencies.end();
            });

            std::lock_guard<std::mutex> att_shader_data_lock(everything_mutex);

            auto entry = dispatch_instruction_latencies.find(requested_dispatch_id.load());
            if(entry != dispatch_instruction_latencies.end() && !entry->second.empty())
            {
                InstructionLatencyMap       decoded_instructions;
                TraceDecodeCallbackUserData callback_user_data{
                    .dispatch_id     = requested_dispatch_id.load(),
                    .ok              = true,
                    .instruction_map = &decoded_instructions};

                rocprofiler_trace_decode(decoder,
                                         trace_decode_callback,
                                         entry->second.data(),
                                         entry->second.size(),
                                         &callback_user_data);

                if(callback_user_data.ok && !decoded_instructions.empty())
                {
                    for(auto& [pc, data] : decoded_instructions)
                    {
                        auto inst = address_table.get(pc.code_object_id, pc.address);
                        if(inst != nullptr)
                            data.instruction = inst->inst;
                    }

                    result = std::vector<InstructionProfile>{};
                    result->reserve(decoded_instructions.size());
                    for(const auto& [pc, data] : decoded_instructions)
                    {
                        result->push_back(data);
                    }
                    Log::info("waitForDispatchData: retrieved {} instructions for "
                              "requested_dispatch_id {}",
                              result->size(),
                              requested_dispatch_id.load());
                }
                else
                {
                    result = std::nullopt;
                    if(decoded_instructions.empty())
                        Log::warn("waitForDispatchData: no instructions decoded for "
                                  "requested_dispatch_id {}",
                                  requested_dispatch_id.load());
                    else
                        Log::warn(
                            "waitForDispatchData: decoding error for requested_dispatch_id {}",
                            requested_dispatch_id.load());
                }
            }
            else
            {
                result = std::nullopt;
                Log::warn("waitForDispatchData: no data for requested_dispatch_id {}",
                          requested_dispatch_id.load());
            }
            return result;
        }

        std::vector<InstructionProfile> loopUntilDispatchData(std::function<void()> dispatch)
        {
            if constexpr(!ENABLE_AGENT)
                return {};

            std::optional<std::vector<InstructionProfile>> data;

            while(true)
            {
                dispatch();
                data = waitForDispatchData(1);
                if(data.has_value())
                {
                    return *data;
                }
                Log::info("loopUntilDispatchData: got no data, invoking another dispatch");
            }
        }

        uint64_t InstructionProfile::meanLatency() const
        {
            if(hitcount == 0)
                return 0;
            return totalLatency / hitcount;
        }

        std::string InstructionProfile::toString() const
        {
            return fmt::format("'{}', totalLatency: {}, "
                               "hitcount: {}, meanLatency: {}",
                               instruction,
                               totalLatency,
                               hitcount,
                               meanLatency());
        }

        std::string toString(std::vector<InstructionProfile> const& profiles)
        {
            std::string result = "";
            for(const auto& profile : profiles)
            {
                result += profile.toString() + "\n";
            }
            result += "\n";
            return result;
        }

    } // namespace profiler
} // namespace rocRoller

extern "C" rocprofiler_tool_configure_result_t*
    rocprofiler_configure(uint32_t, const char*, uint32_t priority, rocprofiler_client_id_t* id)
{
    if constexpr(!ENABLE_AGENT)
        return nullptr;

    if(priority > 0)
        return nullptr;

    id->name = "rocRoller rocprofiler: I'm a string literal, just do a text search to find me :)";

    static auto cfg
        = rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                              &rocRoller::tool_init,
                                              &rocRoller::tool_fini,
                                              nullptr};

    return &cfg;
}
