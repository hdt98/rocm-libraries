/* ************************************************************************
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

#include "helper.hpp"

#include <iostream>

// This file includes platform specific helpers that differ between POSIX and
// Windows. The alternatives are switched at a whole-file level. Please do
// not use inline/fine-grained ifdefs.

#include <amd_comgr/amd_comgr.h>
#include <memory>

inline static void checkComgr(amd_comgr_status_t status, const char* callName)
{
    if(status != AMD_COMGR_STATUS_SUCCESS)
    {
        const char* status_string = nullptr;
        amd_comgr_status_string(status, &status_string);
        throw std::runtime_error(std::string("COMGR call failed: ") + callName
                                 + " (COMGR status: " + std::to_string(status) + ", "
                                 + (status_string ? status_string : "Unknown") + ")");
    }
}

#define CHECK_COMGR(call) checkComgr(call, #call)

template <typename Handle, typename CreateFunction, typename DestroyFunction>
auto makeUniqueHandle(CreateFunction create, DestroyFunction destroy)
{
    struct Deleter
    {
        DestroyFunction destroy;
        void            operator()(Handle* handle) const
        {
            if(handle)
            {
                destroy(*handle);
                delete handle;
            }
        }
    };
    Handle* handle = new Handle{};
    if constexpr(std::is_same_v<Handle, amd_comgr_data_t>)
    {
        if(create(AMD_COMGR_DATA_KIND_SOURCE, handle) != AMD_COMGR_STATUS_SUCCESS)
        {
            delete handle;
            throw std::runtime_error("Failed to create COMGR data");
        }
    }
    else
    {
        if(create(handle) != AMD_COMGR_STATUS_SUCCESS)
        {
            delete handle;
            throw std::runtime_error("Failed to create COMGR handle");
        }
    }
    return std::unique_ptr<Handle, Deleter>(handle, Deleter{destroy});
}

std::pair<int, std::string> runComgr(const std::string& input,
                                     const std::string& gfxStr,
                                     const char**       optionStr,
                                     const size_t       optionSize,
                                     bool               debug)
{
    try
    {

        // Create data object for input assembly code
        auto data = makeUniqueHandle<amd_comgr_data_t,
                                     decltype(&amd_comgr_create_data),
                                     decltype(&amd_comgr_release_data)>(amd_comgr_create_data,
                                                                        amd_comgr_release_data);
        if(input.size() > 0)
        {
            CHECK_COMGR(amd_comgr_set_data(*data, input.size(), input.c_str()));
        }
        CHECK_COMGR(amd_comgr_set_data_name(*data, "input.s"));

        // Create data set and add the input data
        auto dataSet = makeUniqueHandle<amd_comgr_data_set_t,
                                        decltype(&amd_comgr_create_data_set),
                                        decltype(&amd_comgr_destroy_data_set)>(
            amd_comgr_create_data_set, amd_comgr_destroy_data_set);
        auto outputDataSet = makeUniqueHandle<amd_comgr_data_set_t,
                                              decltype(&amd_comgr_create_data_set),
                                              decltype(&amd_comgr_destroy_data_set)>(
            amd_comgr_create_data_set, amd_comgr_destroy_data_set);
        CHECK_COMGR(amd_comgr_data_set_add(*dataSet, *data));

        // Create action info and set the action kind
        auto actionInfo = makeUniqueHandle<amd_comgr_action_info_t,
                                           decltype(&amd_comgr_create_action_info),
                                           decltype(&amd_comgr_destroy_action_info)>(
            amd_comgr_create_action_info, amd_comgr_destroy_action_info);
        CHECK_COMGR(amd_comgr_action_info_set_language(*actionInfo, AMD_COMGR_LANGUAGE_NONE));
        std::string isaName = "amdgcn-amd-amdhsa--" + gfxStr;
        CHECK_COMGR(amd_comgr_action_info_set_isa_name(*actionInfo, isaName.c_str()));
        CHECK_COMGR(amd_comgr_action_info_set_option_list(*actionInfo, optionStr, optionSize));

        // Execute the action
        auto status = amd_comgr_do_action(
            AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, *actionInfo, *dataSet, *outputDataSet);

        // If debug is enabled, retrieve the log
        std::string log;
        if(debug)
        {
            std::string log = "";
            if(status != AMD_COMGR_STATUS_SUCCESS)
            {
                size_t logCount = 0;
                CHECK_COMGR(amd_comgr_action_data_count(
                    *outputDataSet.get(), AMD_COMGR_DATA_KIND_LOG, &logCount));

                if(logCount > 0)
                {
                    amd_comgr_data_t logData;
                    CHECK_COMGR(amd_comgr_action_data_get_data(
                        *outputDataSet, AMD_COMGR_DATA_KIND_LOG, 0, &logData));
                    auto logDataDeleter = [](amd_comgr_data_t* data) {
                        amd_comgr_release_data(*data);
                        delete data;
                    };
                    std::unique_ptr<amd_comgr_data_t, decltype(logDataDeleter)> logDataPtr(
                        new amd_comgr_data_t(logData), logDataDeleter);
                    size_t logSize = 0;
                    CHECK_COMGR(amd_comgr_get_data(*logDataPtr, &logSize, nullptr));
                    log.resize(logSize);
                    CHECK_COMGR(amd_comgr_get_data(*logDataPtr, &logSize, log.data()));
                }
            }
        }

        return {status == AMD_COMGR_STATUS_SUCCESS ? 0 : -1, debug ? log : ""};
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return {-1, ""};
    }
    catch(...)
    {
        std::cerr << "Unknown error occurred." << std::endl;
        return {-1, ""};
    }
    return {-1, ""};
}

#ifdef _WIN32

#include <windows.h>
// windows.h must be loaded before other windows headers.
#include <dbghelp.h>

// Linker hint to ensure dbghelp.lib is linked.
#pragma comment(lib, "dbghelp.lib")

std::string demangle(const char* name)
{
    std::string result              = name;
    char        demangledName[1024] = {0};
    // UNDNAME_COMPLETE flag produces the full undecorated name.
    if(UnDecorateSymbolName(name, demangledName, sizeof(demangledName), UNDNAME_COMPLETE))
    {
        result = demangledName;
    }
    return result;
}

#else
// POSIX implementation.
#ifdef __GNUG__
#include <cxxabi.h>
#endif

std::string demangle(const char* name)
{
    std::string result    = name;
    int         status    = -1;
    char*       demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    result                = (status == 0) ? demangled : name;
    free(demangled);
    return result;
}

#endif // POSIX
