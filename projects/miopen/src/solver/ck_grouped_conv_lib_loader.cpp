// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/solver/ck_grouped_conv_lib_loader.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>

#include <miopen/filesystem.hpp>

#include <cstdlib>
#include <type_traits>

#ifdef _WIN32
// clang-format off
#include <windows.h>
// clang-format on
// <windows.h> defines LoadLibrary as a macro, which collides with our member function name.
#undef LoadLibrary
#else
#include <dlfcn.h>
#endif

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_CK_LIB_PATH)

namespace miopen {
namespace solver {

namespace {

/// Strip architecture-specific suffixes like ":sramecc+:xnack-" from a device
/// name, returning only the base GPU identifier (e.g. "gfx90a").
std::string StripDeviceSuffix(const std::string& device_name)
{
    auto pos = device_name.find(':');
    if(pos != std::string::npos)
        return device_name.substr(0, pos);
    return device_name;
}

/// Build the expected shared library filename for a given device.
std::string MakeLibraryFilename(const std::string& device_name)
{
    return std::string(miopen::library_prefix) + "MIOpenCKGroupedConv_" +
           StripDeviceSuffix(device_name) + std::string(miopen::dynamic_library_postfix);
}

/// Resolve the directory containing the MIOpen shared library.
/// On Linux uses dladdr + realpath; on Windows uses GetModuleHandleExA +
/// GetModuleFileNameA.  Canonicalizes symlinks so per-arch CK libraries
/// are found even when MIOpen is accessed through a symlink.
std::string GetMIOpenLibDir()
{
#ifdef _WIN32
    HMODULE hmod = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(miopenCreate),
                          &hmod) != 0)
    {
        char buf[MAX_PATH];
        DWORD len = GetModuleFileNameA(hmod, buf, MAX_PATH);
        if(len > 0 && len < MAX_PATH)
        {
            std::string path(buf, len);
            auto slash = path.find_last_of("\\/");
            if(slash != std::string::npos)
                return path.substr(0, slash);
        }
    }
#else
    Dl_info info;
    if(dladdr(reinterpret_cast<void*>(miopenCreate), &info) != 0)
    {
        std::unique_ptr<char, decltype(&free)> real(realpath(info.dli_fname, nullptr), free);
        std::string path(real != nullptr ? real.get() : info.dli_fname);
        auto slash = path.rfind('/');
        if(slash != std::string::npos)
            return path.substr(0, slash);
    }
#endif
    return {};
}

} // namespace

// -- Singleton infrastructure -------------------------------------------------

std::mutex& CKGroupedConvLibLoader::CacheMutex()
{
    static std::mutex mtx;
    return mtx;
}

std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>>&
CKGroupedConvLibLoader::Cache()
{
    static std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>> cache;
    return cache;
}

const CKGroupedConvLibLoader& CKGroupedConvLibLoader::Get(const std::string& device_name)
{
    const auto key = StripDeviceSuffix(device_name);
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& cache = Cache();
    auto it     = cache.find(key);
    if(it == cache.end())
    {
        // Use new + reset instead of make_unique because the constructor is private.
        std::unique_ptr<CKGroupedConvLibLoader> ptr(new CKGroupedConvLibLoader(device_name));
        it = cache.emplace(key, std::move(ptr)).first;
    }
    return *it->second;
}

// -- Construction / Destruction -----------------------------------------------

CKGroupedConvLibLoader::CKGroupedConvLibLoader(const std::string& device_name)
{
    LoadLibrary(device_name);
}

CKGroupedConvLibLoader::~CKGroupedConvLibLoader()
{
    if(lib_handle_ != nullptr)
    {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(lib_handle_));
#else
        // RTLD_NODELETE keeps the library mapped, so dlclose only decrements the
        // reference count without unmapping.  We still call it for correctness.
        dlclose(lib_handle_);
#endif
    }
}

// -- Library loading ----------------------------------------------------------

void CKGroupedConvLibLoader::LoadLibrary(const std::string& device_name)
{
    const auto filename = MakeLibraryFilename(device_name);

    // Platform-specific load and error helpers
#ifdef _WIN32
    auto try_load = [](const std::string& path) -> void* {
        return static_cast<void*>(LoadLibraryA(path.c_str()));
    };
    auto get_load_error = []() -> std::string {
        DWORD err = GetLastError();
        char* msg = nullptr;
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                           FORMAT_MESSAGE_IGNORE_INSERTS,
                       nullptr,
                       err,
                       0,
                       reinterpret_cast<LPSTR>(&msg),
                       0,
                       nullptr);
        std::string result = msg ? msg : "unknown error";
        LocalFree(msg);
        return result;
    };
#else
    constexpr int flags = RTLD_NOW | RTLD_NODELETE;
    auto try_load = [](const std::string& path) -> void* { return dlopen(path.c_str(), flags); };
    auto get_load_error = []() -> std::string {
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        const char* err = dlerror();
        return err ? err : "unknown error";
    };
#endif

    // 1. Try MIOPEN_CK_LIB_PATH environment variable
    const auto env_path = env::value(MIOPEN_CK_LIB_PATH);
    if(!env_path.empty())
    {
        auto full_path = env_path + "/" + filename;
        lib_handle_    = try_load(full_path);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2("Loaded CK grouped conv library from env path: " << full_path);
        }
    }

    // 2. Try the directory containing the MIOpen shared library
    if(lib_handle_ == nullptr)
    {
        auto lib_dir = GetMIOpenLibDir();
        if(!lib_dir.empty())
        {
            auto full_path = lib_dir + "/" + filename;
            lib_handle_    = try_load(full_path);
            if(lib_handle_ != nullptr)
            {
                MIOPEN_LOG_I2("Loaded CK grouped conv library from lib dir: " << full_path);
            }
        }
    }

    // 3. Fall back to default search path
    if(lib_handle_ == nullptr)
    {
        lib_handle_ = try_load(filename);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2("Loaded CK grouped conv library from default path: " << filename);
        }
    }

    if(lib_handle_ == nullptr)
    {
        MIOPEN_LOG_W("CK grouped conv library not found for device "
                     << StripDeviceSuffix(device_name) << ": " << get_load_error());
        return;
    }

    if(!LoadSymbols())
    {
        MIOPEN_LOG_W("Failed to resolve symbols in CK grouped conv library for device "
                     << StripDeviceSuffix(device_name));
        loaded_ = false;
        return;
    }

    // API version check
    const int lib_version = get_api_version_fn_();
    if(lib_version != CK_GROUPED_CONV_API_VERSION)
    {
        MIOPEN_LOG_W("CK grouped conv API version mismatch for device "
                     << StripDeviceSuffix(device_name) << ": expected "
                     << CK_GROUPED_CONV_API_VERSION << ", got " << lib_version);
        loaded_ = false;
        return;
    }

    loaded_ = true;
}

// -- Symbol resolution --------------------------------------------------------

void* CKGroupedConvLibLoader::ResolveRawSymbol(const char* symbol_name) const
{
    if(lib_handle_ == nullptr)
        return nullptr;
#ifdef _WIN32
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(lib_handle_), symbol_name));
#else
    dlerror(); // NOLINT(concurrency-mt-unsafe)
    return dlsym(lib_handle_, symbol_name); // NOLINT(concurrency-mt-unsafe)
#endif
}

void CKGroupedConvLibLoader::BindRequiredCommonSymbols(std::vector<std::string>& missing)
{
    auto bind_symbol = [this, &missing](auto& member, const char* symbol_name) {
        using FnPtr = std::remove_reference_t<decltype(member)>;
        member      = reinterpret_cast<FnPtr>(ResolveRawSymbol(symbol_name));
        if(member == nullptr)
            missing.emplace_back(symbol_name);
    };

    bind_symbol(get_api_version_fn_, "ckgrpconv_get_api_version");
    bind_symbol(kernel_list_size_fn_, "ckgrpconv_kernel_list_size");
    bind_symbol(kernel_list_get_fn_, "ckgrpconv_kernel_list_get");
    bind_symbol(kernel_list_free_fn_, "ckgrpconv_kernel_list_free");
    bind_symbol(solution_free_fn_, "ckgrpconv_solution_free");
}

void CKGroupedConvLibLoader::BindSolverSymbols(CKSolverType solver,
                                               const char* prefix,
                                               std::vector<std::string>& missing)
{
    auto bind_symbol = [this, &missing](auto& member, const char* symbol_name) {
        using FnPtr = std::remove_reference_t<decltype(member)>;
        member      = reinterpret_cast<FnPtr>(ResolveRawSymbol(symbol_name));
        if(member == nullptr)
            missing.emplace_back(symbol_name);
    };

    auto& fns             = solver_fns_[ToSolverIndex(solver)];
    const std::string sym = std::string("ckgrpconv_") + prefix + "_";
    bind_symbol(fns.fill_valid_kernels, (sym + "fill_valid_kernels").c_str());
    bind_symbol(fns.is_applicable, (sym + "is_applicable").c_str());
    bind_symbol(fns.is_args_supported, (sym + "is_args_supported").c_str());
    bind_symbol(fns.get_workspace_size, (sym + "get_workspace_size").c_str());
    bind_symbol(fns.get_solution, (sym + "get_solution").c_str());
}

void CKGroupedConvLibLoader::BindOptionalKernelTypeSymbols(std::vector<std::string>& missing)
{
    auto bind_symbol = [this, &missing](auto& member, const char* symbol_name) {
        using FnPtr = std::remove_reference_t<decltype(member)>;
        member      = reinterpret_cast<FnPtr>(ResolveRawSymbol(symbol_name));
        if(member == nullptr)
            missing.emplace_back(symbol_name);
    };

    bind_symbol(solver_fns_[ToSolverIndex(CKSolverType::GrpConv3dFwd)].get_all_kernel_types,
                "ckgrpconv_3d_fwd_get_all_kernel_type_strings");
    bind_symbol(solver_fns_[ToSolverIndex(CKSolverType::GrpConv3dBwd)].get_all_kernel_types,
                "ckgrpconv_3d_bwd_get_all_kernel_type_strings");
}

bool CKGroupedConvLibLoader::LoadSymbols()
{
    std::vector<std::string> missing;
    missing.reserve(32);

    BindRequiredCommonSymbols(missing);

    struct SolverSymbolBinding
    {
        CKSolverType solver;
        const char* prefix;
    };

    static constexpr SolverSymbolBinding solver_bindings[] = {
        {CKSolverType::GrpConvFwd, "fwd"},
        {CKSolverType::GrpConvBwd, "bwd"},
        {CKSolverType::GrpConvWrw, "wrw"},
        {CKSolverType::GrpConv3dFwd, "3d_fwd"},
        {CKSolverType::GrpConv3dBwd, "3d_bwd"}};

    for(const auto& binding : solver_bindings)
        BindSolverSymbols(binding.solver, binding.prefix, missing);

    BindOptionalKernelTypeSymbols(missing);

    if(missing.empty())
        return true;

    std::ostringstream msg;
    for(std::size_t i = 0; i < missing.size(); ++i)
    {
        if(i != 0)
            msg << ", ";
        msg << missing[i];
    }

    MIOPEN_LOG_W("Failed to resolve CK grouped conv symbols (" << missing.size()
                                                               << " missing): " << msg.str());
    return false;
}

// -- Helpers ------------------------------------------------------------------

std::vector<std::string> CKGroupedConvLibLoader::ExtractKernelList(CKKernelListHandle* handle) const
{
    if(handle == nullptr)
        return {};
    std::vector<std::string> result;
    const size_t n = kernel_list_size_fn_(handle);
    result.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        const char* s = kernel_list_get_fn_(handle, i);
        if(s != nullptr)
            result.emplace_back(s);
    }
    kernel_list_free_fn_(handle);
    return result;
}

ConvSolution CKGroupedConvLibLoader::ExtractSolution(ConvSolution* ptr) const
{
    if(ptr == nullptr)
        return ConvSolution{miopenStatusInternalError};
    ConvSolution result = std::move(*ptr);
    solution_free_fn_(ptr);
    return result;
}

// -- Solver-parameterized wrappers --------------------------------------------

std::vector<std::string>
CKGroupedConvLibLoader::FillValidKernels(CKSolverType solver,
                                         const conv::ProblemDescription& problem,
                                         miopenDataType_t dtype,
                                         bool use_tf32) const
{
    if(!IsLoaded())
        return {};
    return ExtractKernelList(
        solver_fns_[ToSolverIndex(solver)].fill_valid_kernels(&problem, dtype, use_tf32));
}

std::vector<std::string>
CKGroupedConvLibLoader::FillValidKernelsWithTf32Fallback(CKSolverType solver,
                                                         const conv::ProblemDescription& problem,
                                                         miopenDataType_t dtype,
                                                         bool& use_tf32) const
{
    auto result = FillValidKernels(solver, problem, dtype, use_tf32);
    if(result.empty() && use_tf32)
    {
        use_tf32 = false;
        result   = FillValidKernels(solver, problem, dtype, false);
    }
    return result;
}

bool CKGroupedConvLibLoader::IsApplicable(CKSolverType solver,
                                          const conv::ProblemDescription& problem,
                                          miopenDataType_t dtype,
                                          bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return solver_fns_[ToSolverIndex(solver)].is_applicable(&problem, dtype, use_tf32);
}

bool CKGroupedConvLibLoader::IsArgsSupported(CKSolverType solver,
                                             const conv::ProblemDescription& problem,
                                             const std::string& kernel_id,
                                             miopenDataType_t dtype,
                                             bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return solver_fns_[ToSolverIndex(solver)].is_args_supported(
        &problem, kernel_id.c_str(), dtype, use_tf32);
}

size_t CKGroupedConvLibLoader::GetWorkspaceSize(CKSolverType solver,
                                                const conv::ProblemDescription& problem,
                                                miopenDataType_t dtype,
                                                bool use_tf32) const
{
    if(!IsLoaded())
        return 0;
    return solver_fns_[ToSolverIndex(solver)].get_workspace_size(&problem, dtype, use_tf32);
}

ConvSolution CKGroupedConvLibLoader::GetSolution(CKSolverType solver,
                                                 const ExecutionContext& ctx,
                                                 const conv::ProblemDescription& problem,
                                                 const std::string& kernel_id,
                                                 bool use_tf32) const
{
    if(!IsLoaded())
        return ConvSolution{miopenStatusInternalError};
    return ExtractSolution(solver_fns_[ToSolverIndex(solver)].get_solution(
        &ctx, &problem, kernel_id.c_str(), use_tf32));
}

std::vector<std::string> CKGroupedConvLibLoader::GetAllKernelTypeStrings(CKSolverType solver) const
{
    if(!IsLoaded())
        return {};
    auto fn = solver_fns_[ToSolverIndex(solver)].get_all_kernel_types;
    if(fn == nullptr)
        return {};
    return ExtractKernelList(fn());
}

} // namespace solver
} // namespace miopen
