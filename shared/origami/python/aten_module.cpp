// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <Python.h>
#include <array>
#include <mutex>
#include <optional>
#include <vector>

#include <torch/torch.h>
#include <c10/util/Logging.h>

#include "origami/origami.hpp"

namespace origami {

static hardware_t& get_hardware_cached() {
    static std::optional<hardware_t> hardware;
    static std::once_flag flag;

    // Only retrieve the hardware config once using GPU_ID 0
    // Assumes all GPUs in the machine are the same model/variant
    std::call_once(flag, [](){
        hardware.emplace(hardware_t::get_hardware_for_device(0));
    });

    return *hardware;
}

std::array<size_t, 3> infer_mi_dimensions(size_t element_bitsize_A,
                                          size_t element_bitsize_B,
                                          size_t N_CU) {
    std::array<size_t, 3> mi_dim;
    size_t max_bitsize = std::max(element_bitsize_A, element_bitsize_B);
    
    // gfx950
    if (N_CU == 256) {
        if (max_bitsize == 32) {
            mi_dim = {16, 16, 4};
        } else if (max_bitsize == 16) {
            mi_dim = {16, 16, 32};
        } else if (max_bitsize <= 8) {
            mi_dim = {16, 16, 128};
        }
    }
    // gfx942 (MI300X)
    else if (N_CU == 304) {
        if (max_bitsize == 32) {
            mi_dim = {16, 16, 4};
        } else if (max_bitsize == 16) {
            mi_dim = {16, 16, 16};
        } else if (max_bitsize == 8) {
            mi_dim = {16, 16, 32};
        } else if (max_bitsize < 8) {
            TORCH_CHECK(false, "[ORIGAMI] MI300X doesn't support F4/F6");
        }
    }
    // gfx942 (MI300A)
    else if (N_CU == 228) {
        if (max_bitsize == 32) {
            mi_dim = {16, 16, 4};
        } else if (max_bitsize == 16) {
            mi_dim = {16, 16, 16};
        } else if (max_bitsize == 8) {
            mi_dim = {16, 16, 32};
        } else if (max_bitsize < 8) {
            TORCH_CHECK(false, "[ORIGAMI] MI300A doesn't support F4/F6");
        }
    }
    // MI200
    else if (N_CU == 104) {
        if (max_bitsize == 32) {
            mi_dim = {16, 16, 4};
        } else if (max_bitsize == 16) {
            mi_dim = {16, 16, 16};
        } else if (max_bitsize == 8) {
            TORCH_CHECK(false, "[ORIGAMI] MI200 doesn't support F8");
        } else if (max_bitsize < 8) {
            TORCH_CHECK(false, "[ORIGAMI] MI200 doesn't support F4/F6");
        }
    }
    
    // Architecture not supported
    TORCH_CHECK(!mi_dim.empty(), \
            "[ORIGAMI] No Valid Matrix Instruction integrated for given datatypes");
    
    return mi_dim;
}

problem_t build_problem(const c10::Dict<std::string, int64_t> problem_dict) {
    problem_t problem;

    // Required values (SIZE_M, SIZE_N, SIZE_K, BATCH, A_TRANSPOSE,
    //                  B_TRANSPOSE, A_DTYPE, B_DTYPE, C_DTYPE, D_DTYPE,
    //                  MI_DTYPE)
    TORCH_CHECK(problem_dict.contains("SIZE_M")     , "[ORIGAMI] Missing 'SIZE_M' key in problem dict");
    TORCH_CHECK(problem_dict.contains("SIZE_N")     , "[ORIGAMI] Missing 'SIZE_N' key in problem dict");
    TORCH_CHECK(problem_dict.contains("SIZE_K")     , "[ORIGAMI] Missing 'SIZE_K' key in problem dict");
    TORCH_CHECK(problem_dict.contains("BATCH")      , "[ORIGAMI] Missing 'BATCH' key in problem dict");
    TORCH_CHECK(problem_dict.contains("A_TRANSPOSE"), "[ORIGAMI] Missing 'A_TRANSPOSE' key in problem dict");
    TORCH_CHECK(problem_dict.contains("B_TRANSPOSE"), "[ORIGAMI] Missing 'B_TRANSPOSE' key in problem dict");
    TORCH_CHECK(problem_dict.contains("A_DTYPE")    , "[ORIGAMI] Missing 'A_DTYPE' key in problem dict");
    TORCH_CHECK(problem_dict.contains("B_DTYPE")    , "[ORIGAMI] Missing 'B_DTYPE' key in problem dict");
    TORCH_CHECK(problem_dict.contains("C_DTYPE")    , "[ORIGAMI] Missing 'C_DTYPE' key in problem dict");
    TORCH_CHECK(problem_dict.contains("D_DTYPE")    , "[ORIGAMI] Missing 'D_DTYPE' key in problem dict");
    TORCH_CHECK(problem_dict.contains("MI_DTYPE")   , "[ORIGAMI] Missing 'MI_DTYPE' key in problem dict");

    dim3_t size;
    size.m = problem_dict.at("SIZE_M");
    size.n = problem_dict.at("SIZE_N");
    size.k = problem_dict.at("SIZE_K");
    problem.size = size;

    problem.batch = problem_dict.at("BATCH");

    problem.a_transpose = static_cast<transpose_t>(problem_dict.at("A_TRANSPOSE"));
    problem.b_transpose = static_cast<transpose_t>(problem_dict.at("B_TRANSPOSE"));

    problem.a_dtype  = int_to_data_type(problem_dict.at("A_DTYPE"));
    problem.b_dtype  = int_to_data_type(problem_dict.at("B_DTYPE"));
    problem.c_dtype  = int_to_data_type(problem_dict.at("C_DTYPE"));
    problem.d_dtype  = int_to_data_type(problem_dict.at("D_DTYPE"));
    problem.mi_dtype = int_to_data_type(problem_dict.at("MI_DTYPE"));

    // We want to warn the user if there are keys present in the problem that
    // we didn't use, so we'll decrement this counter each time we use one.
    // Subtract 11 to start because we wouldn't be at this line if we didn't
    // already use the required keys.
    size_t remaining_key_count = problem_dict.size() - 11;

    // Optional values (A_MX_BLOCK_SIZE, B_MX_BLOCK_SIZE)
    if(problem_dict.contains("A_MX_BLOCK_SIZE")) {
        problem.a_mx_block_size = problem_dict.at("A_MX_BLOCK_SIZE");
        remaining_key_count--;
    }
    if(problem_dict.contains("B_MX_BLOCK_SIZE")) {
        problem.b_mx_block_size = problem_dict.at("B_MX_BLOCK_SIZE");
        remaining_key_count--;
    }

    // By this point we should have touched every key we recognize, so if
    // there's any remaining we'll warn the user to help them debug
    if(remaining_key_count != 0) {
        TORCH_WARN_ONCE("[ORIGAMI] The problem dict contained unrecognized key(s) - these will be ignored");
    }

    return problem;
}

std::vector<config_t> build_configs(const c10::optional<c10::List<c10::Dict<std::string, int64_t>>>& config_dicts,
                                    const std::array<size_t, 3> mi_dim_infer) {
    std::vector<config_t> configs;

    // Extract matrix instruction sizes
    int64_t mi_m_infer = mi_dim_infer[0];
    int64_t mi_n_infer = mi_dim_infer[1];
    int64_t mi_k_infer = mi_dim_infer[2];

    //TODO: support a default set of configs - for now error on no configs
    if(!config_dicts.has_value()) {
        TORCH_CHECK(false, "[ORIGAMI] No configs were provided and default configs not yet implemented");
    }

    // It's now safe to unwrap the optional
    const auto& unwrapped_dicts = config_dicts.value();

    // Create the config_t for each input config dict and add to the list
    for(const c10::Dict<std::string, int64_t>& dict : unwrapped_dicts) {
        config_t new_config;

        // Required values (BLK_M, BLK_N, BLK_K, OCCUPANCY)
        TORCH_CHECK(dict.contains("BLK_M")    , "[ORIGAMI] Missing 'BLK_M' key in at least 1 config dict");
        TORCH_CHECK(dict.contains("BLK_N")    , "[ORIGAMI] Missing 'BLK_N' key in at least 1 config dict");
        TORCH_CHECK(dict.contains("BLK_K")    , "[ORIGAMI] Missing 'BLK_K' key in at least 1 config dict");
        TORCH_CHECK(dict.contains("OCCUPANCY"), "[ORIGAMI] Missing 'OCCUPANCY' key in at least 1 config dict");

        dim3_t mt;
        mt.m = dict.at("BLK_M");
        mt.n = dict.at("BLK_N");
        mt.k = dict.at("BLK_K");
        new_config.mt = mt;

        new_config.occupancy = dict.at("OCCUPANCY");

        // We want to warn the user if there are keys present in the configs
        // that we didn't use, so we'll decrement this counter each time we use
        // one.  Subtract 4 to start because we wouldn't be at this line if we
        // didn't already use the required keys.
        size_t remaining_key_count = dict.size() - 4;

        // Required values we'll infer from the hardware if not provided
        // (MI_DIM_M, MI_DIM_N, MI_DIM_K)
        dim3_t mi;
        if(dict.contains("MI_DIM_M")) {
            mi.m = dict.at("MI_DIM_M");
            remaining_key_count--;
        } else {
            mi.m = mi_m_infer;
        }
        if(dict.contains("MI_DIM_N")) {
            mi.n = dict.at("MI_DIM_N");
            remaining_key_count--;
        } else {
            mi.n = mi_n_infer;
        }
        if(dict.contains("MI_DIM_K")) {
            mi.k = dict.at("MI_DIM_K");
            remaining_key_count--;
        } else {
            mi.k = mi_k_infer;
        }
        new_config.mi = mi;

        // Optional values (WG_MAPPING, WORKSPACE_SIZE, WORKSPACE_SIZE_PER_C,
        //                  REDUCTION_STRATEGY)
        if(dict.contains("WG_MAPPING")) {
            new_config.workgroup_mapping = dict.at("WG_MAPPING");
            remaining_key_count--;
        }
        if(dict.contains("WORKSPACE_SIZE")) {
            new_config.workspace_size = dict.at("WORKSPACE_SIZE");
            remaining_key_count--;
        }
        if(dict.contains("WORKSPACE_SIZE_PER_C")) {
            new_config.workspace_size_per_elem_c = dict.at("WORKSPACE_SIZE_PER_C");
            remaining_key_count--;
        }
        if(dict.contains("REDUCTION_STRATEGY")) {
            new_config.reduction_strategy = int_to_reduction_t(dict.at("REDUCTION_STRATEGY"));
            remaining_key_count--;
        }

        // Currently unsupported optional values - will warn once if we see
        // these, set them, and then keep going (CACHE_HINTS_A, CACHE_HINTS_B)
        if(dict.contains("CACHE_HINTS_A")) {
            TORCH_WARN_ONCE("[ORIGAMI] CACHE_HINTS_A are not used in this version of Origami - ignoring");
            new_config.cache_hints_a = dict.at("CACHE_HINTS_A");
            remaining_key_count--;
        }
        if(dict.contains("CACHE_HINTS_B")) {
            TORCH_WARN_ONCE("[ORIGAMI] CACHE_HINTS_B are not used in this version of Origami - ignoring");
            new_config.cache_hints_b = dict.at("CACHE_HINTS_B");
            remaining_key_count--;
        }

        // By this point we should have touched every key we recognize, so if
        // there's any remaining we'll warn the user to help them debug
        if(remaining_key_count != 0) {
            TORCH_WARN_ONCE("[ORIGAMI] At least one config dict contained unrecognized key(s) - these will be ignored");
        }

        configs.push_back(new_config);
    }

    TORCH_CHECK(!configs.empty(), \
            "[ORIGAMI] Didn't extract any valid configs from input dict");

    return configs;
}

std::tuple<double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
aten_select_config(
    const c10::Dict<std::string, int64_t>& problem_dict,
    const c10::optional<c10::List<c10::Dict<std::string, int64_t>>>& config_dicts) {

    // Build problem using the data from tensors and extra torch assumptions
    problem_t problem = build_problem(problem_dict);

    // Retrieve hardware object assuming all GPUs on the system are the same
    hardware_t hardware = get_hardware_cached();

    // Infer matrix instruction dimensions based on hardware and datatype
    size_t element_bitsize_A = data_type_to_bits(problem.a_dtype);
    size_t element_bitsize_B = data_type_to_bits(problem.b_dtype);
    std::array<size_t, 3> mi_dim_infer = infer_mi_dimensions(element_bitsize_A,
                                                             element_bitsize_B,
                                                             hardware.N_CU);

    // Build the possible configs for this problem
    std::vector<config_t> configs = build_configs(config_dicts, mi_dim_infer);

    // Do the prediction and get a result
    prediction_result_t result = select_config(problem, hardware, configs);

    // Return results as tuple: (BLK_M, BLK_N, BLK_K, GSIZE_M)
    // These are shape-dependent values that torch.compile can safely specialize on
    return std::make_tuple(
        result.latency,
        static_cast<int64_t>(result.config.mt.m),
        static_cast<int64_t>(result.config.mt.n),
        static_cast<int64_t>(result.config.mt.k),
        static_cast<int64_t>(result.config.mi.m),
        static_cast<int64_t>(result.config.mi.n),
        static_cast<int64_t>(result.config.mi.k),
        static_cast<int64_t>(result.config.occupancy),
        static_cast<int64_t>(result.config.workgroup_mapping),
        static_cast<int64_t>(result.config.cache_hints_a),
        static_cast<int64_t>(result.config.cache_hints_b),
        static_cast<int64_t>(result.config.workspace_size),
        static_cast<int64_t>(result.config.workspace_size_per_elem_c),
        static_cast<int64_t>(result.config.reduction_strategy)
    );
}

std::tuple<double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
aten_select_config_meta(
    const c10::Dict<std::string, int64_t>& problem_dict,
    const c10::optional<c10::List<c10::Dict<std::string, int64_t>>>& config_dicts) {
    // For meta tensors, we still need to compute the sizes
    // The meta implementation has the same logic as the real implementation
    return aten_select_config(problem_dict, config_dicts);
}
}  //namespace origami

TORCH_LIBRARY(origami, m) {
    //m.def("select_config(Tensor a, Tensor b, Tensor c, Dict(str, int)[]? config_dicts) -> (int, int, int, int)");
    m.def("select_config(Dict(str, int) problem_dict, Dict(str, int)[]? config_dicts) -> (float, int, int, int, int, int, int, int, int, int, int, int, int, int)");
}

TORCH_LIBRARY_IMPL(origami, CatchAll, m) {
    m.impl("select_config", origami::aten_select_config);
}

TORCH_LIBRARY_IMPL(origami, Meta, m) {
    m.impl("select_config", origami::aten_select_config_meta);
}

// Python module initialization for proper import support
// Creates an empty Python module that triggers torch op registration on import
static struct PyModuleDef aten_module = {
    PyModuleDef_HEAD_INIT,
    "_aten", // module name
    nullptr, // module documentation
    -1,      // size of per-interpreter state, -1 means module uses global state
    nullptr  // module methods (none needed, only registering torch ops)
};

PyMODINIT_FUNC PyInit__aten(void) {
    // Import torch first to ensure PyTorch is initialized
    if (PyImport_ImportModule("torch") == nullptr) {
        return nullptr;
    }
    // Create and return the module (torch ops are already registered via TORCH_LIBRARY)
    return PyModule_Create(&aten_module);
}
