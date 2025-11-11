// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <Python.h>
#include <array>
#include <mutex>
#include <optional>
#include <vector>

#include <torch/torch.h>

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

data_type_t attype_to_dtype(at::ScalarType attype) {
    data_type_t dtype;

    // Torch supports some types origami doesn't and origami some that torch
    // doesn't - this switch list is the intersection of those and will raise
    // an error if the torch inputs are an unsupported type
    switch(attype) {
    case at::ScalarType::Float:
        dtype = origami::data_type_t::Float;
        break;
    case at::ScalarType::Double:
        dtype = origami::data_type_t::Double;
        break;
    case at::ScalarType::ComplexFloat:
        dtype = origami::data_type_t::ComplexFloat;
        break;
    case at::ScalarType::ComplexDouble:
        dtype = origami::data_type_t::ComplexDouble;
        break;
    case at::ScalarType::Half:
        dtype = origami::data_type_t::Half;
        break;
    case at::ScalarType::Int:
        dtype = origami::data_type_t::Int32;
        break;
    case at::ScalarType::BFloat16:
        dtype = origami::data_type_t::BFloat16;
        break;
    case at::ScalarType::Char:
        dtype = origami::data_type_t::Int8;
        break;
    case at::ScalarType::Long:
        dtype = origami::data_type_t::Int64;
        break;
    default:
        TORCH_CHECK(false, "Unsupported torch dtype passed to origami");
    }

    return dtype;
}

problem_t build_problem_from_tensors(const at::Tensor& a,
                                     const at::Tensor& b,
                                     const at::Tensor& c) {
    problem_t problem;

    // Problem dimensions - M and N from C, K from A
    dim3_t dimensions;
    dimensions.m = c.sizes()[0];
    dimensions.n = c.sizes()[1];
    dimensions.k = a.sizes()[1];
    problem.size = dimensions;

    // Batch - set to 1
    problem.batch = 1;

    // Transpose type is TN
    problem.a_transpose = transpose_t::T;
    problem.b_transpose = transpose_t::N;

    // Data types for A, B, C, and D
    problem.a_dtype = attype_to_dtype(a.scalar_type());
    problem.b_dtype = attype_to_dtype(b.scalar_type());
    problem.c_dtype = attype_to_dtype(c.scalar_type());
    problem.d_dtype = problem.c_dtype;

    // Compute types are unchanged from A's type for now
    problem.mi_dtype = problem.a_dtype;

    // MX block size - MX not well supported by torch at the moment
    problem.a_mx_block_size = 0;
    problem.b_mx_block_size = 0;

    return problem;
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
            TORCH_CHECK(false, "MI300X doesn't support F4/F6");
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
            TORCH_CHECK(false, "MI300A doesn't support F4/F6");
        }
    }
    // MI200
    else if (N_CU == 104) {
        if (max_bitsize == 32) {
            mi_dim = {16, 16, 4};
        } else if (max_bitsize == 16) {
            mi_dim = {16, 16, 16};
        } else if (max_bitsize == 8) {
            TORCH_CHECK(false, "MI200 doesn't support F8");
        } else if (max_bitsize < 8) {
            TORCH_CHECK(false, "MI200 doesn't support F4/F6");
        }
    }
    
    // Architecture not supported
    TORCH_CHECK(!mi_dim.empty(), \
            "No Valid Matrix Instruction integrated for given datatypes");
    
    return mi_dim;
}

std::vector<config_t> build_configs(const c10::optional<c10::List<c10::Dict<std::string, int64_t>>>& config_dicts,
                                    const std::array<size_t, 3> mi_dim) {
    std::vector<config_t> configs;

    // Extract matrix instruction sizes
    int64_t mi_m = mi_dim[0];
    int64_t mi_n = mi_dim[1];
    int64_t mi_k = mi_dim[2];

    //TODO: support a default set of configs - for now error on no configs
    if(!config_dicts.has_value()) {
        TORCH_CHECK(false, "Default configs not yet implemented");
    }

    // It's now safe to unwrap the optional
    const auto& unwrapped_dicts = config_dicts.value();

    // Create the config_t for each input config dict and add to the list
    for(const c10::Dict<std::string, int64_t>& dict : unwrapped_dicts) {
        config_t new_config;

        dim3_t mt;
        mt.m = dict.at("BLK_M");
        mt.n = dict.at("BLK_N");
        mt.k = dict.at("BLK_K");
        new_config.mt = mt;

        dim3_t mi;
        mi.m = mi_m;
        mi.n = mi_n;
        mi.k = mi_k;
        new_config.mi = mi;

        new_config.workgroup_mapping = dict.at("GSIZE_M");

        new_config.occupancy = dict.at("OCCUPANCY");

        //TODO: Add other config_t parameters here as we add support for them

        configs.push_back(new_config);
    }

    TORCH_CHECK(!configs.empty(), \
            "Didn't extract any valid configs from input dict");

    return configs;
}

std::tuple<int64_t, int64_t, int64_t, int64_t> at_select_config(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const c10::optional<c10::List<c10::Dict<std::string, int64_t>>>& config_dicts) {

    // Build problem using the data from tensors and extra torch assumptions
    problem_t problem = build_problem_from_tensors(a, b, c);

    // Retrieve hardware object assuming all GPUs on the system are the same
    hardware_t hardware = get_hardware_cached();

    // Infer matrix instruction dimensions based on hardware and datatype
    size_t element_bitsize_A = a.element_size() * 8;
    size_t element_bitsize_B = b.element_size() * 8;
    std::array<size_t, 3> mi_dim = infer_mi_dimensions(element_bitsize_A,
                                                       element_bitsize_B,
                                                       hardware.N_CU);

    // Build the possible configs for this problem
    std::vector<config_t> configs = build_configs(config_dicts, mi_dim);

    // Do the prediction and get a result
    prediction_result_t result = select_config(problem, hardware, configs);

    // Return results as tuple: (BLK_M, BLK_N, BLK_K, GSIZE_M)
    // These are shape-dependent values that torch.compile can safely specialize on
    return std::make_tuple(
        static_cast<int64_t>(result.config.mt.m),
        static_cast<int64_t>(result.config.mt.n),
        static_cast<int64_t>(result.config.mt.k),
        static_cast<int64_t>(result.config.workgroup_mapping)
    );
}

std::tuple<int64_t, int64_t, int64_t, int64_t> at_select_config_meta(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const c10::optional<c10::List<c10::Dict<std::string, int64_t>>>& config_dicts) {
    // For meta tensors, we still need to compute the sizes
    // The meta implementation has the same logic as the real one
    return at_select_config(a, b, c, config_dicts);
}
}  //namespace origami

TORCH_LIBRARY(origami, m) {
    m.def("select_config(Tensor a, Tensor b, Tensor c, Dict(str, int)[]? config_dicts) -> (int, int, int, int)");
}

TORCH_LIBRARY_IMPL(origami, CUDA, m) {
    m.impl("select_config", origami::at_select_config);
}

TORCH_LIBRARY_IMPL(origami, Meta, m) {
    m.impl("select_config", origami::at_select_config_meta);
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

