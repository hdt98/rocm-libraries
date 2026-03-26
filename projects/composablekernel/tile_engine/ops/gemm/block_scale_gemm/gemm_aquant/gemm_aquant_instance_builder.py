# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import argparse
import importlib.util
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Import the base GemmKernelBuilder from the gemm directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(gemm_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder


# def _import_validation_utils():
#     """Import validation utilities from the gemm directory."""
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     gemm_dir = os.path.dirname(os.path.dirname(current_dir))

#     spec = importlib.util.spec_from_file_location(
#         "validation_utils",
#         os.path.join(gemm_dir, "gemm_validation_utils.py"),
#     )
#     validation_utils = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(validation_utils)

#     return validation_utils


GemmKernelBuilder = _import_gemm_kernel_builder()
# _validation_utils = _import_validation_utils()
# is_tile_config_valid = _validation_utils.is_tile_config_valid
# get_abc_layouts = _validation_utils.get_abc_layouts
# get_dtype_string = _validation_utils.get_dtype_string

# # AQuant-specific pipeline unsupported combinations:
# # mem pipeline only supports interwave scheduler
# # compv3 pipeline only supports intrawave scheduler


class GemmAQuantKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json=None,
    ):
        super().__init__(
            kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
        )
        self.group_size_k = self.config.get("group_size_k", 128)

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(
                multiprocessing.cpu_count(), 8
            )  # Limit to avoid memory issues

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        # Prepare work items for parallel processing
        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                work_items.append(
                    (
                        tile_config,
                        trait_combo,
                        self.kernel_name_prefix,
                        self.working_path,
                        self.gpu_target,
                        self.datatype,
                        self.layout,
                        self.config_json,
                    )
                )
        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Total kernels: {len(work_items)}")

        # Show first few work items for debugging
        if work_items:
            print("  First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")  # Show first 3 traits

        # Process work items in parallel
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Submit all work items
            print(f"  Submitting {len(work_items)} tasks to executor...")
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            print("  All tasks submitted, waiting for completion...")

            # Collect results with progress reporting
            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 100 == 0 or completed == len(work_items):
                    print(
                        f"  Progress: {completed}/{len(work_items)} kernels generated"
                    )
                try:
                    result = future.result()
                    if result:
                        kernel_list.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"Kernel generation failed for {item}: {exc}")

        # Sort kernel list for consistent ordering
        kernel_list.sort(key=lambda x: x[0])  # Sort by kernel name

        # Generate CMake include file for individual targets
        self._generate_cmake_individual_targets(kernel_list)

        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )


#     # def _validate_tile_config(
#     #     self,
#     #     tile_m,
#     #     tile_n,
#     #     tile_k,
#     #     warp_m,
#     #     warp_n,
#     #     warp_k,
#     #     warp_tile_m,
#     #     warp_tile_n,
#     #     warp_tile_k,
#     #     pipeline,
#     # ):
#     #     """Validate tile configuration for AQuant GEMM."""
#     #     a_datatype = self.datatype
#     #     b_datatype = self.datatype
#     #     c_datatype = "fp16" if self.datatype in ["fp8", "bf8"] else self.datatype

#     #     # AQuant-specific: tile_k must be a multiple of group_size_k
#     #     # (KPerBlockAQ = KPerBlock / QuantGroupSize::kK must be integer > 0)
#     #     if tile_k % self.group_size_k != 0 or tile_k < self.group_size_k:
#     #         return False

#     #     # AQuant-specific: group_size_k must be divisible by warp_tile_k
#     #     # (from pipeline policy static_assert)
#     #     if self.group_size_k % warp_tile_k != 0:
#     #         return False

#     #     # AQuant-specific: enforce warp_tile_m == warp_tile_n (MFMA requirement)
#     #     if warp_tile_m != warp_tile_n:
#     #         return False

#     #     # AQuant-specific: enforce warp_tile_m ↔ warp_tile_k coupling
#     #     # from get_k_warp_tile() in tile_gemm_shape.hpp
#     #     # For fp8/bf8 on non-WMMA, non-gfx950:
#     #     #   warp_tile_m=32 → warp_tile_k=32
#     #     #   warp_tile_m=16 → warp_tile_k=64
#     #     # For fp8/bf8 on gfx950:
#     #     #   warp_tile_m=32 → warp_tile_k=64
#     #     #   warp_tile_m=16 → warp_tile_k=128
#     #     if self.datatype in ["fp8", "bf8"]:
#     #         if self.gpu_target == "gfx950":
#     #             expected_k = 64 if warp_tile_m == 32 else 128
#     #         else:
#     #             expected_k = 32 if warp_tile_m == 32 else 64
#     #         if warp_tile_k != expected_k:
#     #             return False

#     #     if not is_tile_config_valid(
#     #         tile_m,
#     #         tile_n,
#     #         tile_k,
#     #         warp_m,
#     #         warp_n,
#     #         warp_k,
#     #         warp_tile_m,
#     #         warp_tile_n,
#     #         warp_tile_k,
#     #         a_datatype,
#     #         b_datatype,
#     #         c_datatype,
#     #         pipeline,
#     #         self.layout,
#     #         self.gpu_target,
#     #     ):
#     #         return False

#     #     return True


#     def _generate_kernel_instance(self, tile_config, trait_combo):
#         """Generate a single AQuant kernel instance header."""
#         k_block_per_cu = self.config.get("k_block_per_cu", 1)

#         (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, a_preshuffle_quant) = (
#             trait_combo
#         )

#         # Build kernel name
#         kernel_name = (
#             f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}"
#             f"_{pipeline}_{epilogue}_{scheduler}"
#             f"_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}"
#             f"_{str(pad_k).capitalize()}"
#             f"_{str(a_preshuffle_quant).capitalize()}"
#         )
#         tile_str = (
#             f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
#             f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
#             f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}"
#             f"x{tile_config['warp_tile_k']}"
#         )
#         kernel_name += f"_{tile_str}"

#         # Determine output type
#         c_type = "fp16" if self.datatype in ["fp8", "bf8"] else self.datatype

#         # Layouts
#         a_layout, b_layout, c_layout = get_abc_layouts(self.layout)

#         # AQ layout: follows ALayout except for CRR where AQ is RowMajor
#         # (matching the example's run_gemm_example_with_layouts calls)
#         layout_code = str(self.layout).strip().lower()
#         if layout_code == "crr":
#             aq_layout = "ck_tile::tensor_layout::gemm::RowMajor"
#         else:
#             aq_layout = a_layout

#         # Pipeline mapping
#         pipeline_impl_map = {
#             "mem": "ck_tile::AQuantGemmPipelineAgBgCrMem",
#             "compv3": "ck_tile::AQuantGemmPipelineAgBgCrCompV3",
#         }
#         base_pipeline_map = {
#             "mem": "ck_tile::BaseGemmPipelineAgBgCrMem",
#             "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
#         }
#         scheduler_type_map = {
#             "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
#             "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
#         }

#         is_pad_m = "true" if pad_m in [True, "true"] else "false"
#         is_pad_n = "true" if pad_n in [True, "true"] else "false"
#         is_pad_k = "true" if pad_k in [True, "true"] else "false"
#         is_a_preshuffle = "true" if a_preshuffle_quant in [True, "true"] else "false"

#         instance_code = f"""// Generated AQuant kernel instance for {kernel_name}
# #pragma once

# #include <cstdint>
# #include <utility>
# #include <tuple>
# #include "ck_tile/core.hpp"
# #include "ck_tile/host/kernel_launch.hpp"
# #include "ck_tile/ops/gemm.hpp"
# #include "ck_tile/ops/gemm_quant.hpp"
# #include "ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp"
# #include "ck_tile/ops/common/tensor_layout.hpp"
# #include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"

# using ADataType = {get_dtype_string(self.datatype)};
# using BDataType = {get_dtype_string(self.datatype)};
# using AQDataType = float;
# using AccDataType = float;
# using CDataType = {get_dtype_string(c_type)};

# using ALayout = {a_layout};
# using BLayout = {b_layout};
# using CLayout = {c_layout};
# using AQLayout = {aq_layout};

# // Kernel name for display
# constexpr const char* KERNEL_NAME = "{kernel_name}";

# // Wrapper for simplified launch interface
# struct SelectedKernel {{
#     // Tile configuration
#     static constexpr ck_tile::index_t BlockSize = 256;
#     static constexpr ck_tile::index_t TileM = {tile_config["tile_m"]};
#     static constexpr ck_tile::index_t TileN = {tile_config["tile_n"]};
#     static constexpr ck_tile::index_t TileK = {tile_config["tile_k"]};
#     static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
#     static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
#     static constexpr ck_tile::index_t WarpPerBlock_K = {tile_config["warp_k"]};
#     static constexpr ck_tile::index_t WarpTileM = {tile_config["warp_tile_m"]};
#     static constexpr ck_tile::index_t WarpTileN = {tile_config["warp_tile_n"]};
#     static constexpr ck_tile::index_t WarpTileK = {tile_config["warp_tile_k"]};

#     // AQuant traits
#     static constexpr bool kPadM = {is_pad_m};
#     static constexpr bool kPadN = {is_pad_n};
#     static constexpr bool kPadK = {is_pad_k};
#     static constexpr bool TransposeC = false;
#     static constexpr bool APreshuffleQuant = {is_a_preshuffle};
#     static constexpr bool BPreshuffleQuant = false;
#     static constexpr bool PreshuffleB = false;
#     static constexpr bool DoubleSmemBuffer = false;
#     static constexpr ck_tile::index_t GroupSizeK = {self.group_size_k};

#     // Tile shape
#     using TileShape = ck_tile::TileGemmShape<
#         ck_tile::sequence<TileM, TileN, TileK>,
#         ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
#         ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

#     // Tile partitioner
#     using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

#     // Quantization group size
#     using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, GroupSizeK>>;

#     // Traits
#     using Traits = ck_tile::TileGemmQuantTraits<
#         kPadM, kPadN, kPadK,
#         APreshuffleQuant,
#         BPreshuffleQuant,
#         PreshuffleB,
#         ALayout, BLayout, CLayout,
#         ck_tile::QuantType::AQuantGrouped,
#         AQLayout>;

#     // Pipeline problem (CDataType_ param is the accumulator type for the pipeline,
#     // not the output type — the output conversion is handled by the epilogue)
#     using GemmPipelineProblem = ck_tile::GemmAQuantPipelineProblem<
#         ADataType,
#         AQDataType,
#         BDataType,
#         AccDataType,
#         TileShape,
#         Traits,
#         QuantGroupSize,
#         TransposeC,
#         BDataType,
#         {scheduler_type_map.get(scheduler)}>;

#     // Base pipeline for hot loop detection
#     using BaseGemmPipeline = {base_pipeline_map.get(pipeline)}<GemmPipelineProblem>;

#     // Launch function
#     static float launch(const ck_tile::QuantGemmHostArgs& args,
#                         const ck_tile::stream_config& stream) {{

#         using _GemmPipelineProblem = SelectedKernel::GemmPipelineProblem;
#         using GemmPipeline = {pipeline_impl_map.get(pipeline)}<_GemmPipelineProblem>;

#         // Epilogue
#         using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
#             ADataType,
#             BDataType,
#             ck_tile::tuple<>,
#             AccDataType,
#             CDataType,
#             ck_tile::tuple<>,
#             CLayout,
#             ck_tile::element_wise::PassThrough,
#             SelectedKernel::TileM,
#             SelectedKernel::TileN,
#             SelectedKernel::kPadM,
#             SelectedKernel::kPadN,
#             SelectedKernel::WarpTileM,
#             SelectedKernel::WarpTileN,
#             SelectedKernel::WarpTileK,
#             SelectedKernel::TransposeC>;

#         using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;

#         // Kernel type
#         using Kernel = ck_tile::QuantGemmKernel<
#             SelectedKernel::TilePartitioner, GemmPipeline, GemmEpilogue,
#             ck_tile::QuantType::AQuantGrouped>;

#         // Kernel arguments
#         auto kargs = Kernel::MakeKernelArgs(args);

#         if (!Kernel::IsSupportedArgument(kargs)) {{
#             throw std::runtime_error("Unsupported kernel arguments; skipping GEMM launch.");
#         }}

#         // Get grid and block sizes
#         const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);
#         const dim3 blocks = Kernel::BlockSize();

#         if(stream.log_level_ > 0) {{
#             std::cout << "Launching kernel: " << Kernel::GetName() << '\\n'
#                       << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
#                       << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
#                       << std::endl;
#         }}

#         // Launch kernel
#         constexpr int kBlockPerCu = {k_block_per_cu};
#         float ave_time = ck_tile::launch_kernel(
#             stream,
#             ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));

#         return ave_time;
#     }}
# }};
# """

#         # Write the header file
#         simplified_name = kernel_name
#         if simplified_name.startswith(f"{self.kernel_name_prefix}_"):
#             simplified_name = simplified_name[len(self.kernel_name_prefix) + 1 :]

#         header_file = (
#             self.working_path
#             / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
#         )
#         with open(header_file, "w") as f:
#             f.write(instance_code)

#         print(f"Generated {header_file}")
#         return kernel_name, instance_code

#     def _generate_all_individual(self, num_workers=None):
#         """Generate individual kernel files with parallel processing."""
#         if num_workers is None:
#             num_workers = min(multiprocessing.cpu_count(), 8)

#         tile_configs = self._get_tile_configs()
#         trait_combos = self._generate_trait_combinations()

#         work_items = []
#         for tile_config in tile_configs:
#             for trait_combo in trait_combos:
#                 work_items.append(
#                     (
#                         tile_config,
#                         trait_combo,
#                         self.kernel_name_prefix,
#                         self.working_path,
#                         self.gpu_target,
#                         self.datatype,
#                         self.layout,
#                         self.config_json,
#                     )
#                 )

#         print(
#             f"Generating {len(work_items)} individual kernel files "
#             f"using {num_workers} workers..."
#         )
#         print(f"  Tile configs: {len(tile_configs)}")
#         print(f"  Trait combinations: {len(trait_combos)}")

#         kernel_list = []
#         completed = 0

#         with concurrent.futures.ProcessPoolExecutor(
#             max_workers=num_workers
#         ) as executor:
#             future_to_item = {
#                 executor.submit(_generate_single_kernel_individual, item): item
#                 for item in work_items
#             }
#             for future in concurrent.futures.as_completed(future_to_item):
#                 completed += 1
#                 if completed % 50 == 0 or completed == len(work_items):
#                     print(
#                         f"  Progress: {completed}/{len(work_items)} kernels generated"
#                     )
#                 try:
#                     result = future.result()
#                     if result:
#                         kernel_list.append(result)
#                 except Exception as exc:
#                     item = future_to_item[future]
#                     print(f"Kernel generation failed for {item}: {exc}")

#         kernel_list.sort(key=lambda x: x[0])
#         self._generate_cmake_individual_targets(kernel_list)
#         print(
#             f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
#         )


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual kernel file"""
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ) = work_item

    # Create a temporary builder instance for this worker
    builder = GemmAQuantKernelBuilder(
        kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        # Create simplified filename without the "gemm_aquant_" prefix
        # Remove "gemm_aquant_" from the beginning of kernel_name for the filename
        simplified_name = kernel_name
        if simplified_name.startswith("gemm_aquant_"):
            simplified_name = simplified_name[
                len(kernel_name_prefix) + 1 :
            ]  # Remove "gemm_aquant" prefix

        # Write individual header file
        header_file = working_path / f"gemm_aquant_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="GEMM AQuant kernel instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp8", "bf8"],
        help="Data type (fp8 or bf8)",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single",
        action="store_true",
        help="Generate a single kernel file",
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument("--tile_config", help="Tile configuration string")
    parser.add_argument("--trait_combo", help="Trait combination string")
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )

    args = parser.parse_args()

    assert args.datatype in ["fp16", "bf16", "fp8", "bf8"], (
        f"Invalid datatype string: {args.datatype} (supported datatypes are [fp16, bf16, fp8, and bf8])"
    )

    layout_parts = args.layout.lower()
    assert len(layout_parts) == 3, (
        f"Invalid layout string: {args.layout} (must be 3 characters like 'rcr' where r stands for row major and c stands for column major)"
    )
    assert layout_parts[0] in ["r", "c"] and layout_parts[1] in ["r", "c"], (
        f"Invalid matrix_a layout : {layout_parts[0]} or matrix_b layout: {layout_parts[1]} (matrix_a and matrix_b must be either 'r' for row major or 'c' for column major)"
    )
    assert layout_parts[2] == "r", (
        f"Invalid matrix_c layout: {layout_parts[2]} (must be 'r' only as currently we are supporting only row major)"
    )

    kernel_name_prefix = "gemm_aquant"
    builder = GemmAQuantKernelBuilder(
        kernel_name_prefix,
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.config_json,
    )

    if args.list_kernels:
        builder._list_kernels()
    elif args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        # Parse tile config
        tile_parts = args.tile_config.split("_")
        tile_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")

        tile_config = {
            "tile_m": int(tile_dims[0]),
            "tile_n": int(tile_dims[1]),
            "tile_k": int(tile_dims[2]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "warp_k": int(warp_dims[2]),
            "warp_tile_m": int(warp_tile_dims[0]),
            "warp_tile_n": int(warp_tile_dims[1]),
            "warp_tile_k": int(warp_tile_dims[2]),
        }

        # Parse trait combo:
        # pipeline_epilogue_scheduler_padM_padN_padK_aPreshuffle
        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            trait_parts[0],  # pipeline
            trait_parts[1],  # epilogue
            trait_parts[2],  # scheduler
            trait_parts[3] == "True",  # pad_m
            trait_parts[4] == "True",  # pad_n
            trait_parts[5] == "True",  # pad_k
            trait_parts[6] == "True",  # a_preshuffle_quant
        )

        builder._generate_kernel_instance(tile_config, trait_combo)
    elif args.gen_all_individual:
        builder._generate_all_individual(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
