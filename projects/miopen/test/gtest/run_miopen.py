#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import logging
import os
import platform
from pathlib import Path
import re
import shlex
import subprocess

logging.basicConfig(level=logging.INFO)

TEST_DIR_NAME = "MIOpen"
CTEST_TIMEOUT_SECONDS = 7200
_ROCMINFO_NAME_PATTERN = re.compile(r"^\s*Name:\s+(gfx[0-9a-z]+)\s*$", re.IGNORECASE)

# These filters mirror the pre-RFC0010 TheRock MIOpen runner. The fallback path
# uses them directly so the installed script can run without vendoring TheRock's
# shared utility implementation.
POSITIVE_FILTERS = [
    "*Fusion*",
    "*/GPU_BNBWD*_*",
    "*/GPU_BNOCLBWD*_*",
    "*/GPU_BNFWD*_*",
    "*/GPU_BNOCLFWD*_*",
    "*/GPU_BNInfer*_*",
    "*/GPU_BNActivInfer_*",
    "*/GPU_BNOCLInfer*_*",
    "*/GPU_bn_infer*_*",
    "CPU_*",
    "*/CPU_*",
    "*/GPU_Cat_*",
    "*/GPU_ConvBiasActiv*",
    "*/GPU_Conv*",
    "*/GPU_conv*",
    "*/GPU_UnitTestConv*",
    "*/GPU_GetitemBwd*",
    "*/GPU_GLU_*",
    "*/GPU_GroupConv*",
    "*/GPU_GroupNorm_*",
    "*/GPU_GRUExtra_*",
    "*/GPU_TestActivation*",
    "*/GPU_HipBLASLtGEMMTest*",
    "*/GPU_KernelTuningNetTestConv*",
    "*/GPU_Kthvalue_*",
    "*/GPU_LayerNormTest*",
    "*/GPU_LayoutTransposeTest_*",
    "*/GPU_Lrn*",
    "*/GPU_lstm_extra*",
    "*/GPU_MultiMarginLoss_*",
    "*/GPU_ConvNonpack*",
    "*/GPU_PerfConfig_HipImplicitGemm*",
    "*/GPU_AsymPooling2d_*",
    "*/GPU_WidePooling2d_*",
    "*/GPU_PReLU_*",
    "*/GPU_Reduce*",
    "*/GPU_reduce_custom_*",
    "*/GPU_regression_issue_*",
    "*/GPU_RNNExtra_*",
    "*/GPU_RoPE*",
    "*/GPU_SoftMarginLoss*",
    "*/GPU_T5LayerNormTest_*",
    "*/GPU_Op4dTensorGenericTest_*",
    "*/GPU_TernaryTensorOps_*",
    "*/GPU_unaryTensorOps_*",
    "*/GPU_Transformers*",
    "*/GPU_TunaNetTest_*",
    "*/GPU_UnitTestActivationDescriptor_*",
    "*/GPU_FinInterfaceTest*",
    "*/GPU_VecAddTest_*",
    "*/GPU_KernelTuningNetTest*",
    "*/GPU_Bwd_Mha_*",
    "*/GPU_Fwd_Mha_*",
    "*/GPU_Softmax*",
    "*/GPU_Dropout*",
    "*/GPU_MhaBackward_*",
    "*/GPU_MhaForward_*",
    "*GPU_TestMhaFind20*",
    "*/GPU_MIOpenDriver*",
]

NEGATIVE_FILTERS = [
    "*DeepBench*",
    "*MIOpenTestConv*",
    # Keep the old TheRock smoke policy for standalone fallback runs.
    "Full/GPU_MIOpenDriverConv2dTransTest*",
    "Full/GPU_Reduce_FP64*",
    "Full/GPU_BNOCLFWDTrainSerialRun3D_BFP16*",
    "Full/GPU_Lrn_FP32*",
    "Full/GPU_Lrn_FP16*",
    "Full/GPU_BNOCLInferSerialRun3D_BFP16*",
    "Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*",
    "Smoke/GPU_BNOCLInferLarge2D_BFP16*",
    "Full/GPU_BNOCLBWDSerialRun3D_BFP16*",
    "Smoke/GPU_BNOCLBWDLarge2D_BFP16*",
    "Full/GPU_UnitTestActivationDescriptor_FP32*",
    "Full/GPU_UnitTestActivationDescriptor_FP16*",
    "Full/GPU_MIOpenDriverRegressionBigTensorTest_FP32*",
    "Smoke/GPU_BNOCLBWDLargeFusedActivation2D_BFP16*",
    "Smoke/GPU_BNOCLBWDLargeFusedActivation2D_FP16*",
    "Full/GPU_ConvGrpBiasActivInfer_BFP16*",
    "Full/GPU_ConvGrpBiasActivInfer_FP32*",
    "Full/GPU_ConvGrpBiasActivInfer_FP16*",
    "Full/GPU_ConvGrpActivInfer_BFP16*",
    "Full/GPU_ConvGrpActivInfer_FP32*",
    "Full/GPU_ConvGrpActivInfer_FP16*",
    "Full/GPU_ConvGrpBiasActivInfer3D_BFP16*",
    "Full/GPU_ConvGrpBiasActivInfer3D_FP32*",
    "Full/GPU_ConvGrpBiasActivInfer3D_FP16*",
    "Full/GPU_ConvGrpActivInfer3D_BFP16*",
    "Full/GPU_ConvGrpActivInfer3D_FP32*",
    "Full/GPU_ConvGrpActivInfer3D_FP16*",
]

TESTS_TO_IGNORE = {
    "gfx110X-all": {
        "windows": [
            "Smoke/CPU_Handle_NONE.TestHIP/with_stream_false_test_id_0",
            "Full/GPU_reduce_custom_fp32_fp16_FP32.FloatTest_reduce_custom_fp32_fp16/1",
            "Full/GPU_reduce_custom_fp32_fp16_FP32.FloatTest_reduce_custom_fp32_fp16/5",
            "Full/GPU_reduce_custom_fp32_fp16_FP32.FloatTest_reduce_custom_fp32_fp16/9",
            "Full/GPU_reduce_custom_fp32_fp16_FP32.FloatTest_reduce_custom_fp32_fp16/13",
            "Full/GPU_reduce_custom_fp32_fp16_FP32.FloatTest_reduce_custom_fp32_fp16/17",
            "Full/GPU_reduce_custom_fp32_fp16_FP16.HalfTest_reduce_custom_fp32_fp16/1",
            "Full/GPU_reduce_custom_fp32_fp16_FP16.HalfTest_reduce_custom_fp32_fp16/5",
            "Full/GPU_reduce_custom_fp32_fp16_FP16.HalfTest_reduce_custom_fp32_fp16/9",
            "Full/GPU_reduce_custom_fp32_fp16_FP16.HalfTest_reduce_custom_fp32_fp16/13",
            "Full/GPU_reduce_custom_fp32_fp16_FP16.HalfTest_reduce_custom_fp32_fp16/17",
        ]
    },
    "gfx1151": {
        "windows": ["Full/GPU_UnitTestConvSolverGemmBwdRestBwd_FP16.GemmBwdRest/0"]
    },
    "gfx950-dcgpu": {"linux": ["*DBSync*"]},
}

NAVI_NEGATIVE_FILTERS = [
    "Smoke/GPU_BNFWDTrainLargeFusedActivation2D_FP32.BnV2LargeFWD_TrainCKfp32Activation/NCHW_BNSpatial_testBNAPIV1_Dim_2_test_id_32",
    "Smoke/GPU_BNFWDTrainLarge2D_FP32.BnV2LargeFWD_TrainCKfp32/NCHW_BNSpatial_testBNAPIV2_Dim_2_test_id_64",
    "*SerialRun3D*",
    "*gfx942*",
    "*GPU_UnitTestConvSolverFFTFwd_FP32*",
    "*GPU_UnitTestConvSolverFFTBwd_FP32*",
    "*GPU_TernaryTensorOps_FP64*",
    "*GPU_TernaryTensorOps_FP16*",
    "*GPU_TernaryTensorOps_FP32*",
    "*GPU_Op4dTensorGenericTest_FP32*",
    "*GPU_UnitTestActivationDescriptor_FP16*",
    "*GPU_UnitTestActivationDescriptor_FP32*",
    "*CPU_TuningPolicy_NONE*",
    "*GPU_Dropout_FP32*",
    "*GPU_Dropout_FP16*",
    "*GPU_GroupConv3D_BackwardData_FP16.GroupConv3D_BackwardData_half_Test*",
    "*GPU_GroupConv3D_BackwardData_BFP16.GroupConv3D_BackwardData_bfloat16_Test*",
    "*GPU_UnitTestConvSolverImplicitGemmGroupWrwXdlops_BFP16.ConvHipImplicitGemmGroupWrwXdlops*",
    "Smoke/GPU_MultiMarginLoss*",
    "*CPU_UnitTestConvSolverImplicitGemmGroupWrwXdlopsDevApplicability_FP16.ConvHipImplicitGemmGroupWrwXdlops*",
    "Full/GPU_Softmax_FP32*",
    "Full/GPU_Softmax_BFP16*",
    "Full/GPU_Softmax_FP16*",
    "Smoke/GPU_Reduce_FP32*",
    "Smoke/GPU_Reduce_FP16*",
]

QUICK_FILTERS = [
    "Smoke/GPU_BNCKFWDTrainLarge2D_FP16*",
    "Smoke/GPU_BNOCLFWDTrainLarge2D_FP16*",
    "Smoke/GPU_BNOCLFWDTrainLarge3D_FP16*",
    "Smoke/GPU_BNCKFWDTrainLarge2D_BFP16*",
    "Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*",
    "Smoke/GPU_BNOCLFWDTrainLarge3D_BFP16*",
    "Smoke/GPU_UnitTestConvSolverImplicitGemmFwdXdlops_FP16*",
    "Smoke/GPU_UnitTestConvSolverImplicitGemmFwdXdlops_BFP16*",
]


def load_shared_test_utils():
    tools_dir = os.getenv("THEROCK_TEST_TOOLS_DIR")
    if not tools_dir:
        return None

    utils_path = Path(tools_dir) / "test_utils.py"
    if not utils_path.is_file():
        return None

    spec = importlib.util.spec_from_file_location("therock_test_utils", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared test utilities from {utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logging.info("++ Loaded shared test utilities from %s", utils_path)
    return module


def miopen_gtest_candidates(bin_dir: Path) -> list[Path]:
    candidates = [bin_dir / "miopen_gtest"]
    if os.name == "nt":
        candidates.insert(0, bin_dir / "miopen_gtest.exe")
    return candidates


def resolve_miopen_gtest(bin_dir: Path) -> Path:
    for candidate in miopen_gtest_candidates(bin_dir):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"MIOpen gtest executable not found in {bin_dir}: "
        f"{', '.join(str(p.name) for p in miopen_gtest_candidates(bin_dir))}"
    )


def has_miopen_gtest(bin_dir: Path) -> bool:
    return any(candidate.is_file() for candidate in miopen_gtest_candidates(bin_dir))


def derive_rocm_path(script_dir: Path) -> Path:
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "bin" / TEST_DIR_NAME / "CTestTestfile.cmake").is_file():
            return candidate
        if candidate.name == "bin":
            if (candidate / TEST_DIR_NAME / "CTestTestfile.cmake").is_file():
                return candidate.parent
            if has_miopen_gtest(candidate):
                return candidate.parent
        if has_miopen_gtest(candidate / "bin"):
            return candidate
    raise RuntimeError(
        "ROCM_PATH is required when run_miopen.py is not executed from an "
        "installed ROCm tree containing bin/miopen_gtest."
    )


def positive_int(name: str, value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def gtest_shard_env(shard_index: str | int, total_shards: str | int) -> dict[str, str]:
    parsed_shard_index = positive_int("shard_index", shard_index)
    parsed_total_shards = positive_int("total_shards", total_shards)
    if parsed_shard_index > parsed_total_shards:
        raise ValueError(
            "shard_index must be less than or equal to total_shards, "
            f"got {parsed_shard_index} > {parsed_total_shards}"
        )
    return {
        "GTEST_SHARD_INDEX": str(parsed_shard_index - 1),
        "GTEST_TOTAL_SHARDS": str(parsed_total_shards),
    }


def ctest_parallel_count() -> int:
    amdgpu_families = os.getenv("AMDGPU_FAMILIES", "")
    if "gfx1152" in amdgpu_families or "gfx1153" in amdgpu_families:
        return 4
    return 8


def rocminfo_command(bin_dir: Path) -> str:
    rocminfo = bin_dir / ("rocminfo.exe" if os.name == "nt" else "rocminfo")
    return str(rocminfo) if rocminfo.is_file() else "rocminfo"


def parse_rocminfo_gpu_archs(output: str) -> list[str]:
    gpu_archs = []
    for line in output.splitlines():
        match = _ROCMINFO_NAME_PATTERN.match(line)
        if match:
            gpu_archs.append(match.group(1).lower())
    return gpu_archs


def detect_gpu_arch(bin_dir: Path) -> str:
    try:
        result = subprocess.run(
            [rocminfo_command(bin_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logging.warning("rocminfo not found; GPU-specific MIOpen filters disabled")
        return ""

    if result.returncode != 0:
        logging.warning("rocminfo failed; GPU-specific MIOpen filters disabled")
        return ""

    gpu_archs = parse_rocminfo_gpu_archs(result.stdout or "")
    if not gpu_archs:
        logging.warning("No GPU architecture found in rocminfo output")
        return ""

    logging.info("Detected GPU architecture: %s", gpu_archs[0])
    return gpu_archs[0]


def amdgpu_family_for_filters(bin_dir: Path) -> str:
    amdgpu_families = os.getenv("AMDGPU_FAMILIES")
    if amdgpu_families:
        return amdgpu_families
    # Standalone runs outside TheRock CI usually do not have AMDGPU_FAMILIES.
    # Use rocminfo so the old architecture-specific filter policy still works.
    return detect_gpu_arch(bin_dir)


def build_miopen_gtest_filter(bin_dir: Path) -> str:
    amdgpu_family = amdgpu_family_for_filters(bin_dir)
    os_type = platform.system().lower()
    positive_filters = list(POSITIVE_FILTERS)
    negative_filters = list(NEGATIVE_FILTERS)

    if amdgpu_family in TESTS_TO_IGNORE and os_type in TESTS_TO_IGNORE[amdgpu_family]:
        negative_filters.extend(TESTS_TO_IGNORE[amdgpu_family][os_type])

    if "gfx110" in amdgpu_family:
        negative_filters.extend(
            [
                "*/GPU_MIOpenDriver*",
                "Smoke/CPU_Handle_NONE*",
                "Full/GPU_reduce_custom_fp32*",
            ]
        )

    if any(prefix in amdgpu_family for prefix in ["gfx110", "gfx115", "gfx120"]):
        negative_filters.extend(NAVI_NEGATIVE_FILTERS)

    quick_filters = list(QUICK_FILTERS)
    if amdgpu_family != "gfx950-dcgpu":
        quick_filters.append("*DBSync*")
        positive_filters.append("*DBSync*")

    if os.getenv("TEST_TYPE", "full").strip().lower() == "quick":
        return "--gtest_filter=" + ":".join(quick_filters)
    return "--gtest_filter=" + ":".join(positive_filters) + "-" + ":".join(
        negative_filters
    )


def build_env(rocm_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_PATH"] = str(rocm_path)
    env.update(
        gtest_shard_env(os.getenv("SHARD_INDEX", 1), os.getenv("TOTAL_SHARDS", 1))
    )
    return env


def rocm_bin_dir(rocm_path: Path) -> Path:
    bin_dir_env = os.getenv("ROCM_BIN_DIR") or os.getenv("THEROCK_BIN_DIR")
    return Path(bin_dir_env).resolve() if bin_dir_env else rocm_path / "bin"


def run_with_shared_utils(test_utils, rocm_path: Path, test_dir: Path) -> int:
    settings = test_utils.TestRunSettings.from_env(
        test_dir=test_dir,
        rocm_path=rocm_path,
    ).with_ctest(
        parallel=ctest_parallel_count(),
        timeout_seconds=CTEST_TIMEOUT_SECONDS,
    )
    env = test_utils.build_test_env(settings, base_env=os.environ)
    result = test_utils.run_ctest(
        settings,
        cwd=rocm_path,
        env=env,
        check=False,
        discover_labels=True,
    )
    return result.returncode


def run_with_fallback(rocm_path: Path) -> int:
    bin_dir = rocm_bin_dir(rocm_path)
    cmd = [str(resolve_miopen_gtest(bin_dir)), build_miopen_gtest_filter(bin_dir)]
    logging.info("++ Exec [%s]$ %s", rocm_path, shlex.join(cmd))
    result = subprocess.run(cmd, cwd=rocm_path, env=build_env(rocm_path), check=False)
    return result.returncode


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    rocm_path_env = os.getenv("ROCM_PATH")
    rocm_path = (
        Path(rocm_path_env).resolve() if rocm_path_env else derive_rocm_path(script_dir)
    )
    test_dir = rocm_bin_dir(rocm_path) / TEST_DIR_NAME

    test_utils = load_shared_test_utils()
    if test_utils is not None:
        if not (test_dir / "CTestTestfile.cmake").is_file():
            raise FileNotFoundError(f"MIOpen CTest metadata not found in {test_dir}")
        return run_with_shared_utils(test_utils, rocm_path, test_dir)
    return run_with_fallback(rocm_path)


if __name__ == "__main__":
    raise SystemExit(main())
