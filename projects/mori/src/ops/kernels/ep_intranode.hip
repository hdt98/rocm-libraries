// KernelType::IntraNode — dispatch + combine for intra-node (P2P) mode.
#include "src/ops/kernels/ep_common.hip"
MORI_DEFINE_GPU_STATES

// Dispatch
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, , false)
WRAP_ALL_TYPES_BOOL(EpDispatchIntraNodeKernel, _stdmoe, true)

#define WRAP_STANDARD_EP_COMPACT_DISPATCH(suffix, T)                                      \
  extern "C" __global__ void StandardEpCompactDispatchIntraNodeKernel_##suffix(           \
      StandardEpCompactDispatchArgs args) {                                                \
    StandardEpCompactDispatchIntraNodeKernel_body<T>(args);                                \
  }
WRAP_STANDARD_EP_COMPACT_DISPATCH(f32, float)
WRAP_STANDARD_EP_COMPACT_DISPATCH(bf16, hip_bfloat16)

#define WRAP_STANDARD_EP_COMPACT_COMBINE(suffix, T)                                      \
  extern "C" __global__ void StandardEpCompactCombineIntraNodeKernel_##suffix(           \
      StandardEpCompactCombineArgs args) {                                                \
    StandardEpCompactCombineIntraNodeKernel_body<T>(args);                                \
  }
WRAP_STANDARD_EP_COMPACT_COMBINE(f32, float)
WRAP_STANDARD_EP_COMPACT_COMBINE(bf16, hip_bfloat16)

#define WRAP_STANDARD_EP_COMPACT_WEIGHTED_OUTPUT_BACKWARD(suffix, T)                       \
  extern "C" __global__ void StandardEpCompactWeightedOutputBackwardIntraNodeKernel_##suffix( \
      StandardEpCompactWeightedOutputBackwardArgs args) {                                  \
    StandardEpCompactWeightedOutputBackwardIntraNodeKernel_body<T>(args);                  \
  }
WRAP_STANDARD_EP_COMPACT_WEIGHTED_OUTPUT_BACKWARD(f32, float)
WRAP_STANDARD_EP_COMPACT_WEIGHTED_OUTPUT_BACKWARD(bf16, hip_bfloat16)

// Combine
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p, true, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _nop2p, false, false, false)
WRAP_ALL_TYPES_BOOL3(EpCombineIntraNodeKernel, _p2p_stdmoe, true, true, false)
WRAP_BOOL3(EpCombineIntraNodeKernel, bf16_nop2p_fp8cast, hip_bfloat16, false, false, true)
MORI_FP8_ANY(WRAP_BOOL4(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq, hip_bfloat16, false,
                        false, false, true))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block128_vec8,
                        hip_bfloat16, false, false, false, true, false, 128))
MORI_FP8_ANY(WRAP_BOOL6(EpCombineIntraNodeKernel, bf16_nop2p_fp8bwq_noweight_block256_vec8,
                        hip_bfloat16, false, false, false, true, false, 256))

// Convert (used with ENABLE_STANDARD_MOE_ADAPT)
extern "C" __global__ void mori_ConvertDispatchOutputKernel(ConvertDispatchOutputArgs args) {
  ConvertDispatchOutputDevice(args);
}
#define WRAP_CONVERT_COMBINE(suffix, T)                                                         \
  extern "C" __global__ void ConvertCombineInputKernel_##suffix(ConvertCombineInputArgs args) { \
    ConvertCombineInputDevice<T, true>(args);                                                   \
  }
WRAP_CONVERT_COMBINE(f32, float)
WRAP_CONVERT_COMBINE(bf16, hip_bfloat16)
MORI_FP8_OCP(WRAP_CONVERT_COMBINE(fp8_ocp, __hip_fp8_e4m3))
MORI_FP8_FNUZ(WRAP_CONVERT_COMBINE(fp8_fnuz, __hip_fp8_e4m3_fnuz))

#define WRAP_HOT_HELPER_PACK(suffix, T)                                      \
  extern "C" __global__ void HotHelperPackKernel_##suffix(HotHelperPackArgs args) { \
    HotHelperPackDevice<T>(args);                                            \
  }
WRAP_HOT_HELPER_PACK(f32, float)
WRAP_HOT_HELPER_PACK(bf16, hip_bfloat16)

#define WRAP_HOT_HELPER_COMPACT_PACK(suffix, T)                                      \
  extern "C" __global__ void HotHelperCompactPackKernel_##suffix(HotHelperCompactPackArgs args) { \
    HotHelperCompactPackDevice<T>(args);                                            \
  }
WRAP_HOT_HELPER_COMPACT_PACK(f32, float)
WRAP_HOT_HELPER_COMPACT_PACK(bf16, hip_bfloat16)
