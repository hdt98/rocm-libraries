// Single-instance opt3 test — matches doc 13 setup exactly.
// BS=256, M=128, N=128, KPerBlock=32, AK1=8, BK1=8, Default
// CDEBlockTransferScalarPerVector_NPerBlock = 1 (noshuffle)

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"
#include "common.hpp"

using OutDataType      = BF16;
using WeiDataType      = BF16;
using AccDataType      = FP32;
using CShuffleDataType = BF16;
using DsDataType       = ck::Tuple<>;
using InDataType       = BF16;

using OutLayout = ck::tensor_layout::convolution::GNHWK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using DsLayout  = ck::Tuple<>;
using InLayout  = ck::tensor_layout::convolution::GNHWC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = PassThrough;

// clang-format off
using DeviceConvInstance = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
// ######| NDimSpatial|   ALayout|   BLayout|   DsLayout|  ELayout| AData| BData| AccData| CShuffle| DsData| EData| AElemOp| BElemOp| CDEElemOp| ConvBwdSpec| DoPad| DoPad| NumGemmK| Block| MPer| NPer| KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffleMXdl| CShuffleNXdl| CDEBlockTransfer| CDEBlockTransfer|
         < NDimSpatial, OutLayout, WeiLayout,   DsLayout, InLayout, OutDataType, WeiDataType, AccDataType, CShuffleDataType, DsDataType, InDataType, OutElementOp, WeiElementOp, InElementOp, ConvBwdDataDefault, true, true, 1, 256, 128, 128, 32, 8, 8, 32, 32, 2, 2, S<4, 64, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 1, S<4, 16, 1>, S<0, 2, 1>, S<0, 2, 1>, 1, 8, 8, 1, 1, 1, S<1, 32, 1, 8>, 1>;
// clang-format on

#include "run_grouped_conv_bwd_data_example.inc"

int main(int argc, char* argv[]) { return run_grouped_conv_bwd_data_example(argc, argv); }
