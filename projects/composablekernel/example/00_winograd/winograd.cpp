#include "ck/ck.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/utility/amd_xdlops.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/reduction_operator.hpp"

#include "debug_utils.h"

#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "mio_conv_args.h"

using namespace ck;
#include "winograd.h"

template <typename DeviceConv>
__global__ void k_entry(typename DeviceConv::Args args)
{
    DeviceConv{}.run(args);
}

static constexpr ck::index_t NDimSpatial = 2;

using InpLayout = ck::tensor_layout::convolution::NGCHW;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using OutLayout = ck::tensor_layout::convolution::NGKHW;

using InpDataType = half_t;
using WeiDataType = half_t;
using OutDataType = half_t;

int main(int argc, char* argv[])
{
    ParseHostArgs(argc, argv);
    HIP_CHECK_ERROR(hipSetDevice(1));

    ck::utils::conv::ConvParam conv_param{NDimSpatial,
                                          G,
                                          N,
                                          K / G,
                                          C / G,
                                          {R, S},
                                          {H, W},
                                          {conv_stride_h, conv_stride_w},
                                          {dilation_h, dilation_w},
                                          {pad_h, pad_w},
                                          {pad_h, pad_w}};

    const auto inp_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InpLayout>(conv_param);
    const auto wei_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InpDataType> inp(inp_desc);
    Tensor<WeiDataType> wei(wei_desc);
    Tensor<OutDataType> out(out_desc);

    constexpr int modv = 10;
    inp.GenerateTensorValue([&](auto... is) { return inp.GetOffsetFromMultiIndex(is...) % modv; });

    {
        HostTensorDescriptor whack =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
                ck::tensor_layout::convolution::GKCYX>(conv_param);
        wei.GenerateTensorValue(
            [&](auto... is) { return whack.GetOffsetFromMultiIndex(is...) % modv; });
    }

    out.SetZero();

    DeviceMem inp_device_buf(sizeof(InpDataType) * inp.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    inp_device_buf.ToDevice(inp.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    using DeviceConv = WinogradConv;

    auto args = DeviceConv::Args{
        static_cast<InpDataType*>(inp_device_buf.GetDeviceBuffer()),
        static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()), // buffer pointers
        N,
        C / G,
        H,
        W,
        K / G,
        G,
        R,
        S,
        pad_h,
        pad_w,
        0,
        0, // out
        0,
        0,
        0, // d_stride
        0,
        0,
        0, // f_stride
        0,
        0,
        0 // 0_stride
    };

    const auto grid_size  = DeviceConv::GetGridSize(args);
    const auto block_size = DeviceConv::GetBlockSize(args);

    auto stream_config = StreamConfig{nullptr, // stream_id
                                      true,    // time_kernel
                                      0,       // log_level

                                      0, // cold_niters
                                      1, // nrepeat

                                      //   5,  // cold_niters
                                      //   20, // nrepeat

                                      true, // flush_cache
                                      1};   // rotating_count

    const auto kernel = k_entry<DeviceConv>;
    float ave_time = launch_and_time_kernel(stream_config, kernel, grid_size, block_size, 0, args);

    out_device_buf.FromDevice(out.mData.data());

    printf("Elapsed: %f ms\n", ave_time);

    out_device_buf.FromDevice(out.mData.data());

    // PrintTensor(inp);
    // PrintTensor(wei);
    // PrintTensor(out);

    constexpr bool do_verification = true;
    if(do_verification)
    {
        Tensor<OutDataType> out_host(out_desc);
        using InpElementOp = ck::tensor_operation::element_wise::PassThrough;
        using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
        using OutElementOp = ck::tensor_operation::element_wise::UnaryConvert;

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
            NDimSpatial,
            InpDataType,
            WeiDataType,
            OutDataType,
            InpElementOp,
            WeiElementOp,
            ck::tensor_operation::element_wise::PassThrough>();

        auto ref_invoker = ref_conv.MakeInvoker();
        auto ref_argument =
            ref_conv.MakeArgument(inp,
                                  wei,
                                  out_host,
                                  conv_param.conv_filter_strides_,
                                  conv_param.conv_filter_dilations_,
                                  conv_param.input_left_pads_,
                                  conv_param.input_right_pads_,
                                  InpElementOp{},
                                  WeiElementOp{},
                                  ck::tensor_operation::element_wise::PassThrough{});

        ref_invoker.Run(ref_argument);
        // cde_elementwise
        out_host.ForEach([&](auto&, auto idx) { OutElementOp{}(out_host(idx), out_host(idx)); });

        // PrintTensor(out_host);

        auto valid = ck::utils::check_err(
            out.mData, out_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

        if(valid)
            std::cout << "Test pass\n";

        return 0;
    }
}