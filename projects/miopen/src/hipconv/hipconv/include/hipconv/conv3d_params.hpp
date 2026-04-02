#pragma once

namespace hipconv
{

// Parameters for a 3D convolution (forward only).
//
// Layout: NDHWC input, KTRSC weights (T=filter depth), NOPQK output.
// Constraint: c == k (same channel count in and out), groups == 1.
// Supported: unit stride, unit dilation, pad_d == 0.
struct Conv3dParams
{
    int n;                              // batch size
    int id, ih, iw;                    // input  depth / height / width
    int c;                             // input channels
    int k;                             // output channels (== c)
    int kd = 3, kh = 3, kw = 3;       // filter size
    int pad_d = 0, pad_h = 1, pad_w = 1; // padding (depth padding must be 0)
    int stride_d = 1, stride_h = 1, stride_w = 1;
    int dilation_d = 1, dilation_h = 1, dilation_w = 1;
    int od, oh, ow;                    // output depth / height / width

    void compute_output_size()
    {
        od = (id + 2 * pad_d - kd) / stride_d + 1;
        oh = (ih + 2 * pad_h - kh) / stride_h + 1;
        ow = (iw + 2 * pad_w - kw) / stride_w + 1;
    }
};

} // namespace hipconv
