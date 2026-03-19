#pragma once

namespace hipconv
{
enum class DataType
{
    fp16,
    bf16,
    fp32,
    fp8,
    bf8,
};

// The direction of the operation (foreward (Fprop) or gradient type).
enum class Direction
{
    // Foreward
    Fprop = 1,

    // Input gradient
    Dgrad = 2,

    // Weights gradient
    Wgrad = 4
};

// The tensor order (i.e. layout).
enum TensorOrder
{
    // Channels first
    NCHW,

    // Channels last
    NHWC
};

// Define the parameters for a conv2d layer.
struct Conv2dParams
{
    Direction direction = Direction::Fprop;
    int n;                              // batch size
    int h, w;                           // input size
    int c;                              // input channels
    int k;                              // output channels
    int kh, kw;                         // filter size
    int pad_h = 1, pad_w = 1;           // padding
    int stride_h = 1, stride_w = 1;     // stride
    int dilation_h = 1, dilation_w = 1; // dilation
    int p, q;                           // output size
    int groups        = 1;              // number of channel groups
    DataType in_type  = DataType::fp16;
    DataType wei_type = DataType::fp16;
    DataType out_type = DataType::fp16;
    TensorOrder order = TensorOrder::NHWC;

    // Overwrite (p, q) with output size derived from input size, padding, and stride.
    void compute_output_size()
    {
        p = (h + 2 * pad_h - kh) / stride_h + 1;
        q = (w + 2 * pad_w - kw) / stride_w + 1;
    }

    bool is_valid() const
    {
        if(n <= 0 || h <= 0 || w <= 0 || c <= 0 || k <= 0 || kh <= 0 || kw <= 0)
        {
            return false;
        }
        if(groups <= 0 || c % groups != 0 || k % groups != 0)
        {
            return false;
        }
        return true;
    }

    int channels_per_group() const { return c / groups; }
    int filters_per_group() const { return k / groups; }
};

} // namespace hipconv
