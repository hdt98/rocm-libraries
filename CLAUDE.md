We are building a pure HIP based direct 3D convolution variant for cases where either the number of input channels (C) or the number of output channel (K) is low.
In this case, we have non-grouped onvolution. We are currently looking into a 3D forward convolution problem characterized by the following parameters

- Di, Hi, Wi: Input (feature map) depth, width, and height
- Z, Y, X: Filter (weight) depth, height, and width
- C: number of input channels
- K: number of output channels
- N: batch size

The source code is located at `projects/miopen/src/hipconv` and we have an architecture document describing the design: 
`projects/miopen/src/hipconv/hipconv_3d_design.md`. 

Currently the `hipconv` project contains specialized kernels for 2D convolutions that have a low number of channels per group.
We want to extend the direct convolution approach to 3D forward convolutions that have the following input/outptu channel combinations
- C = 3 and K = 96
- C = 96 and K = 3
- C = 16 and K = 384
These are shapes where the im2col based approaches are struggling to get good performance. The filter size we are interested in is 3x3x3.
We have unit stride and dilation. The padding can be either
- Unit padding in height and width directions, no padding in depth direction.
- No padding.