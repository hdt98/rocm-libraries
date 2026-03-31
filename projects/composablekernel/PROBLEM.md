# WAN Forward Convolution Tuning

## Branch

`users/vpietila/ck/wan-fwd-conv-tuning` (based on `features/grouped-conv-perf-uplift`)

## Goal

Tune and benchmark grouped forward convolution kernels in ComposableKernel for performance
on gfx950 (MI350), targeting a WAN (video generation model) workload that makes heavy use
of grouped 2D and 3D convolutions.

## What Has Been Done

- Stripped `profiler/src/CMakeLists.txt` down to compile **only** grouped convolution forward ops
  and device instances, dramatically reducing ckProfiler build time.


## Profiling Grouped Forward Convolution

```
./bin/ckProfiler grouped_conv_fwd <data_type> <layout> <index_type> <verify> <init> <print> <time> \
    <G> <N> <C> <K> <Y> <X> <Hi> <Wi> <conv_stride_h> <conv_stride_w> \
    <dilation_h> <dilation_w> <pad_h> <pad_w> <out_pad_h> <out_pad_w>
```

**Data types:** 0=f32, 1=f16, 2=bf16, 3=int8, 4=f8, 5=bf8/f8, 6=f8/bf8, 7=bf8/f8, 8=tf32

**Layouts:**
- 0: `GNHWC_GKYXC_GNHWK`
- 1: `NHWGC_GKYXC_NHWGK`
- 2: `NGCHW_GKYXC_NGKHW`
- 3: `NGCHW_GKCYX_NGKHW`

**Useful flags:**
- `--list-instances` — list all valid kernel instances without running
- `--instance <id>` — run only the instance with the given 0-based index

## Instance Files

Instantiated kernels for grouped 2D/3D convolution live in:

```
library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/xdl/
library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd/xdl/
```

Key naming patterns: 
- `device_grouped_conv2d_fwd_xdl_<layout>_<dtype>_instance.cpp`
- `device_grouped_conv3d_fwd_xdl_<layout>_<dtype>_instance.cpp`

Variants of note:
- `*_nhwgc_gkyxc_nhwgk_*` — NHWGC layout (most common for WAN)
- `*_16x16_instance` — smaller tile size, better for small-C or small-K problems
- `*_nongroup_ported_*` — ported from non-grouped implementation

## Tensor Layout Naming

CK uses a compact notation for tensor dimension ordering:

| Letter | Dimension |
|--------|-----------|
| G      | Groups    |
| N      | Batch     |
| H/W    | Height/Width (spatial) |
| C      | Input channels (per group) |
| K      | Output channels (per group) |
| Y/X    | Filter height/width |

Example: `NHWGC` means the input tensor has dimensions `[N, H, W, G, C]`.

## Problem statement

We have a set of 2d and 3D forward convolution shapes in the form of MIOpenDriver commands
- projects/composablekernel/build-gfx950/data/i2v_miopendriver_commands_unique.txt
- projects/composablekernel/build-gfx950/data/t2v_miopendriver_commands_unique.txt
For Wan2.2-14B image-to-video and text-to-video tasks.

We would like to optimize the kernel instances (c.f. section Instance Files) to get the best possible 
performance for these shapes.

In order to run the CK profiler for these shapes, one can use a script that converts the MIOpenDriver commands into 
CK Profiler commands: projects/composablekernel/script/convert_miopen_driver_to_profiler.py
The script can run a single MIOpenDriver command, or a batch of the MIOpenDriver commands from a file.
The files listed above are intended for the batch mode of the script.

The shapes have been benchmarked using the current baseline. The results are located here

- projects/composablekernel/build-gfx950/data/i2v_baseline.txt
- projects/composablekernel/build-gfx950/data/t2v_baseline.txt

We should first look at the shapes that have the lowest performance interms of the TFLOPs count. 
We should try to identify classes of shapes that are not performing well and try to add instances 
that would increase the performance. 

### Testing and profiling an individual kernel instance

In order to isolate the kernel instance and profile it in isolation, we can use the CK fwd convolution example

- projects/composablekernel/example/30_grouped_conv_fwd_multiple_d/grouped_conv_fwd_xdl_bf16.cpp
- projects/composablekernel/example/30_grouped_conv_fwd_multiple_d/run_grouped_conv_fwd_example.inc

The .inc file contains the kernel instance definition that can be used to test and profile an individual kernel 
for a given convolution. The fwd conv example can also be used to verify that the results match with the reference 
implementation output. It is advisable to use GPU verifycation (value "2" for the corresponding cmd line parameter) 
to speed-up verification.

The CK fwd conv example also provides an entry point for profiling the kernel with 
`rockprof-compute` profiler. Instruction for running the rocprof-compute profiler can be found from here:
https://github.com/ROCm/composable_kernel/blob/vpietila/ck-profiling-documentation/docs/profiling/rocprof-compute.md

### Logging 
Sometime a newly constructed kernel is not applicable to the given conv problem. In this case, we set env variable 
CK_LOGGING=1 and we get log messages that explain why the applicability check failed. This will help us to modify 
the kernel instance template parameters such that the new kernel is applicable and improves preformance. 