
# Project scope

We want to improve the im2col index calculation by making the im2col transformations in CK Tile aware of he tiling

Im2Col transformations:

- projects/composablekernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp
- projects/composablekernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp
- projects/composablekernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp
- projects/composablekernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm_v2.hpp

The V2 transformation for the 2D convs is our placeholder tile-aware im2col transfromation.

The key integration points to tile aware im2col transformation are

- projects/composablekernel/include/ck_tile/core/tensor/tensor_coordinate.hpp
- projects/composablekernel/include/ck_tile/core/tensor/tensor_descriptor_tiled.hpp
- projects/composablekernel/include/ck_tile/core/tensor/tensor_view.hpp


# Building the code and tests

Use directory `projects/composablekernel/build-gfx950` and build with Ninja.

To run the configure step in the build directory

```bash
    cmake                                                                                             \
    -D CMAKE_PREFIX_PATH=/opt/rocm                                                                    \
    -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                         \
    -D CMAKE_BUILD_TYPE=Release                                                                       \
    -D GPU_TARGETS="gfx942"                                                                            \
    -G Ninja                                                                                          \
    ..
```

You need to to create an Enroot environment to run build and tests:

1. enroot create --name ck_dev_container $HOME/enroot/image_ck_dev.sqsh
2. enroot start --rw --env INSIDE_ENROOT=1 --mount $HOME:$HOME ck_dev_container
3. cd $HOME/git/ck3/projects/composablekernel/build-gfx950

The entry-point for running teh full end-to-end workflow is the CK Tile fwd conv example `projects/composablekernel/example/ck_tile/20_grouped_convolution/grouped_convolution_forward.cpp`
that can be built with command `ninja -j8 `

# Tile-aware im2col transformation

At the moment, we'll focus on fwd 2D convolutions. The key ideas for the tile-ware im2col transformation are listed in document `projects/composablekernel/docs/im2col_tile_analysis_general.md` 
which we should update when we get our ideas tested.

They key idea is to reduce the expensive im2col index calculations by precomputing the indices and utilizing the fact that the index calculation can be split into two separate parts for the input tensor.

# Profiling

Let's use the CK Tile fwd conv example (projects/composablekernel/build/bin/tile_example_grouped_conv_fwd) as the vehicle for running 
`rocprof-compute`. We'll define two test cases

- tile_example_grouped_conv_fwd -g=1 -n=32 -k=256 -c=256 -d=1 -h=100, -w=100 -z=1 -y=3 -x=3 -lpad_d=0 -lpad_h=1 -lpad_w=1 -rpad_d=0 -rpad_h=1 -rpad_w=1
- tile_example_grouped_conv_fwd -g=32 -n=32 -k=8 -c=8 -d=1 -h=100, -w=100 -z=1 -y=3 -x=3 -lpad_d=0 -lpad_h=1 -lpad_w=1 -rpad_d=0 -rpad_h=1 -rpad_w=1

and we'll need to run them with flag `-tiled_im2col` taking either value `0` (baseline) or `1` (tile-aware im2col).

We run full profiling step after which we can run the analysis step to compare the tile-aware im2col to the baseline. We are escpecially interested in the VALU and MFMA utilization as our working hypothesis is that tile-aware im2col should reduced index calculation operations and therefore reduce the number of VALU ops. The MFMA utilization should increase as we can load data faster. The pre-computation of the 
im2col indices might increase the register usage, which we need to look into in our profiling.

## rocprof-compute instructions

See the instructions for running `rocprof-compute` from here: https://github.com/ROCm/composable_kernel/blob/vpietila/ck-profiling-documentation/docs/profiling/rocprof-compute.md
