
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
    -D GPU_TARGETS="gfx50"                                                                            \
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
