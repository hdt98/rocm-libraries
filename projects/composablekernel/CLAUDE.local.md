# Personal Development Notes

## Problem statement

We are interested in writing specialized kernels forward, backward weight, and backward data convolutions. The main interest is currently 2D,
but we can later extend the work to 3D convolutions.

The 2D convolution problem is described by
- N = input batch size
- G = number of conv groups
- C = number of input filters per group
- K = number of output filters per group
- Hi = input height (corresponds y-direction)
- Wi = input width (corresponds to x-direction)
- Y = filter height
- X = filter width
- Ho = output height (corresponds to y-direction)
- Wo = output width (corresponds to x-direction)
- Sy = stride in y-direction 
- Sx = stride in x-direction
- Dy = dilation in y-direction
- Dx = dilation in x-direction  
- LPy = left padding in y-direction 
- Lpx = left padding in x-direction 
- RPy = right padding in y-direction 
- RPx = right padding in x-direction

We are interested in memory bound cases where number of input/output channels per group is small (4, 8, 16) and K=C. 
Moreover, we are interested in filter size 3 x 3 (= X x Y). The number of conv groups is typically G = 32. 
We typically have unit stride, dilation, and padding.

We are targeting AMD GPUs, particularly gfx950 architecture (MI350X and MI355X) GPUs and we are using HIP C++ programming.
The starting point is the Composable Kernel library (CK), which applies C++ template metaprogramming to offload computations to compile-time.

We can assume channels first-layout for all tesors, i.e.,
- Input[G, N, C, Hi, Wi] for input (tensor I)
- Weight[G, K, C, Y, X] for weight (tensor W)
- Output[G, N, K, Ho, Wo] for output (tensor O)

## Specialized approach for small number of chanels per group

CK and CK Tile libraries implement an implicit GEMM algorithm to compute convolutions. You can find the conv implementation from

projects/composablekernel/include/ck_tile/ops/grouped_convolution

This directory has three parts:

- kernel: Actual kernel implementations for the fwd, bwd weigth, and bwd data directions
- pipeline: GEMM pipelines that are used in the implicit-GEMM algorithm. Currently there's only a "universal" pipeline with A and B matrices in global memory and C matrix in registers.
- utils: This directory contains the transformations that map the multidimensional input, weight, and output tensors into matric A, B, and C in the corresponding GEMM problem.

The implicit GEMMalgorithm uses im2col algorithm to construct mapping from the conv problem to GEMM problem. The full matrix is never constructed, only the tiles processed by
workgroups are constructed. The global memory loads are vectorized, which saves some index calculation, but nevertheless the index calculations in im2col are taking lot of VALU ops.

The im2col and implicit GEMM approach does not work well for the memory bound cases where C and K are small. For this reason, we want to investigate if the im2win approach would yield better
results. The original reference to im2win on GPU is this paper
- https://arxiv.org/abs/2306.14316

As a first step, we need to implement the im2win transformation as described in the original article.

### im2win transformation

We need to transform the original input tensor I to the im2win input tensor I'. 
The relation between the two tensor is denoted by equation

I'(g, i, r, m, k * Y + u) = I(g, i, r, m + u, n + v),

where 

g = 0, ..., G - 1
m = 0, ..., Ho - 1
n = 0, ..., Wo - 1
u = 0, ..., Y - 1
v = 0, ..., X - 1
i = 0, ..., N - 1
r = 0, ..., C - 1
k = 0, ..., Wi - 1

this gives mapping I' = im2win(I)

The original convolution can be written as

O(g, i,j,m,n) = \sum_{j=0}^{C-1}\sum_{m=0}^{Y-1}\sum_{n=0}^{X-1} [I(g, i, j, m * s + u, m * s + v) * W(g,j,r,u,v)]

where 

j = 0, ..., K - 1

### Tling strategy

Similar to the implicit GEMM, we use block tile of size (Mb, Nb, Kb), which divide tensor I' and W into tiles of size Mb x Kb and kB x Nb.
At thread level, we divide the problem into Mt and Nt to utilize the efficiently the MFMA instructions for computing the Mt x Nt tile 
of the output tensor O.

### Pseudo code implementation of the algorithm

Input: Input tensorI, Weight tensor W, Stride s = Sx = Sy 
Output:Output tensor O 
Im2win Tensor: I' = im2win(I,W,s) 
Dimensions: M'=K, N'=N × Ho × Wo, K'= C × Y × X 
Number of blocks: M'/Mb × N'/Nb 
Number of threads per block: Mb/Mt × Nb/Nt 
Registers: Ri[2][Nt] (I' tensor), Rw[2][Mt] (W tensor), Ro[Mt×Mt] //doublebuffer 
Shared memory: Si[2][Kb×Nb] (I' tensor), Sw[2][Mb×Kb] (W tensor) //doublebuffer

Si[0][kB×nB] load ←−−− kB×nB of I'(0,by) 
Sw[0][mB×kB] load ←−−− mB×kB of W(bx,0) 
__syncthreads() 
Ri[0][nT] vec_load ←−−−−−−− nT of Si[0][0×nB] 
Rw[0][mT] vec_load ←−−−−−−− mT of Sw[0][mB×0] 

for kk = 0 to C×Y×X/Kb − 1 do 
  for k'= 1 to Kfb −1 do 
    Ri[load][nT] vec_load ←−−−−−−− nT of Si[store][k' × nB] //prefetching 
    Rw[load][mT] vec_load ←−−−−−−− mT of Sw[store][mB × k'] //prefetching 
    Ro[mT × nT] += Rw[store][mT] × Ri[store][nT] // micro-kernel i.e., use MFMA to compute the Mt × Nt tile
  if kk != C×Y×X/Kb − 1 then 
    Si[load][kB × nB] load ←−−− kB×nB of I'(kk+1,by) //prefetching 
    Sw[load][mB × kB] load ←−−− mB×kB of W(bx,kk+1)  //prefetching 
    __syncthreads()
    
  Ri[0][nT] vec_load ←−−−−−−− nT of Si[store][0 × nB] 
  Rw[0][mT] vec_load ←−−−−−−− mT of Sw[store][mB × 0] 
  RO[mT × nT] += Ri[1][nT] × Ri[1][nT] //micro-kernel, i.e., use MFMA to compute the Mt × Nt tile

O(bx,by) store ←−−− RO[mT × nT]

## CK Tile resources for implementing the im2win tensor descriptors and memory access

Tile distribution: mapping between data and threads
- projects/composablekernel/include/ck_tile/core/tensor/tile_distribution.hpp
- projects/composablekernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp
- Can be used to describe how the 

Tensor descriptor: How tensor look like in memory - mapping between visible and hidden indices
- projects/composablekernel/include/ck_tile/core/tensor/tensor_descriptor.hpp
- Can be used to express mapping from I to I'.

Static distributed tensor: How to distribute tensor over threads
- projects/composablekernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp

Tensor view: abstraction of tensor with respect to the underlying memory
- projects/composablekernel/include/ck_tile/core/tensor/tensor_view.hpp
- Vectorized access to the tensor elements.

Tile window: partial (windowed) access to the GPU memory
- projects/composablekernel/include/ck_tile/core/tensor/tile_window.hpp
- This can be used to access the memory corresponding to a given tile.

Additional resources can be found under directory `projects/composablekernel/include/ck_tile/core/tensor`.

## CK Tile resources for using MFMA ops for tharead level computations

At warp/wavefront level, we can use MFMA for the micro-kernel ops that were defined in the pseudo-code
- projects/composablekernel/include/ck_tile/ops/gemm/warp/warp_gemm.hpp
- projects/composablekernel/include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp
- projects/composablekernel/include/ck_tile/ops/gemm/warp/warp_gemm_impl.hpp

An example on how the GEMM pipelines are defined at tile-level, one can refer to `projects/composablekernel/include/ck_tile/ops/gemm/pipeline`.

## Testing

Each new component/pipeline should be unit tested such that the unit tests can serve as additional documentation.
GTest framework should be used for testing. The existing CK Tile unit tests can be found from directory `projects/composablekernel/test/ck_tile`

## Progress

Let's start from forward convolution and record progress here.