# MAGMA Source Files for rocSOLVER

This directory contains source files from [MAGMA](https://icl.utk.edu/magma/)
(Matrix Algebra on GPU and Multicore Architectures), licensed under BSD-3-Clause
(see MAGMA_LICENSE). These provide GPU-accelerated implementations for:

- **geev**: Non-symmetric eigenvalue decomposition using GPU-accelerated
  Hessenberg reduction (gehrd) and orthogonal matrix generation (orghr)

The BLAS wrappers use rocBLAS directly (converted from MAGMA's hipBLAS wrappers).
