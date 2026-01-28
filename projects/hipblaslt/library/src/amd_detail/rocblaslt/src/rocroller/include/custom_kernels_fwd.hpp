#pragma once

using CustomKernelPtr = void (*)(void* D, void* C, void* A, void* B, float alpha, float beta,
    unsigned int stride_D0, unsigned int stride_D1,
    unsigned int stride_C0, unsigned int stride_C1,
    unsigned int stride_A0, unsigned int stride_A1,
    unsigned int stride_B0, unsigned int stride_B1,
    unsigned int m, unsigned int n, unsigned int k,
    void* scaleA, void* scaleB,
    unsigned int stride_scaleA0, unsigned int stride_scaleA1,
    unsigned int stride_scaleB0, unsigned int stride_scaleB1,
    int log2_split_k);
