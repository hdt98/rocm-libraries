// Test: O(1) runtime buffer indexing with StaticallyIndexedArray_v2
// This tests: runtime indexing via direct data_[i] access produces identical assembly
// to compile-time Number<I> indexing
//
// This is a STANDALONE test using a minimal reimplementation of StaticBuffer
// to verify the O(1) approach. The cluster build verifies the real CK code.

#include <hip/hip_runtime.h>

using index_t = int32_t;

template <index_t I>
struct Number {
    static constexpr index_t value = I;
    __host__ __device__ constexpr operator index_t() const { return I; }
};

// Minimal O(1) buffer implementation (mirrors StaticallyIndexedArray_v2 + StaticBuffer)
template <typename T, index_t N>
struct O1Buffer {
    T data_[N];

    // Compile-time read access (existing API)
    template <index_t I>
    __host__ __device__ constexpr const T& operator[](Number<I>) const { return data_[I]; }

    // Compile-time write access (existing API)
    template <index_t I>
    __host__ __device__ constexpr T& operator()(Number<I>) { return data_[I]; }

    // O(1) runtime read access - DIRECT ARRAY ACCESS
    __host__ __device__ __forceinline__ const T& At(index_t i) const { return data_[i]; }

    // O(1) runtime write access - DIRECT ARRAY ACCESS
    __host__ __device__ __forceinline__ T& At(index_t i) { return data_[i]; }
};

// BASELINE: Compile-time indexing with Number<>
__global__ void kernel_compile_time(const float* src, float* dst)
{
    O1Buffer<float, 8> buf;

    // Initialize with compile-time indices
    buf(Number<0>{}) = src[0];
    buf(Number<1>{}) = src[1];
    buf(Number<2>{}) = src[2];
    buf(Number<3>{}) = src[3];
    buf(Number<4>{}) = src[4];
    buf(Number<5>{}) = src[5];
    buf(Number<6>{}) = src[6];
    buf(Number<7>{}) = src[7];

    if (threadIdx.x == 0) {
        // Manual unroll with Number<>
        dst[0] = buf[Number<0>{}];
        dst[1] = buf[Number<1>{}];
        dst[2] = buf[Number<2>{}];
        dst[3] = buf[Number<3>{}];
        dst[4] = buf[Number<4>{}];
        dst[5] = buf[Number<5>{}];
        dst[6] = buf[Number<6>{}];
        dst[7] = buf[Number<7>{}];
    }
}

// OPTIMIZED: Runtime indexing with O(1) direct data_[i] access
__global__ void kernel_runtime(const float* src, float* dst)
{
    O1Buffer<float, 8> buf;

    // Initialize with compile-time indices (same as baseline)
    buf(Number<0>{}) = src[0];
    buf(Number<1>{}) = src[1];
    buf(Number<2>{}) = src[2];
    buf(Number<3>{}) = src[3];
    buf(Number<4>{}) = src[4];
    buf(Number<5>{}) = src[5];
    buf(Number<6>{}) = src[6];
    buf(Number<7>{}) = src[7];

    if (threadIdx.x == 0) {
        // Runtime loop with #pragma unroll - compiler unrolls with constexpr i
        #pragma unroll
        for (index_t i = 0; i < 8; ++i) {
            dst[i] = buf.At(i);  // O(1) runtime indexing via data_[i]
        }
    }
}

int main() {
    float *d_src, *d_dst1, *d_dst2;
    hipMalloc(&d_src, 8 * sizeof(float));
    hipMalloc(&d_dst1, 8 * sizeof(float));
    hipMalloc(&d_dst2, 8 * sizeof(float));

    kernel_compile_time<<<1, 256>>>(d_src, d_dst1);
    kernel_runtime<<<1, 256>>>(d_src, d_dst2);

    hipDeviceSynchronize();
    hipFree(d_src);
    hipFree(d_dst1);
    hipFree(d_dst2);
    return 0;
}
