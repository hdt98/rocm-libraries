// Test: Does binary search dispatch with constexpr indices optimize away?
// This tests the CORE CLAIM: runtime indexing with constexpr offsets = zero overhead

#include <hip/hip_runtime.h>

using index_t = int32_t;

template <index_t I>
struct Number {
    static constexpr index_t value = I;
};

// Tuple-like structure (mimics CK's StaticallyIndexedArray)
template <typename T, index_t N>
struct MyTuple;

template <typename T>
struct MyTuple<T, 0> {};

template <typename T, index_t N>
struct MyTuple {
    T head;
    MyTuple<T, N-1> tail;

    template <index_t I>
    __device__ const T& get(Number<I>) const {
        if constexpr (I == 0) return head;
        else return tail.get(Number<I-1>{});
    }

    template <index_t I>
    __device__ T& get(Number<I>) {
        if constexpr (I == 0) return head;
        else return tail.get(Number<I-1>{});
    }

    // Runtime access via binary search
    template <index_t Begin, index_t End>
    __device__ const T& runtime_get(index_t i, Number<Begin>, Number<End>) const {
        if constexpr (End - Begin == 1) {
            return get(Number<Begin>{});
        } else {
            constexpr index_t Mid = (Begin + End) / 2;
            if (i < Mid) {
                return runtime_get(i, Number<Begin>{}, Number<Mid>{});
            } else {
                return runtime_get(i, Number<Mid>{}, Number<End>{});
            }
        }
    }

    __device__ const T& operator[](index_t i) const {
        return runtime_get(i, Number<0>{}, Number<N>{});
    }
};

// BASELINE: Compile-time indexing (8 instantiations)
__global__ void kernel_compile_time(const float* src, float* dst)
{
    MyTuple<float, 8> buf;

    // Initialize
    buf.get(Number<0>{}) = src[0];
    buf.get(Number<1>{}) = src[1];
    buf.get(Number<2>{}) = src[2];
    buf.get(Number<3>{}) = src[3];
    buf.get(Number<4>{}) = src[4];
    buf.get(Number<5>{}) = src[5];
    buf.get(Number<6>{}) = src[6];
    buf.get(Number<7>{}) = src[7];

    if (threadIdx.x == 0) {
        // Manual unroll with Number<>
        dst[0] = buf.get(Number<0>{});
        dst[1] = buf.get(Number<1>{});
        dst[2] = buf.get(Number<2>{});
        dst[3] = buf.get(Number<3>{});
        dst[4] = buf.get(Number<4>{});
        dst[5] = buf.get(Number<5>{});
        dst[6] = buf.get(Number<6>{});
        dst[7] = buf.get(Number<7>{});
    }
}

// OPTIMIZED: Runtime indexing with constexpr indices
__global__ void kernel_runtime(const float* src, float* dst)
{
    MyTuple<float, 8> buf;

    // Initialize with runtime loop
    #pragma unroll
    for (index_t i = 0; i < 8; ++i) {
        // This creates a constexpr i value per unrolled iteration
        buf.get(Number<0>{}) = src[0]; // Hack: can't write to buf[i]
        // Actually, let me use compile-time init
    }
    buf.get(Number<0>{}) = src[0];
    buf.get(Number<1>{}) = src[1];
    buf.get(Number<2>{}) = src[2];
    buf.get(Number<3>{}) = src[3];
    buf.get(Number<4>{}) = src[4];
    buf.get(Number<5>{}) = src[5];
    buf.get(Number<6>{}) = src[6];
    buf.get(Number<7>{}) = src[7];

    if (threadIdx.x == 0) {
        // Runtime loop with #pragma unroll - creates constexpr i per iteration
        #pragma unroll
        for (index_t i = 0; i < 8; ++i) {
            dst[i] = buf[i];  // Runtime indexing!
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
