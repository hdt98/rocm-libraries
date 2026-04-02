// Test: Verify REAL CK StaticBuffer runtime indexing works
// This uses the actual StaticallyIndexedArray (Tuple-based), not a toy example

#include <hip/hip_runtime.h>
#include <cstdint>

using index_t = int32_t;

template <index_t I>
struct Number {
    static constexpr index_t value = I;
};

// Import REAL CK Tuple implementation (simplified version)
#include "include/ck/utility/tuple.hpp"
#include "include/ck/utility/statically_indexed_array.hpp"
#include "include/ck/utility/static_buffer.hpp"
#include "include/ck/utility/amd_address_space.hpp"

using namespace ck;

// Test kernel using REAL CK StaticBuffer
__global__ void test_real_ck_buffer(const float* src, float* dst)
{
    // Real CK StaticBuffer (uses StaticallyIndexedArray which is Tuple-based)
    StaticBuffer<AddressSpaceEnum::Vgpr, float, 8, false> buf;

    // Initialize with compile-time indexing (existing API)
    buf(Number<0>{}) = src[0];
    buf(Number<1>{}) = src[1];
    buf(Number<2>{}) = src[2];
    buf(Number<3>{}) = src[3];
    buf(Number<4>{}) = src[4];
    buf(Number<5>{}) = src[5];
    buf(Number<6>{}) = src[6];
    buf(Number<7>{}) = src[7];

    if (threadIdx.x == 0) {
        // Read with NEW runtime indexing API (O(1) direct data_[i] access)
        #pragma unroll
        for (index_t i = 0; i < 8; ++i) {
            dst[i] = buf.At(i);  // O(1) runtime indexing!
        }
    }
}

int main()
{
    float *d_src, *d_dst, h_src[8], h_dst[8];

    // Initialize
    for (int i = 0; i < 8; ++i) h_src[i] = float(i + 1);

    hipMalloc(&d_src, 8 * sizeof(float));
    hipMalloc(&d_dst, 8 * sizeof(float));
    hipMemcpy(d_src, h_src, 8 * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel
    test_real_ck_buffer<<<1, 256>>>(d_src, d_dst);

    // Get results
    hipMemcpy(h_dst, d_dst, 8 * sizeof(float), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    // Verify
    bool pass = true;
    for (int i = 0; i < 8; ++i) {
        if (h_dst[i] != float(i + 1)) {
            printf("FAIL: h_dst[%d] = %f, expected %f\n", i, h_dst[i], float(i + 1));
            pass = false;
        }
    }

    if (pass) {
        printf("✅ PASS: Real CK StaticBuffer runtime indexing works!\n");
    }

    hipFree(d_src);
    hipFree(d_dst);
    return pass ? 0 : 1;
}
