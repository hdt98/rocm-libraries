# Quick Start: Unified Tensor Wrapper

## The Problem

**Different APIs for the same operations:**

### Creating Descriptors

```cpp
// CK_Tile
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, K),
    make_tuple(stride, 1),
    number<VectorSize>{},
    number<1>{});

// MINT
auto desc = make_aliased_naive_packed_tensor_descriptor(
    aliases<"M", "K">{},  // COMPILE-TIME STRINGS!
    alias<"Offset">{},
    {M, K});
```

### Creating Views

```cpp
// CK_Tile
auto view = make_tensor_view<address_space_enum::global>(ptr, desc);

// MINT
auto view = make_tensor_view(desc, make_global_memory_view(ptr, size));
```

## The Solution

**One unified API for both backends:**

```cpp
#include "simple_tensor_descriptor_wrapper.hpp"

using namespace unified_wrapper;

// Step 1: Choose backend (just change the tag type!)
using DescWrapper = TensorDescriptorWrapper<CKTileBackend, float, 4>;
using ViewWrapper = TensorViewWrapper<CKTileBackend, float, 4>;
// or
// using DescWrapper = TensorDescriptorWrapper<MintBackend, float>;
// using ViewWrapper = TensorViewWrapper<MintBackend, float>;

// Step 2: Create descriptor (same API for both!)
auto desc = DescWrapper::create_a_descriptor(
    M, K,        // dimensions
    K,           // stride
    true         // row-major
);

// Step 3: Create view (same API for both!)
auto view = ViewWrapper::create_view(a_ptr, desc);

// Done! Works with both backends.
```

## Complete Example

```cpp
template <typename BackendTag>
__global__ void my_kernel(const float* a, const float* b, float* c,
                          index_t M, index_t N, index_t K)
{
    // Define wrappers
    using DescWrapper = TensorDescriptorWrapper<BackendTag, float, 4>;
    using ViewWrapper = TensorViewWrapper<BackendTag, float, 4>;

    // Create descriptors
    auto a_desc = DescWrapper::create_a_descriptor(M, K, K, true);
    auto b_desc = DescWrapper::create_b_descriptor(K, N, N, true);
    auto c_desc = DescWrapper::create_c_descriptor(M, N, N, true);

    // Create views
    auto a_view = ViewWrapper::create_view(a, a_desc);
    auto b_view = ViewWrapper::create_view(b, b_desc);
    auto c_view = ViewWrapper::create_view(c, c_desc);

    // Use views for computation...
}

// Launch with CK_Tile
my_kernel<CKTileBackend><<<grid, block>>>(a, b, c, M, N, K);

// Launch with MINT
my_kernel<MintBackend><<<grid, block>>>(a, b, c, M, N, K);
```

## API Cheat Sheet

### Descriptor Creation

| Method | Purpose | Parameters |
|--------|---------|------------|
| `create_a_descriptor(M, K, stride, row_major)` | A matrix (M×K) | Dimensions, stride, layout |
| `create_b_descriptor(K, N, stride, row_major)` | B matrix (K×N) | Dimensions, stride, layout |
| `create_c_descriptor(M, N, stride, row_major)` | C matrix (M×N) | Dimensions, stride, layout |

### View Creation

| Method | Purpose | Parameters |
|--------|---------|------------|
| `create_view(ptr, desc)` | Generic view | Pointer, descriptor |
| `create_a_view(ptr, desc)` | A matrix view | Pointer, descriptor |
| `create_b_view(ptr, desc)` | B matrix view | Pointer, descriptor |
| `create_c_view(ptr, desc)` | C matrix view | Pointer, descriptor |

### Type Aliases

```cpp
// CK_Tile shortcuts
template <typename T, int V = 1>
using CKDescWrapper = TensorDescriptorWrapper<CKTileBackend, T, V>;

template <typename T, int V = 1>
using CKViewWrapper = TensorViewWrapper<CKTileBackend, T, V>;

// MINT shortcuts
template <typename T>
using MintDescWrapper = TensorDescriptorWrapper<MintBackend, T>;

template <typename T>
using MintViewWrapper = TensorViewWrapper<MintBackend, T>;
```

## Key Benefits

✅ **User provides just numbers** - No backend-specific knowledge needed
✅ **Same code, different backends** - Change one type alias to switch
✅ **Zero overhead** - All dispatch at compile-time
✅ **No macros** - Clean template-based design
✅ **Type-safe** - Compiler catches errors

## How It Works

```
User Input: (M=128, K=64, stride=64, row_major=true)
                        │
                        ▼
        TensorDescriptorWrapper<BackendTag, ...>
                        │
         ┌──────────────┴──────────────┐
         │                             │
    CKTileBackend                 MintBackend
         │                             │
         ▼                             ▼
make_naive_tensor_desc       make_aliased_naive_packed
    (dims, strides, ...)         (aliases<"M","K">, {M,K})
```

**No runtime overhead!** Backend selection happens at compile-time via `std::conditional_t`.

## What's Abstracted?

| User Sees | CK_Tile Gets | MINT Gets |
|-----------|--------------|-----------|
| `M, K, stride, row_major` | `make_tuple(M,K)` + `make_tuple(stride,1)` | `aliases<"M","K">{}` + `{M,K}` |
| `ptr, desc` | `make_tensor_view<global>(ptr, desc)` | `make_tensor_view(desc, make_global_memory_view(ptr, size))` |

## Files

- **[simple_tensor_descriptor_wrapper.hpp](include/simple_tensor_descriptor_wrapper.hpp)** - Main implementation
- **[README_COMPLETE.md](README_COMPLETE.md)** - Full documentation
- **[descriptor_and_view_example.cpp](examples/descriptor_and_view_example.cpp)** - Usage examples

## Next Steps

1. ✅ Descriptor creation - **Done**
2. ✅ View creation - **Done**
3. ⏳ Tile operations (load/store) - **Coming next**
4. ⏳ Complete GEMM kernel - **Future**
