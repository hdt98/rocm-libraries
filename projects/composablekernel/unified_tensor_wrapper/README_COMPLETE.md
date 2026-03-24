# Complete Unified Tensor Wrapper

A unified C++ wrapper providing backend-agnostic APIs for tensor descriptor and view creation with CK_Tile and MINT backends.

## Overview

This wrapper eliminates backend-specific API differences by providing a single unified interface:

```cpp
// User just provides numbers - no backend knowledge needed!
auto desc = DescWrapper::create_a_descriptor(M, K, stride, row_major);
auto view = ViewWrapper::create_view(ptr, desc);

// Works with both CK_Tile and MINT!
```

## Problem Statement

### Different APIs for Same Operations

**Creating Tensor Descriptors:**

| Backend | API |
|---------|-----|
| **CK_Tile** | `make_naive_tensor_descriptor(dims, strides, vector_hint, align_hint)` |
| **MINT** | `make_aliased_naive_packed_tensor_descriptor(aliases<"M","K">{}, alias<"Offset">{}, {M,K})` |

**Creating Tensor Views:**

| Backend | API |
|---------|-----|
| **CK_Tile** | `make_tensor_view<address_space_enum::global>(pointer, descriptor)` |
| **MINT** | `make_tensor_view(descriptor, make_global_memory_view(pointer, size))` |

**Challenge:**
- MINT requires compile-time string aliases (`aliases<"M", "K">{}`) which cannot be generated from runtime data
- CK_Tile uses template parameters for address space
- Different parameter orders and requirements

## Solution: Template Tag Dispatch

Use **backend tag types** to select the correct implementation at compile-time:

```cpp
struct CKTileBackend {};  // Tag for CK_Tile
struct MintBackend {};    // Tag for MINT

template <typename BackendTag, typename DataType, int VectorSize = 1>
class TensorDescriptorWrapper {
    // Automatically picks right backend via std::conditional_t
};
```

## Complete API

### 1. Descriptor Creation

```cpp
// Create wrapper for your chosen backend
using DescWrapper = TensorDescriptorWrapper<CKTileBackend, float, 4>;
// or
using DescWrapper = TensorDescriptorWrapper<MintBackend, float>;

// Create descriptors with just numbers!
auto a_desc = DescWrapper::create_a_descriptor(
    M, K,
    K,      // stride
    true    // row-major
);

auto b_desc = DescWrapper::create_b_descriptor(K, N, N, true);
auto c_desc = DescWrapper::create_c_descriptor(M, N, N, true);
```

### 2. Tensor View Creation

```cpp
// Create wrapper for your chosen backend
using ViewWrapper = TensorViewWrapper<CKTileBackend, float, 4>;
// or
using ViewWrapper = TensorViewWrapper<MintBackend, float>;

// Create views from pointer + descriptor
auto a_view = ViewWrapper::create_view(a_ptr, a_desc);
auto b_view = ViewWrapper::create_view(b_ptr, b_desc);
auto c_view = ViewWrapper::create_view(c_ptr, c_desc);

// Or use convenience methods
auto a_view = ViewWrapper::create_a_view(a_ptr, a_desc);
auto b_view = ViewWrapper::create_b_view(b_ptr, b_desc);
auto c_view = ViewWrapper::create_c_view(c_ptr, c_desc);
```

### 3. Generic Code (Backend-Agnostic)

```cpp
template <typename BackendTag>
__global__ void my_kernel(const float* a, const float* b, float* c,
                          index_t M, index_t N, index_t K)
{
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

    // Use views for tile operations...
    // Same code works for both backends!
}

// Launch with CK_Tile backend
my_kernel<CKTileBackend><<<...>>>(a, b, c, M, N, K);

// Launch with MINT backend
my_kernel<MintBackend><<<...>>>(a, b, c, M, N, K);
```

## How It Works

### Architecture

```
┌─────────────────────────────────┐
│  User Code (Backend-Agnostic)   │
│  create_descriptor(M, K, ...)   │
│  create_view(ptr, desc)         │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│  TensorDescriptorWrapper        │
│  TensorViewWrapper              │
│  (Tag dispatch via template)    │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼──────┐ ┌───▼──────────┐
│ CK_Tile    │ │ MINT         │
│ Adapter    │ │ Adapter      │
└─────┬──────┘ └───┬──────────┘
      │             │
┌─────▼──────┐ ┌───▼──────────┐
│ CK_Tile    │ │ MINT         │
│ Native API │ │ Native API   │
└────────────┘ └──────────────┘
```

### Compile-Time Dispatch

```cpp
template <typename BackendTag, typename DataType, int VectorSize = 1>
class TensorDescriptorWrapper {
private:
    // Compile-time if-else
    using Adapter = std::conditional_t<
        std::is_same_v<BackendTag, CKTileBackend>,
        CKTileDescriptorAdapter<DataType, VectorSize>,  // CK_Tile
        MintDescriptorAdapter<DataType>                  // MINT
    >;

public:
    static auto create_a_descriptor(...) {
        // Delegates to selected adapter
        return Adapter::create_a_descriptor(...);
    }
};
```

**Zero runtime overhead!** The compiler resolves everything at compile-time.

### Backend Adapters

#### CK_Tile Adapter

**Descriptor:**
```cpp
return make_naive_tensor_descriptor(
    make_tuple(M, K),         // dimensions
    make_tuple(stride, 1),    // strides
    number<VectorSize>{},     // vector hint
    number<1>{}               // alignment
);
```

**View:**
```cpp
return make_tensor_view<address_space_enum::global>(ptr, desc);
```

#### MINT Adapter

**Descriptor:**
```cpp
return make_aliased_naive_packed_tensor_descriptor(
    aliases<"M", "K">{},      // Hard-coded dimension names
    alias<"Offset">{},        // Offset name
    {M, K}                    // Dimension values (runtime)
);
```

**View:**
```cpp
const auto bottom_size = desc.bottom_lengths()[0];
auto mem_view = make_global_memory_view(ptr, bottom_size);
return make_tensor_view(desc, mem_view);
```

## Usage Patterns

### Pattern 1: Direct Backend Selection

```cpp
// Choose backend explicitly
using CKDescWrapper = CKTileDescriptorWrapper<float, 4>;
using CKViewWrapper = CKTileViewWrapper<float, 4>;

auto desc = CKDescWrapper::create_a_descriptor(M, K, K, true);
auto view = CKViewWrapper::create_view(a_ptr, desc);
```

### Pattern 2: Template Parameter

```cpp
template <typename BackendTag>
void my_function() {
    using DescWrapper = TensorDescriptorWrapper<BackendTag, float, 4>;
    using ViewWrapper = TensorViewWrapper<BackendTag, float, 4>;

    auto desc = DescWrapper::create_a_descriptor(M, K, K, true);
    auto view = ViewWrapper::create_view(a_ptr, desc);
}

// Call with different backends
my_function<CKTileBackend>();
my_function<MintBackend>();
```

### Pattern 3: Runtime Backend Selection

```cpp
enum class RuntimeBackend { CK_TILE, MINT };

void launch_kernel(RuntimeBackend backend, ...) {
    if (backend == RuntimeBackend::CK_TILE) {
        my_kernel<CKTileBackend><<<...>>>(...);
    } else {
        my_kernel<MintBackend><<<...>>>(...);
    }
}
```

## Type Aliases Reference

### Descriptor Wrappers

```cpp
// CK_Tile
template <typename DataType, int VectorSize = 1>
using CKTileDescriptorWrapper =
    TensorDescriptorWrapper<CKTileBackend, DataType, VectorSize>;

// MINT
template <typename DataType>
using MintDescriptorWrapper =
    TensorDescriptorWrapper<MintBackend, DataType>;
```

### View Wrappers

```cpp
// CK_Tile
template <typename DataType, int VectorSize = 1>
using CKTileViewWrapper =
    TensorViewWrapper<CKTileBackend, DataType, VectorSize>;

// MINT
template <typename DataType>
using MintViewWrapper =
    TensorViewWrapper<MintBackend, DataType>;
```

## Key Design Decisions

### 1. Why Tag Dispatch?

| Alternative | Issues |
|-------------|--------|
| **Macros** | Can't mix backends in one file, global state, not type-safe |
| **Virtual Functions** | Runtime overhead, device code restrictions |
| **Tag Dispatch** | ✅ Zero overhead, type-safe, compile-time, flexible |

### 2. Why Hard-code MINT Aliases?

MINT requires **compile-time string literals** (`aliases<"M", "K">{}`), which fundamentally cannot be generated from runtime strings. The wrapper pre-defines standard aliases:

- **A matrix:** `aliases<"M", "K">{}` (row-major) or `aliases<"K", "M">{}` (col-major)
- **B matrix:** `aliases<"N", "K">{}`
- **C matrix:** `aliases<"M", "N">{}`

For custom aliases, users would need to create specialized adapters.

### 3. Why VectorSize Template Parameter?

CK_Tile uses `number<VectorSize>{}` as a compile-time constant for optimization hints. Template parameter allows:

```cpp
TensorDescriptorWrapper<CKTileBackend, float, 4>  // Vector size 4
TensorDescriptorWrapper<CKTileBackend, float, 8>  // Vector size 8
```

MINT doesn't use vector hints, so it ignores this parameter.

### 4. Why Separate Descriptor and View Wrappers?

**Design choice:** Follow natural workflow:
1. **Create descriptor** (logical dimensions/layout)
2. **Create view** (bind descriptor to memory)

Mirrors both CK_Tile and MINT idioms.

## Full Example: GEMM Setup

```cpp
#include "simple_tensor_descriptor_wrapper.hpp"

template <typename BackendTag>
__global__ void gemm_kernel(
    const float* a_ptr,
    const float* b_ptr,
    float* c_ptr,
    index_t M, index_t N, index_t K)
{
    // Type aliases
    using DescWrapper = TensorDescriptorWrapper<BackendTag, float, 4>;
    using ViewWrapper = TensorViewWrapper<BackendTag, float, 4>;

    // 1. Create descriptors (just numbers!)
    auto a_desc = DescWrapper::create_a_descriptor(M, K, K, true);
    auto b_desc = DescWrapper::create_b_descriptor(K, N, N, true);
    auto c_desc = DescWrapper::create_c_descriptor(M, N, N, true);

    // 2. Create views
    auto a_view = ViewWrapper::create_a_view(a_ptr, a_desc);
    auto b_view = ViewWrapper::create_b_view(b_ptr, b_desc);
    auto c_view = ViewWrapper::create_c_view(c_ptr, c_desc);

    // 3. Use views for tile operations
    // (load_tile, store_tile, etc.)
    // Same code for both backends!
}

// Launch
int main() {
    // Allocate memory, initialize data...

    // Option 1: CK_Tile backend
    gemm_kernel<CKTileBackend><<<grid, block>>>(a, b, c, M, N, K);

    // Option 2: MINT backend
    gemm_kernel<MintBackend><<<grid, block>>>(a, b, c, M, N, K);

    return 0;
}
```

## Benefits

✅ **Unified API** - One interface for both backends
✅ **Zero overhead** - All dispatch at compile-time
✅ **Type-safe** - Template-based, no macros
✅ **Backend-agnostic code** - Write once, run with either backend
✅ **Easy migration** - Switch backends by changing type alias
✅ **No macros** - Clean C++ templates

## Limitations

1. **MINT aliases fixed** - Can't generate custom dimension names at runtime
2. **Packed layout** - MINT adapter assumes packed (contiguous) tensors
3. **Matrix operations only** - Currently focused on A, B, C matrices for GEMM

## Future Extensions

1. **Strided MINT support** - Use `make_aliased_naive_tensor_descriptor` instead of packed version
2. **Custom aliases** - Template parameter for MINT dimension names
3. **More tensor types** - Beyond A/B/C matrices
4. **Tile operations** - Wrappers for load/store/compute operations

## Files

```
unified_tensor_wrapper/
├── include/
│   └── simple_tensor_descriptor_wrapper.hpp  # Main implementation
├── examples/
│   ├── simple_descriptor_example.cpp         # Descriptor examples
│   └── descriptor_and_view_example.cpp       # Complete workflow
└── README_COMPLETE.md                        # This file
```

## Building

```bash
cd unified_tensor_wrapper/examples

# Build with CK_Tile
hipcc -std=c++17 \
      -I../include \
      -I../../include \
      descriptor_and_view_example.cpp \
      -o view_example

# Build with MINT (ensure MINT is available)
hipcc -std=c++17 \
      -I../include \
      -I../../include \
      -I/path/to/mint/include \
      descriptor_and_view_example.cpp \
      -o view_example
```

## Summary

This wrapper provides:

1. **Descriptor creation** - Backend-agnostic numerical specification
2. **View creation** - Unified pointer + descriptor → view API
3. **Tag dispatch** - Compile-time backend selection (zero overhead)
4. **Generic code** - Write once, works with both CK_Tile and MINT

**Next step:** Extend to tile operations (load/store) and complete GEMM kernels!
