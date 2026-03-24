# Unified Tensor Wrapper Design

## Overview

A unified C++ wrapper that provides a backend-agnostic interface for creating tensor descriptors and views, supporting both **MINT** and **CK_tile** backends.

## Architecture

```
┌─────────────────────────────────────┐
│     User Code                       │
│  (Backend-agnostic API)             │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Unified Wrapper Interface          │
│  - TensorSpec (dimensions, strides) │
│  - TransformSpec (operations)       │
│  - BackendSelector                  │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
┌───────▼────────┐  ┌───────▼────────┐
│ MINT Backend   │  │ CK_tile Backend│
│ Adapter        │  │ Adapter        │
├────────────────┤  ├────────────────┤
│ • morphers     │  │ • coordinate_  │
│ • polymorpher  │  │   transform    │
│ • tensor_desc  │  │ • tensor_desc  │
│ • tensor_view  │  │ • tensor_view  │
└───────┬────────┘  └───────┬────────┘
        │                    │
┌───────▼────────┐  ┌───────▼────────┐
│   MINT API     │  │  CK_tile API   │
└────────────────┘  └────────────────┘
```

## Core Components

### 1. Unified Tensor Specification

```cpp
struct TensorSpec {
    std::vector<int> dimensions;    // [M, N, K, ...]
    std::vector<int> strides;       // Memory layout strides
    std::string data_type;          // "float", "half", etc.
};
```

### 2. Transform Specification

```cpp
enum class TransformType {
    PASS_THROUGH,
    MERGE,
    SPLIT,
    PAD,
    SHIFT,
    SLICE,
    // ... more
};

struct TransformOp {
    TransformType type;
    std::vector<int> params;        // Type-specific parameters
    std::vector<int> input_dims;    // Which dimensions to transform
    std::vector<int> output_dims;   // Result dimensions
};
```

### 3. Backend Selection

```cpp
enum class Backend {
    MINT,
    CK_TILE
};

class BackendSelector {
public:
    static void set_backend(Backend backend);
    static Backend get_backend();
};
```

### 4. Unified Wrapper Interface

```cpp
template <Backend B>
class TensorWrapper {
public:
    // Create descriptor from spec
    auto create_descriptor(const TensorSpec& spec,
                          const std::vector<TransformOp>& transforms);

    // Create tensor view
    template <typename T>
    auto create_view(void* data, const auto& descriptor);

    // Query information
    auto get_dimensions() const;
    auto get_strides() const;
};
```

## Backend Adapters

### MINT Backend Adapter

**Maps unified spec to MINT constructs:**

1. **TransformOp → Morpher**
   - `MERGE` → `poly::merge<...>`
   - `SPLIT` → `poly::split<...>`
   - `SHIFT` → `poly::shift<...>`
   - `PASS_THROUGH` → `poly::pass_through`

2. **Multiple TransformOps → Polymorpher**
   - Chain morphers using `poly::make_polymorpher`
   - Define dimension pairs
   - Create alias maps

3. **TensorSpec + Polymorpher → tensor_descriptor**

4. **tensor_descriptor + pointer → tensor_view**

### CK_tile Backend Adapter

**Maps unified spec to CK_tile constructs:**

1. **TransformOp → coordinate_transform**
   - `MERGE` → `make_merge_transform(...)`
   - `SPLIT` → `make_unmerge_transform(...)`
   - `PAD` → `make_pad_transform(...)`
   - `PASS_THROUGH` → `make_pass_through_transform(...)`

2. **TensorSpec → naive_tensor_descriptor**
   - `make_naive_tensor_descriptor(lengths, strides)`

3. **Apply transformations**
   - `transform_tensor_descriptor(desc, transforms, ...)`

4. **Create tensor_view**
   - `make_tensor_view<address_space>(ptr, desc)`

## Example Usage

### User Code (Backend-Agnostic)

```cpp
#include "unified_tensor_wrapper.hpp"

// Set backend at runtime or compile-time
BackendSelector::set_backend(Backend::MINT);

// Define tensor spec
TensorSpec matrixA {
    .dimensions = {128, 64},  // M=128, K=64
    .strides = {64, 1},       // Row-major
    .data_type = "float"
};

// Define transformations
std::vector<TransformOp> transforms = {
    // Split M into tiles
    TransformOp{
        .type = TransformType::SPLIT,
        .params = {32, 4},      // M0=32, M1=4 (M=128=32*4)
        .input_dims = {0},       // Dimension 0 (M)
        .output_dims = {0, 1}    // M0, M1
    },
    // Merge threads
    TransformOp{
        .type = TransformType::MERGE,
        .params = {32, 8},       // tid_m=32, tid_k=8
        .input_dims = {2, 3},    // New dimensions
        .output_dims = {2}       // P (thread ID)
    }
};

// Create wrapper (backend selected at compile-time via template)
TensorWrapper<Backend::MINT> wrapper;

// Create descriptor
auto descriptor = wrapper.create_descriptor(matrixA, transforms);

// Create view
float* data = /* ... */;
auto view = wrapper.create_view<float>(data, descriptor);

// Use view (same API regardless of backend!)
auto value = view.get(coord);
```

### Same Code with CK_tile Backend

```cpp
// Just change the backend!
BackendSelector::set_backend(Backend::CK_TILE);
TensorWrapper<Backend::CK_TILE> wrapper;

// Rest of the code remains identical
auto descriptor = wrapper.create_descriptor(matrixA, transforms);
auto view = wrapper.create_view<float>(data, descriptor);
```

## Implementation Strategy

### Phase 1: Core Infrastructure
- [ ] Define unified specs (TensorSpec, TransformOp)
- [ ] Implement BackendSelector
- [ ] Create base TensorWrapper template

### Phase 2: MINT Backend Adapter
- [ ] TransformOp → Morpher mapping
- [ ] Polymorpher construction
- [ ] tensor_descriptor creation
- [ ] tensor_view creation

### Phase 3: CK_tile Backend Adapter
- [ ] TransformOp → coordinate_transform mapping
- [ ] naive_tensor_descriptor creation
- [ ] transform_tensor_descriptor application
- [ ] tensor_view creation

### Phase 4: Testing & Examples
- [ ] Unit tests for each backend
- [ ] GEMM example with both backends
- [ ] Performance comparison
- [ ] Documentation

## Key Design Decisions

### 1. Compile-time vs Runtime Backend Selection

**Chosen: Hybrid Approach**
- Template parameter for compile-time selection
- BackendSelector for runtime queries
- Allows optimization while maintaining flexibility

### 2. Transform Representation

**Chosen: High-level TransformOp struct**
- Abstracts both MINT and CK_tile primitives
- Extensible for new transform types
- Maps cleanly to both backends

### 3. Memory Management

**Chosen: Non-owning views**
- User manages memory allocation
- Wrapper creates views over existing data
- Consistent with both MINT and CK_tile philosophy

### 4. Type Safety

**Chosen: Strong typing with templates**
- Data type specified at view creation
- Compile-time dimension checking where possible
- Template specialization for backend differences

## File Structure

```
unified_tensor_wrapper/
├── DESIGN.md                      # This file
├── include/
│   ├── unified_tensor_wrapper.hpp # Main public interface
│   ├── tensor_spec.hpp            # TensorSpec, TransformOp
│   ├── backend_selector.hpp       # Backend selection
│   ├── adapters/
│   │   ├── mint_adapter.hpp       # MINT backend adapter
│   │   └── ck_tile_adapter.hpp    # CK_tile backend adapter
│   └── detail/
│       ├── mint_detail.hpp        # MINT implementation details
│       └── ck_tile_detail.hpp     # CK_tile implementation details
├── src/
│   ├── backend_selector.cpp       # Backend selection implementation
│   ├── mint_adapter.cpp           # MINT adapter implementation
│   └── ck_tile_adapter.cpp        # CK_tile adapter implementation
├── examples/
│   ├── gemm_mint.cpp              # GEMM using MINT backend
│   ├── gemm_ck_tile.cpp           # GEMM using CK_tile backend
│   └── gemm_unified.cpp           # GEMM with runtime backend selection
└── tests/
    ├── test_mint_adapter.cpp
    ├── test_ck_tile_adapter.cpp
    └── test_unified_interface.cpp
```

## API Mapping Tables

### Transform Type Mapping

| Unified | MINT | CK_tile |
|---------|------|---------|
| PASS_THROUGH | `poly::pass_through` | `make_pass_through_transform` |
| MERGE | `poly::merge<dims>` | `make_merge_transform` |
| SPLIT | `poly::split<dims>` | `make_unmerge_transform` |
| SHIFT | `poly::shift<value>` | `make_pad_transform` |
| PAD | Custom morpher | `make_pad_transform` |
| SLICE | Custom morpher | `make_slice_transform` |
| FREEZE | N/A | `make_freeze_transform` |
| REPLICATE | N/A | `make_replicate_transform` |

### Descriptor Creation

| Backend | Method |
|---------|--------|
| MINT | `tensor_descriptor<Poly, TopAliases, BotAliases>` |
| CK_tile | `make_naive_tensor_descriptor` + `transform_tensor_descriptor` |

### View Creation

| Backend | Method |
|---------|--------|
| MINT | `make_tensor_view(ptr, descriptor)` |
| CK_tile | `make_tensor_view<address_space>(ptr, descriptor)` |

## Future Enhancements

1. **Auto-optimization**: Choose best backend based on tensor size/operation
2. **Mixed backend**: Use different backends for different tensors in same kernel
3. **Python bindings**: PyTorch/JAX integration
4. **Profiling hooks**: Measure backend performance
5. **Extended transforms**: Support all CK_tile transforms (XOR, modulo, etc.)
6. **Distributed descriptors**: Support MINT's distributed_tensor_descriptor

## Benefits

✅ **Portability**: Write once, run with either backend
✅ **Flexibility**: Switch backends without code changes
✅ **Learning**: Understand both frameworks through unified API
✅ **Testing**: Compare backends easily
✅ **Migration**: Gradual transition from CK_tile to MINT or vice versa
