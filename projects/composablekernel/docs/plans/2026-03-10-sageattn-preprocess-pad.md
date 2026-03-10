# SageAttn Preprocess Internal Padding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the sageattn_preprocess kernel internally pad seqlen_q / seqlen_k to multiples of kRows, expose a `get_buffer_sizes` helper, and add irregular-seqlen test cases.

**Architecture:** Each of the four files is modified independently. The pipeline gains `n_rows_valid` params; the kernels compute padded tile counts; `sageattn_preprocess_run` overrides all output strides with padded-based values; a new `get_buffer_sizes<InputT,kRows>` free function tells callers how much memory to allocate.

**Tech Stack:** HIP C++20, CK Tile, GTest; target arch gfx950.

---

## Background / conventions

```
seqlen_q_padded = ceil_div(seqlen_q, kRows) * kRows
seqlen_k_padded = ceil_div(seqlen_k, kRows) * kRows
num_q_tiles     = seqlen_q_padded / kRows          (≡ ceil_div(seqlen_q, kRows))
num_k_tiles     = seqlen_k_padded / kRows
```

kRows=128 is a multiple of `kVGroup*kVGroupsPerBlock=64`, so V kernel alignment is automatic.

Key files:
- **Pipeline:** `include/ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp`
- **Kernel:**   `include/ck_tile/ops/sageattn_preprocess/kernel/sageattn_preprocess_kernel.hpp`
- **Run/API:**  `include/ck_tile/ops/sageattn_preprocess/sageattn_preprocess.hpp`
- **Test:**     `test/ck_tile/fmha/test_fmha_sageattn_preprocess.cpp`

Build command (from `/root/rocm-libraries/projects/composablekernel/build`):
```bash
cmake --build . --target test_ck_tile_fmha_sageattn_preprocess -j$(nproc)
```
Run:
```bash
./bin/test_ck_tile_fmha_sageattn_preprocess
```

---

### Task 1: Add `n_rows_valid` to Pipeline functions

**Files:**
- Modify: `include/ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp`

The three device functions (`RunQMean`, `RunQQuantize`, `RunKSmoothAndQuantize`) each need to know how many rows of the kRows-row tile are real data. Rows beyond `n_rows_valid` must produce zeros in all outputs.

**Step 1: Replace `RunQMean` signature and body**

Old:
```cpp
CK_TILE_DEVICE void RunQMean(const InputT* __restrict__ src_ptr,
                             InputT* __restrict__ q_mean_ptr,
                             void* smem) const
```

New (divide by n_rows_valid, not kRows):
```cpp
CK_TILE_DEVICE void RunQMean(const InputT* __restrict__ src_ptr,
                             InputT* __restrict__ q_mean_ptr,
                             void* smem,
                             index_t n_rows_valid) const
{
    float*        smem_f = reinterpret_cast<float*>(smem);
    const index_t tid    = get_thread_id();

    for(index_t d = tid; d < kCols; d += kBlockSize)
    {
        float sum = 0.0f;
        for(index_t r = 0; r < n_rows_valid; r++)
            sum += static_cast<float>(src_ptr[r * kCols + d]);
        const float mean = sum / static_cast<float>(n_rows_valid);
        smem_f[d]     = mean;
        q_mean_ptr[d] = static_cast<InputT>(mean);
    }
    // Caller issues block_sync_lds() before RunQQuantize.
}
```

**Step 2: Replace `RunQQuantize` signature and body**

Old:
```cpp
CK_TILE_DEVICE void RunQQuantize(const InputT* __restrict__ src_ptr,
                                 uint8_t* __restrict__ dst_hat_ptr,
                                 uint8_t* __restrict__ dst_scale_ptr,
                                 const void* smem) const
```

New (zero-fill rows >= n_rows_valid):
```cpp
CK_TILE_DEVICE void RunQQuantize(const InputT* __restrict__ src_ptr,
                                 uint8_t* __restrict__ dst_hat_ptr,
                                 uint8_t* __restrict__ dst_scale_ptr,
                                 const void* smem,
                                 index_t n_rows_valid) const
{
    const float*  smem_mean_f = reinterpret_cast<const float*>(smem);
    const index_t tid         = get_thread_id();

    constexpr index_t kNumGroups = kCols / kScaleGranularity;
    const index_t     row_idx    = tid / kNumGroups;
    const index_t     grp_idx    = tid % kNumGroups;
    const index_t     d_start    = grp_idx * kScaleGranularity;

    if(row_idx >= n_rows_valid)
    {
        // pad row — zero hat and scale
        dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
        uint8_t* hat_dst = dst_hat_ptr + row_idx * (kCols / 2) +
                           grp_idx * (kScaleGranularity / 2);
        for(index_t j = 0; j < kScaleGranularity / 2; j++)
            hat_dst[j] = 0;
        return;
    }

    constexpr float rcp_dst_max = 1.0f / 6.0f;

    const InputT* src_row = src_ptr + row_idx * kCols;

    float group_data[kScaleGranularity];
    float max_abs = 0.0f;
    for(index_t j = 0; j < kScaleGranularity; j++)
    {
        const float val = static_cast<float>(src_row[d_start + j]) -
                          smem_mean_f[d_start + j];
        group_data[j] = val;
        max_abs       = max(max_abs, abs(val));
    }

    const float scale = bit_cast<float>(
        (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
        numeric_traits<float>::head_mask);

    dst_scale_ptr[row_idx * kNumGroups + grp_idx] =
        static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

    PackFP4Group(group_data,
                 dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2),
                 scale);
}
```

**Step 3: Replace `RunKSmoothAndQuantize` signature and body**

Old:
```cpp
CK_TILE_DEVICE void RunKSmoothAndQuantize(const InputT* __restrict__ src_ptr,
                                          const InputT* __restrict__ k_mean_ptr,
                                          InputT* __restrict__ k_prime_ptr,
                                          index_t k_prime_stride,
                                          uint8_t* __restrict__ dst_hat_ptr,
                                          uint8_t* __restrict__ dst_scale_ptr) const
```

New:
```cpp
CK_TILE_DEVICE void RunKSmoothAndQuantize(const InputT* __restrict__ src_ptr,
                                          const InputT* __restrict__ k_mean_ptr,
                                          InputT* __restrict__ k_prime_ptr,
                                          index_t k_prime_stride,
                                          uint8_t* __restrict__ dst_hat_ptr,
                                          uint8_t* __restrict__ dst_scale_ptr,
                                          index_t n_rows_valid) const
{
    const index_t tid = get_thread_id();

    constexpr index_t kNumGroups = kCols / kScaleGranularity;
    const index_t     row_idx    = tid / kNumGroups;
    const index_t     grp_idx    = tid % kNumGroups;
    const index_t     d_start    = grp_idx * kScaleGranularity;

    if(row_idx >= n_rows_valid)
    {
        // pad row — zero k_prime, hat, scale
        InputT* dst_row = k_prime_ptr + row_idx * k_prime_stride;
        for(index_t j = 0; j < kScaleGranularity; j++)
            dst_row[d_start + j] = static_cast<InputT>(0);
        dst_scale_ptr[row_idx * kNumGroups + grp_idx] = 0;
        uint8_t* hat_dst = dst_hat_ptr + row_idx * (kCols / 2) +
                           grp_idx * (kScaleGranularity / 2);
        for(index_t j = 0; j < kScaleGranularity / 2; j++)
            hat_dst[j] = 0;
        return;
    }

    constexpr float rcp_dst_max = 1.0f / 6.0f;

    const InputT* src_row = src_ptr + row_idx * kCols;
    InputT*       dst_row = k_prime_ptr + row_idx * k_prime_stride;

    float group_data[kScaleGranularity];
    float max_abs = 0.0f;
    for(index_t j = 0; j < kScaleGranularity; j++)
    {
        const float val = static_cast<float>(src_row[d_start + j]) -
                          static_cast<float>(k_mean_ptr[d_start + j]);
        group_data[j]        = val;
        dst_row[d_start + j] = static_cast<InputT>(val);
        max_abs              = max(max_abs, abs(val));
    }

    const float scale = bit_cast<float>(
        (bit_cast<uint32_t>(max_abs * rcp_dst_max) + numeric_traits<float>::mant_mask) &
        numeric_traits<float>::head_mask);

    dst_scale_ptr[row_idx * kNumGroups + grp_idx] =
        static_cast<uint8_t>(bit_cast<uint32_t>(scale) >> 23);

    PackFP4Group(group_data,
                 dst_hat_ptr + row_idx * (kCols / 2) + grp_idx * (kScaleGranularity / 2),
                 scale);
}
```

No compile step yet — all in one file, verified as a unit in Task 3.

---

### Task 2: Update `SageAttnPreprocessKernel` and `SageAttnVPreprocessKernel`

**Files:**
- Modify: `include/ck_tile/ops/sageattn_preprocess/kernel/sageattn_preprocess_kernel.hpp`

#### 2a — `SageAttnPreprocessKernel::operator()`

The `num_q_tiles` / `num_k_tiles` in kargs will now be padded counts (set by `sageattn_preprocess_run`). The kernel must compute `n_rows_valid` per tile and pass it to the pipeline. Remove the old `if(row_start < seqlen_q/k)` guards.

Replace the entire `operator()` body:

```cpp
CK_TILE_DEVICE void operator()(Kargs kargs) const
{
    const index_t tile_x    = get_block_id();
    const index_t head_idx  = blockIdx.y;
    const index_t batch_idx = blockIdx.z;

    const bool do_q = tile_x < kargs.num_q_tiles;
    const bool do_k = tile_x < kargs.num_k_tiles;

    __shared__ char smem[GetSmemSize()];

    Pipeline pipeline{};

    // ---- Step 1: Q mean ------------------------------------------------
    if(do_q)
    {
        const index_t row_start  = tile_x * kargs.q_tile_size;
        const index_t n_rows_q   = min(static_cast<index_t>(kRows),
                                       kargs.seqlen_q - row_start);

        const InputT* src_q =
            kargs.q_ptr + batch_idx * kargs.batch_stride_q +
            head_idx * kargs.nhead_stride_q + row_start * kargs.stride_q;

        InputT* q_mean =
            kargs.q_mean_ptr + batch_idx * kargs.batch_stride_q_mean +
            head_idx * kargs.nhead_stride_q_mean + tile_x * kargs.stride_q_mean;

        pipeline.RunQMean(src_q, q_mean, smem, n_rows_q);
    }
    block_sync_lds(); // smem q_mean visible for step 2

    // ---- Step 2: Q quantize --------------------------------------------
    if(do_q)
    {
        const index_t row_start  = tile_x * kargs.q_tile_size;
        const index_t n_rows_q   = min(static_cast<index_t>(kRows),
                                       kargs.seqlen_q - row_start);

        const InputT* src_q =
            kargs.q_ptr + batch_idx * kargs.batch_stride_q +
            head_idx * kargs.nhead_stride_q + row_start * kargs.stride_q;

        uint8_t* dst_q_hat =
            kargs.q_hat_ptr + batch_idx * kargs.batch_stride_q_hat +
            head_idx * kargs.nhead_stride_q_hat + row_start * kargs.stride_q_hat;

        uint8_t* dst_q_scale =
            kargs.q_scale_ptr + batch_idx * kargs.batch_stride_q_scale +
            head_idx * kargs.nhead_stride_q_scale + row_start * kargs.stride_q_scale;

        pipeline.RunQQuantize(src_q, dst_q_hat, dst_q_scale, smem, n_rows_q);
    }

    // ---- Step 3: K smooth + quantize -----------------------------------
    if(do_k)
    {
        const index_t row_start  = tile_x * kRows;
        const index_t n_rows_k   = min(static_cast<index_t>(kRows),
                                       kargs.seqlen_k - row_start);

        const InputT* src_k =
            kargs.k_ptr + batch_idx * kargs.batch_stride_k +
            head_idx * kargs.nhead_stride_k + row_start * kargs.stride_k;

        const InputT* k_mean =
            kargs.k_mean_ptr + batch_idx * kargs.batch_stride_k_mean +
            head_idx * kargs.nhead_stride_k_mean;

        InputT* k_prime =
            kargs.k_prime_ptr + batch_idx * kargs.batch_stride_k_prime +
            head_idx * kargs.nhead_stride_k_prime + row_start * kargs.stride_k_prime;

        uint8_t* dst_k_hat =
            kargs.k_hat_ptr + batch_idx * kargs.batch_stride_k_hat +
            head_idx * kargs.nhead_stride_k_hat + row_start * kargs.stride_k_hat;

        uint8_t* dst_k_scale =
            kargs.k_scale_ptr + batch_idx * kargs.batch_stride_k_scale +
            head_idx * kargs.nhead_stride_k_scale + row_start * kargs.stride_k_scale;

        pipeline.RunKSmoothAndQuantize(
            src_k, k_mean, k_prime, kargs.stride_k_prime,
            dst_k_hat, dst_k_scale, n_rows_k);
    }
}
```

Note: `seqlen_q - row_start` and `seqlen_k - row_start` are always >= 1 (guaranteed by
padded tile count), so no zero-guard needed.

#### 2b — `SageAttnVPreprocessKernel`: separate real vs padded seqlen_k

Add `seqlen_k_real` to `SageAttnVPreprocessHostArgs` and `SageAttnVPreprocessKargs`:

In `SageAttnVPreprocessHostArgs` add after `seqlen_k`:
```cpp
index_t     seqlen_k_real;    // original un-padded seqlen_k (for input bounds check)
```

In `SageAttnVPreprocessKargs` add after `seqlen_k`:
```cpp
index_t       seqlen_k_real;
```

In `MakeKargs`:
```cpp
k.seqlen_k_real = h.seqlen_k_real;
```

In `GridSize`, change to use padded `seqlen_k` (= h.seqlen_k):
```cpp
// Already uses h.seqlen_k; after our change h.seqlen_k = seqlen_k_padded, no change here.
```

In `operator()`, change the bounds check from `kargs.seqlen_k` to `kargs.seqlen_k_real`:
```cpp
if(row < kargs.seqlen_k_real)
```

#### 2c — `SageAttnKMeanKernel`: no changes needed

The existing code already clamps `n_rows = min(row_start + kRows, seqlen_k) - row_start` (giving 0 for fully-out-of-bounds tiles) and normalizes by real `seqlen_k`. When `sageattn_preprocess_run` passes padded `num_k_tiles`, extra tiles run with `n_rows = 0`, contribute nothing to the partial sum, and increment the counter normally. Already correct.

---

### Task 3: Add `get_buffer_sizes` and update `sageattn_preprocess_run`

**Files:**
- Modify: `include/ck_tile/ops/sageattn_preprocess/sageattn_preprocess.hpp`

#### 3a — Add `SageAttnPreprocessBufferSizes` struct and `get_buffer_sizes`

Add before `sageattn_preprocess_run`:

```cpp
// Buffer size descriptor returned by get_buffer_sizes.
// All *_bytes fields are byte counts for device allocation.
// seqlen_q_padded / seqlen_k_padded are exposed so callers can
// set up correct strides (or just call sageattn_preprocess_run and let it
// override output strides automatically).
struct SageAttnPreprocessBufferSizes
{
    // ---- outputs produced by sageattn_preprocess_run ----
    size_t  delta_s_bytes;        // float  [B, H, num_q_tiles, seqlen_k_padded]
    size_t  k_prime_bytes;        // InputT [B, H, seqlen_k_padded, hdim]
    size_t  q_hat_bytes;          // uint8  [B, H, seqlen_q_padded, hdim/2]
    size_t  q_scale_bytes;        // uint8  [B, H, seqlen_q_padded, hdim/32]
    size_t  q_mean_bytes;         // InputT [B, H, num_q_tiles, hdim]
    size_t  k_hat_bytes;          // uint8  [B, H, seqlen_k_padded, hdim/2]
    size_t  k_scale_bytes;        // uint8  [B, H, seqlen_k_padded, hdim/32]
    size_t  v_hat_bytes;          // uint8  [B, H, hdim, seqlen_k_padded/2]
    size_t  v_scale_bytes;        // uint8  [B, H, hdim, seqlen_k_padded/32]
    // ---- caller-allocated scratch (unchanged by padding) ----
    size_t  k_mean_bytes;         // InputT [B, H, hdim]
    size_t  k_mean_partial_bytes; // float  [B, H, hdim]
    size_t  counter_bytes;        // int32  [B, H]
    // ---- padded dims ----
    index_t seqlen_q_padded;
    index_t seqlen_k_padded;
    index_t num_q_tiles;
    index_t num_k_tiles;
};

template <typename InputT, index_t kRows>
SageAttnPreprocessBufferSizes get_buffer_sizes(
    index_t batch, index_t nhead, index_t seqlen_q, index_t seqlen_k, index_t hdim)
{
    constexpr index_t kG = 32; // MXFP4 scale granularity

    const index_t sq_pad = ((seqlen_q + kRows - 1) / kRows) * kRows;
    const index_t sk_pad = ((seqlen_k + kRows - 1) / kRows) * kRows;
    const index_t nqt    = sq_pad / kRows;
    const index_t nkt    = sk_pad / kRows;

    SageAttnPreprocessBufferSizes s{};
    s.seqlen_q_padded = sq_pad;
    s.seqlen_k_padded = sk_pad;
    s.num_q_tiles     = nqt;
    s.num_k_tiles     = nkt;

    s.delta_s_bytes        = static_cast<size_t>(batch * nhead * nqt * sk_pad) * sizeof(float);
    s.k_prime_bytes        = static_cast<size_t>(batch * nhead * sk_pad * hdim) * sizeof(InputT);
    s.q_hat_bytes          = static_cast<size_t>(batch * nhead * sq_pad * (hdim / 2));
    s.q_scale_bytes        = static_cast<size_t>(batch * nhead * sq_pad * (hdim / kG));
    s.q_mean_bytes         = static_cast<size_t>(batch * nhead * nqt * hdim) * sizeof(InputT);
    s.k_hat_bytes          = static_cast<size_t>(batch * nhead * sk_pad * (hdim / 2));
    s.k_scale_bytes        = static_cast<size_t>(batch * nhead * sk_pad * (hdim / kG));
    s.v_hat_bytes          = static_cast<size_t>(batch * nhead * hdim * (sk_pad / 2));
    s.v_scale_bytes        = static_cast<size_t>(batch * nhead * hdim * (sk_pad / kG));
    s.k_mean_bytes         = static_cast<size_t>(batch * nhead * hdim) * sizeof(InputT);
    s.k_mean_partial_bytes = static_cast<size_t>(batch * nhead * hdim) * sizeof(float);
    s.counter_bytes        = static_cast<size_t>(batch * nhead) * sizeof(int32_t);
    return s;
}
```

#### 3b — Update `sageattn_preprocess_run` to use padded dims

At the top of the function body, after the existing dimension extractions, add:

```cpp
const auto bsz = get_buffer_sizes<InputT, kRows>(batch, nhead, seqlen_q, seqlen_k, hdim);
const index_t seqlen_q_padded = bsz.seqlen_q_padded;
const index_t seqlen_k_padded = bsz.seqlen_k_padded;
const index_t num_q_tiles_pad = bsz.num_q_tiles;
const index_t num_k_tiles_pad = bsz.num_k_tiles;
```

Replace the existing:
```cpp
const index_t num_q_tiles = prep_args.num_q_tiles;
const index_t num_k_tiles = prep_args.num_k_tiles;
```
with (the padded variants computed above; hargs.num_q_tiles/num_k_tiles are now ignored):
```cpp
const index_t num_q_tiles = num_q_tiles_pad;
const index_t num_k_tiles = num_k_tiles_pad;
```

In **Launch 1** (SageAttnPreprocessKernel), override the output strides before calling MakeKargs. Change:
```cpp
SageAttnPreprocessHostArgs prep = prep_args;
prep.k_mean_ptr           = k_mean_buf;
prep.k_prime_ptr          = k_prime_buf;
prep.stride_k_prime       = hdim;
prep.nhead_stride_k_prime = seqlen_k * hdim;
prep.batch_stride_k_prime = nhead * seqlen_k * hdim;
```
To:
```cpp
SageAttnPreprocessHostArgs prep = prep_args;
// Use padded tile counts (real seqlen_q/k stay as-is for bounds checking).
prep.num_q_tiles = num_q_tiles_pad;
prep.num_k_tiles = num_k_tiles_pad;
// Output strides based on padded seqlen.
prep.nhead_stride_q_hat   = seqlen_q_padded * (hdim / 2);
prep.batch_stride_q_hat   = nhead * seqlen_q_padded * (hdim / 2);
prep.nhead_stride_q_scale = seqlen_q_padded * (hdim / 32);
prep.batch_stride_q_scale = nhead * seqlen_q_padded * (hdim / 32);
prep.nhead_stride_q_mean  = num_q_tiles_pad * hdim;
prep.batch_stride_q_mean  = nhead * num_q_tiles_pad * hdim;
prep.nhead_stride_k_hat   = seqlen_k_padded * (hdim / 2);
prep.batch_stride_k_hat   = nhead * seqlen_k_padded * (hdim / 2);
prep.nhead_stride_k_scale = seqlen_k_padded * (hdim / 32);
prep.batch_stride_k_scale = nhead * seqlen_k_padded * (hdim / 32);
prep.k_mean_ptr           = k_mean_buf;
prep.k_prime_ptr          = k_prime_buf;
prep.stride_k_prime       = hdim;
prep.nhead_stride_k_prime = seqlen_k_padded * hdim;
prep.batch_stride_k_prime = nhead * seqlen_k_padded * hdim;
```

In **Launch 1b** (SageAttnVPreprocessKernel), set seqlen_k fields:
```cpp
v_hargs.seqlen_k             = seqlen_k_padded;   // used for GridSize
v_hargs.seqlen_k_real        = seqlen_k;           // used for input bounds check
// Override output strides to padded seqlen
v_hargs.stride_v_hat         = seqlen_k_padded / 2;
v_hargs.nhead_stride_v_hat   = hdim * (seqlen_k_padded / 2);
v_hargs.batch_stride_v_hat   = nhead * hdim * (seqlen_k_padded / 2);
v_hargs.stride_v_scale       = seqlen_k_padded / 32;
v_hargs.nhead_stride_v_scale = hdim * (seqlen_k_padded / 32);
v_hargs.batch_stride_v_scale = nhead * hdim * (seqlen_k_padded / 32);
```

In **Launch 2** (BatchedGemm), replace `N = seqlen_k` with:
```cpp
const index_t N = seqlen_k_padded;
```
(M = num_q_tiles_pad, K = hdim unchanged.)

Also update the batch strides:
```cpp
/*batch_stride_B=*/N * K,    // seqlen_k_padded * hdim
/*batch_stride_C=*/M * N,    // num_q_tiles * seqlen_k_padded
```

In **Launch 0** (SageAttnKMeanKernel), update:
```cpp
kmean_hargs.num_k_tiles = num_k_tiles_pad;
```

---

### Task 4: Update test — allocate padded buffers and add irregular cases

**Files:**
- Modify: `test/ck_tile/fmha/test_fmha_sageattn_preprocess.cpp`

#### 4a — Replace buffer allocation with `get_buffer_sizes`

In `RunGPUTest<InputT>()`, replace the section that computes `num_q_tiles` / `num_k_tiles` and allocates output buffers with:

```cpp
// Use get_buffer_sizes to get padded dimensions.
const auto bsz = (hd == 128)
    ? ck_tile::get_buffer_sizes<InputT, 128>(b, h, sq, sk, hd)
    : ck_tile::get_buffer_sizes<InputT, 128>(b, h, sq, sk, hd); // always kRows=128

const int sq_pad      = static_cast<int>(bsz.seqlen_q_padded);
const int sk_pad      = static_cast<int>(bsz.seqlen_k_padded);
const int num_q_tiles = static_cast<int>(bsz.num_q_tiles);
const int num_k_tiles = static_cast<int>(bsz.num_k_tiles);
```

Then allocate output GPU buffers using padded sizes:
```cpp
ck_tile::DeviceMem q_hat_dev(bsz.q_hat_bytes);
ck_tile::DeviceMem q_scale_dev(bsz.q_scale_bytes);
ck_tile::DeviceMem q_mean_dev(bsz.q_mean_bytes);
ck_tile::DeviceMem k_hat_dev(bsz.k_hat_bytes);
ck_tile::DeviceMem k_scale_dev(bsz.k_scale_bytes);
ck_tile::DeviceMem delta_s_dev(bsz.delta_s_bytes);
ck_tile::DeviceMem v_hat_dev(bsz.v_hat_bytes);
ck_tile::DeviceMem v_scale_dev(bsz.v_scale_bytes);

ck_tile::DeviceMem k_mean_buf(bsz.k_mean_bytes);
ck_tile::DeviceMem k_prime_buf(bsz.k_prime_bytes);
ck_tile::DeviceMem k_mean_partial_buf(bsz.k_mean_partial_bytes);
ck_tile::DeviceMem counter_buf(bsz.counter_bytes);
```

#### 4b — Update hargs to use padded output strides

In the hargs setup block, replace all `sq`-based output strides with `sq_pad` and `sk`-based output strides with `sk_pad`. Input Q/K/V strides remain real-seqlen based:

```cpp
// Q hat / scale strides: padded seqlen
hargs.stride_q_hat         = hd / 2;
hargs.nhead_stride_q_hat   = sq_pad * (hd / 2);
hargs.batch_stride_q_hat   = h * sq_pad * (hd / 2);
hargs.stride_q_scale       = hd / kG;
hargs.nhead_stride_q_scale = sq_pad * (hd / kG);
hargs.batch_stride_q_scale = h * sq_pad * (hd / kG);
hargs.stride_q_mean        = hd;
hargs.nhead_stride_q_mean  = num_q_tiles * hd;
hargs.batch_stride_q_mean  = h * num_q_tiles * hd;

// K hat / scale strides: padded seqlen
hargs.stride_k_hat         = hd / 2;
hargs.nhead_stride_k_hat   = sk_pad * (hd / 2);
hargs.batch_stride_k_hat   = h * sk_pad * (hd / 2);
hargs.stride_k_scale       = hd / kG;
hargs.nhead_stride_k_scale = sk_pad * (hd / kG);
hargs.batch_stride_k_scale = h * sk_pad * (hd / kG);

// V hat / scale strides: padded seqlen
hargs.stride_v_hat         = sk_pad / 2;
hargs.nhead_stride_v_hat   = hd * (sk_pad / 2);
hargs.batch_stride_v_hat   = h * hd * (sk_pad / 2);
hargs.stride_v_scale       = sk_pad / kG;
hargs.nhead_stride_v_scale = hd * (sk_pad / kG);
hargs.batch_stride_v_scale = h * hd * (sk_pad / kG);

hargs.num_q_tiles = num_q_tiles;
hargs.num_k_tiles = num_k_tiles;
```

(sageattn_preprocess_run will override output strides internally anyway,
but setting them correctly here keeps hargs consistent for debug.)

#### 4c — Update CPU readback buffer sizes to padded dims

```cpp
std::vector<float>   delta_s_gpu(b * h * num_q_tiles * sk_pad);
std::vector<uint8_t> q_hat_gpu(b * h * sq_pad * (hd / 2));
std::vector<uint8_t> q_scale_gpu(b * h * sq_pad * (hd / kG));
std::vector<uint8_t> k_hat_gpu(b * h * sk_pad * (hd / 2));
std::vector<uint8_t> k_scale_gpu(b * h * sk_pad * (hd / kG));
std::vector<uint8_t> v_hat_gpu(b * h * hd * (sk_pad / 2));
std::vector<uint8_t> v_scale_gpu(b * h * hd * (sk_pad / kG));
```

Also update the q_mean readback:
```cpp
std::vector<InputT>  q_mean_gpu_raw(b * h * num_q_tiles * hd);
```

#### 4d — Scope verification loops to real seqlen (not padded)

delta_s check: inner loop `kj < sk` (not `sk_pad`).
q_hat/q_scale check: loop `n < sq` (not `sq_pad`).
k_hat/k_scale check: loop `n < sk` (not `sk_pad`).
v_hat/v_scale check: loop `n < sk` (not `sk_pad`).

For delta_s, the GPU stride between k-columns is `sk_pad` now, so update the offset formula:
```cpp
const int off = bi * h * num_q_tiles * sk_pad + hi * num_q_tiles * sk_pad + qi * sk_pad + kj;
```

For q_hat GPU (nhead_stride = sq_pad * hd/2):
```cpp
// existing reference uses sq-based layout; GPU uses sq_pad-based layout
// When verifying, construct GPU offset using sq_pad stride:
// e.g., for row n in [0,sq): gpu_offset = bi*h*sq_pad*(hd/2) + hi*sq_pad*(hd/2) + n*(hd/2)
```

The dequant helper `reference_sageattn_dequant_mxfp4` takes `rows` and `cols` — pass `sq_pad` / `sk_pad` as rows but verify results only for valid rows. Simpler: just pass `sq` / `sk` for rows and pass the padded stride manually. Actually the reference is a flat array operation, so pass the real dimensions for the reference dequant and compare only valid elements from the GPU padded buffer.

To avoid rewriting the dequant call, extract only the valid rows from GPU buffers into a separate `sq`-row vector before calling dequant:

For Q hat verification:
```cpp
std::vector<uint8_t> q_hat_real(b * h * sq * (hd / 2));
for(int bi = 0; bi < b; bi++)
    for(int hi = 0; hi < h; hi++)
        for(int n = 0; n < sq; n++)
        {
            const int src_off = bi*h*sq_pad*(hd/2) + hi*sq_pad*(hd/2) + n*(hd/2);
            const int dst_off = bi*h*sq*(hd/2)     + hi*sq*(hd/2)     + n*(hd/2);
            std::copy(q_hat_gpu.begin()+src_off, q_hat_gpu.begin()+src_off+(hd/2),
                      q_hat_real.begin()+dst_off);
        }
// Then call dequant with q_hat_real and sq
```

Apply the same extraction for q_scale, k_hat, k_scale.

For V (transposed layout `[B,H,hdim,sk/2]`), extract columns `0..sk-1` from each row:
```cpp
std::vector<uint8_t> v_hat_real(b * h * hd * (sk / 2));
for(int bi = 0; bi < b; bi++)
    for(int hi = 0; hi < h; hi++)
        for(int d = 0; d < hd; d++)
        {
            const int src_off = bi*h*hd*(sk_pad/2) + hi*hd*(sk_pad/2) + d*(sk_pad/2);
            const int dst_off = bi*h*hd*(sk/2)     + hi*hd*(sk/2)     + d*(sk/2);
            std::copy(v_hat_gpu.begin()+src_off, v_hat_gpu.begin()+src_off+(sk/2),
                      v_hat_real.begin()+dst_off);
        }
```

Then call `reference_sageattn_dequant_mxfp4` with `v_hat_real` and `sk` as cols.

Apply same for v_scale (cols = sk/32 per row).

#### 4e — Add irregular test cases

Replace (or extend) `INSTANTIATE_TEST_SUITE_P`:

```cpp
INSTANTIATE_TEST_SUITE_P(
    Shapes,
    SageAttnPreprocessTest,
    ::testing::Values(
        // --- existing aligned cases ---
        std::make_tuple(1, 1, 256, 128, 128, true),
        std::make_tuple(2, 4, 128, 128, 128, true),
        std::make_tuple(1, 2, 128, 256, 128, true),
        std::make_tuple(1, 1, 128, 128, 256, true),
        std::make_tuple(1, 2, 128, 256, 256, true),
        // --- irregular seqlen (non-multiples of kRows=128) ---
        std::make_tuple(1, 1,  65,  96, 128, true),   // both < kRows
        std::make_tuple(1, 1, 127, 127, 128, true),   // both = kRows-1
        std::make_tuple(1, 2, 130, 100, 128, true),   // sq > kRows, sk < kRows
        std::make_tuple(2, 4, 300, 200, 128, true),   // multi-tile, both misaligned
        std::make_tuple(1, 1,  65,  96, 256, true),   // hdim=256 + misaligned
        std::make_tuple(1, 2, 300, 200, 256, true)    // hdim=256 + multi-tile
    ));
```

Remove the `ASSERT_EQ(sk % kG, 0)` guard — after padding, V alignment is guaranteed internally.
Keep `ASSERT_EQ(hd % kG, 0)` (hdim must still be divisible by 32).

---

### Task 5: Build and verify

**Step 1: Find the test target name**
```bash
cd /root/rocm-libraries/projects/composablekernel/build
grep -r "sageattn_preprocess" CMakeFiles/ --include="*.txt" -l 2>/dev/null | head -3
# or:
ninja -t targets | grep sageattn_preprocess
```

**Step 2: Build**
```bash
cd /root/rocm-libraries/projects/composablekernel/build
cmake --build . --target test_ck_tile_fmha_sageattn_preprocess -j$(nproc)
```
Expected: zero errors. If the target name differs, use the one found in Step 1.

**Step 3: Run tests**
```bash
./bin/test_ck_tile_fmha_sageattn_preprocess
```
Expected: all cases PASS, including the new irregular-seqlen cases.

**Step 4: Commit**
```bash
cd /root/rocm-libraries/projects/composablekernel
git add include/ck_tile/ops/sageattn_preprocess/pipeline/sageattn_preprocess_pipeline.hpp \
        include/ck_tile/ops/sageattn_preprocess/kernel/sageattn_preprocess_kernel.hpp \
        include/ck_tile/ops/sageattn_preprocess/sageattn_preprocess.hpp \
        test/ck_tile/fmha/test_fmha_sageattn_preprocess.cpp
git commit -m "ck_tile/sageattn: pad seqlen to kRows internally, add get_buffer_sizes"
```
