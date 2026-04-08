# kpack-cli

CLI tool for inspecting `.kpack` kernel archives.

## Requirements

```bash
pip install msgpack
```

## Usage

```
kpack [-h] {info,list,ls,spec,blobs,toc,header,diff} ...
```

### `kpack info` — Archive summary

```
$ kpack info gemm.kpack
kpack gemm.kpack

  Version:             1
  File size:           27.8 MiB
  Kernel data:         27.7 MiB
  TOC size:            14.5 KiB
  Compression:         zstd-per-kernel
  Blobs:               0
  Kernels:             24
  Variant specs:       24
  Architectures:       gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx90a gfx942 gfx950
  Zstd blob:           offset=16, size=27.7 MiB
```

### `kpack list` — List kernels

Lists all kernel variants with their type, target architectures, and per-arch blob sizes.

```
$ kpack list gemm.kpack
  Kernel                         Type         Architectures                Blob sizes
  ────────────────────────────────────────────────────────────────────────────────────
  gemm_bf16                      GEMM         gfx90a gfx942 gfx950        10.0 MiB, 9.5 MiB, 9.5 MiB
  gemm_fp16                      GEMM         gfx90a gfx942 gfx950        10.0 MiB, 9.5 MiB, 9.5 MiB
  gemm_fp16_add                  GEMM         gfx90a gfx942 gfx950        10.6 MiB, 10.2 MiB, 10.2 MiB
  gemm_fp16_wmma                 GEMM         gfx1100 gfx1101 ... gfx1151 13.6 MiB, 13.6 MiB, ...
  gemm_fp8_fnuz                  GEMM         gfx942 gfx950               10.3 MiB, 10.3 MiB
  gemm_i4_bquant                 GEMM         gfx942 gfx950               15.8 MiB, 15.9 MiB
  gemm_i8                        GEMM         gfx942 gfx950               12.5 MiB, 12.5 MiB
  ...
```

### `kpack spec` — Show variant specs

Shows the full spec for a kernel variant — every field from the kpack TOC, pretty-printed.

```
$ kpack spec gemm.kpack gemm_fp16_add
  gemm_fp16_add  GemmSpec
  targets:                 gfx90a gfx942 gfx950
  physical_tensors:
                            - {name=A, dtype=FP16, layout=Row, args_slot=0}
                            - {name=B, dtype=FP16, layout=Col, args_slot=1}
                            - {name=D, dtype=FP16, layout=Row, args_slot=2}
                            - {name=bias, dtype=FP16, layout=Row, args_slot=3}
  acc_dtype:               FP32
  block_tile:              {m=128, n=128, k=32}
  block_waves:             {m=2, n=2, k=1}
  wave_tile:               {m=16, n=16, k=16}
  workgroup_size:          256
  k_batch:                 1
  pipeline:                V1
  pipeline_scheduler:      Intrawave
  tile_partitioner:        Linear
  epilogue_ops:            [Add]
  store_strategy:          CShuffle
  pad_m:                   false
  pad_n:                   false
  group_size:              0
```

Omit the variant name to show all specs.

### `kpack blobs` — Show blob layout

Shows the offset and size of each code object blob in the data section, mapped to its kernel variant and architecture.

```
$ kpack blobs gemm.kpack
  #     Offset         Size           Kernel
  ────────────────────────────────────────────────────────────────────────────────────
  0     0x10           9.5 MiB        gemm_fp16/gfx90a
  1     0x98a010       9.1 MiB        gemm_fp16/gfx942
  ...
```

Note: archives using `zstd-per-kernel` compression store all code objects in a single compressed blob rather than individual entries, so `blobs` may report `(no blobs)`.

### `kpack header` — Show raw header bytes

```
$ kpack header gemm.kpack
  Offset  Hex                                    Decoded
  ────────────────────────────────────────────────────────────
  0x00    4b 50 41 4b                             b'KPAK'
  0x04    01 00 00 00                             version = 1
  0x08    c0 e0 bb 01 00 00 00 00                 toc_offset = 29089984 (0x1bbe0c0)
```

### `kpack toc` — Dump raw TOC as JSON

Dumps the full table of contents as JSON for programmatic consumption.

```
$ kpack toc gemm.kpack | head -12
{
  "compression_scheme": "zstd-per-kernel",
  "gfx_arches": [
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx90a",
    "gfx942",
    "gfx950"
  ],
  ...
}
```

### `kpack diff` — Compare two kpack files

Side-by-side comparison of two archives: file sizes, kernel counts, architecture coverage, and which kernels exist in only one file.

```
$ kpack diff old.kpack new.kpack
                        File size Left                     Right
  ────────────────────────────────────────────────────────────────────────────────────
                        File size 25.1 MiB                 = 27.8 MiB
                          Version 1                        = 1
                     Architectures [gfx90a, gfx942]        != [gfx90a, gfx942, gfx950]
                  Kernels (total) 20                       24
  ...
```

## Color output

Output is colorized by default. Set `NO_COLOR=1` to disable.
