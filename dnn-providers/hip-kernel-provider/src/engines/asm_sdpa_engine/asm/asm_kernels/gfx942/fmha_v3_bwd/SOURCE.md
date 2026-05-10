# Source Provenance — gfx942 / fmha_v3_bwd

The CSVs and `.co` binaries in this directory are a snapshot of the FMHA
backward kernels published by the AITER project.

| Field | Value |
|---|---|
| Upstream repository | https://github.com/ROCm/aiter |
| Source commit | `9522048dc10de20ba9dcda1c0a3f640867e7a586` |
| Source path | `hsa/gfx942/fmha_v3_bwd/` |

`fmha_bwd_dq_shuffle.csv` is intentionally omitted; in the upstream snapshot it
contains only a header row and no kernel rows.

## Local override

`bwd_hd128_odo_bf16.co` is **not** the upstream AITER binary. The upstream
version (9344 bytes) produces D-aux values that diverge from the in-tree CPU
backward reference enough to fail the existing hd128/bf16 SDPA backward
correctness test. The version held here (10280 bytes) is the same binary
that develop has shipped against and that the CPU reference was tuned to.
Future AITER refreshes must either keep this override in place or coordinate
a tolerance / reference update.

This file documents the provenance for the binary-versioning story tracked under
ALMIOPEN-1828; future updates should refresh both the binaries and this record
in lockstep.
