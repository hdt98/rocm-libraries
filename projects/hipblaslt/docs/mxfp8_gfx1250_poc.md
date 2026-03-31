# MXFP8 gfx1250 POC Bring-up

This note describes the minimum performance suite and run command for MXFP8 bring-up on `gfx1250`.

## Scope

- Datatypes: `A/B=f8_r`, `C/D=f32_r`, `compute_type=c_f32_r`
- Layout: `transA=T`, `transB=N`
- Scale modes:
  - `scaleA/scaleB=3` (`Block_32_UE8M0`)
  - `scaleA/scaleB=1001` (`Block_32_UE8M0_32_8_EXT`)
- Minimal shapes:
  - `256x256x256`
  - `4096x4096x4096`
- Extension shape:
  - `7168x4096x16384`

## Where the POC Problem Set Lives

- YAML: `clients/scripts/performance/problems/matmul_mxfp8_gfx1250_poc_bench.yaml`
- Suite name: `mxfp8_gfx1250_poc` in `clients/scripts/performance/suites.py`
- Runner: `clients/scripts/performance/run_mxfp8_gfx1250_poc.sh`

## Run

From source tree:

```bash
cd projects/hipblaslt/clients/scripts/performance
./run_mxfp8_gfx1250_poc.sh --exec-folder ../../../build/release/clients
```

If clients are built in a custom location:

```bash
./run_mxfp8_gfx1250_poc.sh --exec-folder /path/to/clients --samples 1 --tag debug
```

## Acceptance (POC)

POC is considered successful when:

1. The suite finds at least one supported solution for each minimal shape and both scale modes.
2. The benchmark run exits successfully and produces non-empty CSV results.
3. There is no `No solution found` for all minimal cases.
4. Baseline non-MX FP8 regression remains green.

## Notes

- If the run reports no supported solution for all cases, this usually indicates missing logic/solution coverage rather than a runtime API wiring issue.
- This POC is intentionally small to unblock offline tuning iteration.
