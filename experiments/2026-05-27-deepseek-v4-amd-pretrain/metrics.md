# DeepSeek V4-Pro Layer Microbench Metrics

Status as of 2026-06-05: standalone single-layer PyTorch reference and FP4 packed-sim ceiling runs remain the single-GPU math reference, while the current best 4xMI350 TorchTitan canary is the MORI+AITER compact owner-VJP run with saved forward dispatch reuse recorded in [`run_artifacts/torchtitan_mori_saved_dispatch_reuse_ab_20260605.json`](run_artifacts/torchtitan_mori_saved_dispatch_reuse_ab_20260605.json). It sustains `12,570.33` tok/GPU/s, `~50,281.3` node tok/s, and `237.19 TFLOPs/GPU` over steps 4-6 at `S=4096`, `GBS=128`, `MBS=4`, clearing the current 4x target of `45-47k` node tok/s. Peak logged memory is `115.04 GiB`.

## 2026-06-08 8xMI350 Standard-EP Owner-Compact Row Gate

The owner-compact row materialization gate for the current 12-layer standard-EP
top-8 helper branch completed six steps on `do-vunguyen-mi350-gpu`, but is a
neutral result rather than a new promoted baseline. Artifact:
[`run_artifacts/torchtitan_dsv4_flash12_standard_ep_top8_ownercompact_rows_selected_weight_decision_20260608.json`](run_artifacts/torchtitan_dsv4_flash12_standard_ep_top8_ownercompact_rows_selected_weight_decision_20260608.json).

| Variant | steps 2-6 tok/GPU/s | steps 2-6 TFLOPs/GPU | steps 4-6 tok/GPU/s | Peak logged memory |
| --- | ---: | ---: | ---: | ---: |
| retained top-8 SDMA | `8,029.0` | `406.522` | `8,134.0` | `193.80 GiB` |
| owner-compact rows + selected weights | `8,029.4` | `406.532` | `8,154.33` | `194.28 GiB` |

The timing sidecar shows the mechanism worked: helper-side gather/sort/count
work drops from `46.53 ms` rank-max in the presort-off trace to `0.35 ms` with
owner-compact materialized rows. The e2e result stays neutral because the
remaining larger buckets are hot-weight movement (`196.43 ms`), helper grouped
MLP/VJP (`162.53 ms`), and weighted scatter (`117.06 ms`). Decision: keep the
code path, but do not spend the next branch on more presort/materialize knobs.

## 2026-06-05 4xMI350 MORI Compact Owner VJP

The promoted 4x run uses direct TorchTitan, AdamW, `TP=1`, `PP=1`, `CP=1`, `FSDP=4`, `EP=4`, no activation checkpointing, checkpoint disabled, MORI+AITER fused forward, routed backward, compact `expert_loop` owner VJP, saved forward dispatch reuse, Triton SwiGLU, compact xgrad, expandable allocator, and CE chunks 8. Raw log:

`/scratch/sonle5/dsv4_pretrain_canary_20260527/logs/torchtitan_dsv4_mori_saved_dispatch_expert_loop_stability_4x_s4096_gbs128_mbs4_steps6_20260605/docker-run.log`

| Step window | tok/GPU/s | node tok/s | TFLOPs/GPU | Peak logged memory |
| --- | ---: | ---: | ---: | ---: |
| steps 2-6 mean | n/a | n/a | `236.192` | `115.04 GiB` |
| steps 3-6 mean | n/a | n/a | `237.32` | `115.04 GiB` |
| steps 4-6 mean | `12,570.33` | `50,281.32` | `237.19` | `115.04 GiB` |

The saved-dispatch A/B answers whether forward intermediates are worth reusing for this path. With `TORCHTITAN_MORI_AITER_SAVE_FORWARD_DISPATCH=true`, backward reuses compact AITER forward dispatch state (`recv_count`, `src_pos`, `packed_x_local`, `packed_weight_local`, `packed_ids_local`) instead of rebuilding the backward dispatch input. The no-save expert-loop control averaged `12,006.67` tok/GPU/s, `~48,026.7` node tok/s, and `227.19 TFLOPs/GPU` over steps 4-6, so saved dispatch gives a materialized late-step uplift of about `4.4%` in TFLOPs/GPU and `4.7%` in tok/GPU/s, with peak logged memory increasing from `111.25 GiB` to `115.04 GiB`.

Ascend MoE/EP source mapping is now persisted in [`changesets/torchtitan_mori_aiter_ascend_moe_ep_lessons.md`](changesets/torchtitan_mori_aiter_ascend_moe_ep_lessons.md) and [`run_artifacts/ascend_moe_ep_lessons_amd_mapping_20260605.json`](run_artifacts/ascend_moe_ep_lessons_amd_mapping_20260605.json). The key read is that Ascend does not prove "save every forward tensor"; it proves compact route metadata, native grouped expert GEMM, token permute/unpermute, independent FSDP/EP communication domains, and vector-fusion/codegen are the high-return mechanisms. For AMD, this keeps the production target as AITER fused forward plus compact saved route/intermediate state plus fused grouped owner-rank training backward. Padded slabs and Python recompute adapters stay demoted.

Comparison against the padded MORI owner-VJP line is the key diagnosis. The route-padding profile shows learned-router collapse causing `22.03x` mean and `44.204x` max zero-padding in the padded owner VJP. The older streamed-SwiGLU padded six-step run settled at `147.04 TFLOPs/GPU` late with `246.03 GiB` peak memory. The compact expert-loop path is therefore `1.54x` faster late-step and uses about `45%` of the peak memory. Read: MORI EP/MoE is currently dominated by route-imbalance-induced padded GEMM work and transient materialization plus dX-return skew, not raw MORI collective bandwidth alone.

This is a retained canary baseline, not the final kernel. The next production path should preserve AITER fused forward, reuse compact saved route/intermediate state, then replace Python per-expert `expert_loop` with a fused compact grouped owner-rank training backward.

Follow-up profiling for the same retained shape is captured in [`run_artifacts/torchtitan_mori_saved_dispatch_profile_4x_s4096_20260605.json`](run_artifacts/torchtitan_mori_saved_dispatch_profile_4x_s4096_20260605.json). The timing run completed two steps and logged step 2 at `12,050` tok/GPU/s, `228.01 TFLOPs/GPU`, and `112.24 GiB` peak memory; because the MORI timing hook synchronizes recorded calls, this is attribution evidence rather than a throughput promotion. The useful answer on forward-state reuse is now concrete: forward saved-dispatch overhead is tiny (`~9.7 ms` max-rank total across step 2), backward reuse is tiny (`~0.2 ms` max-rank total), but the retained A/B still materializes a `~4.4%` TFLOPs uplift because it avoids rebuilding compact dispatch inputs in backward.

The remaining MORI profile wall is not the saved-dispatch path. Step-2 MORI backward top-level time is `~1.87 s` max-rank under timing; the largest stages are `backward.local_vjp.expert_loop` at `~1.28 s`, `backward.combine_dx.call` at `~0.52 s`, and `backward.dispatch_grad_output.call` at `~0.22 s`. Inside the local VJP, metadata work is small (`gather ~20.8 ms`, `sort_counts ~16.4 ms`, `route_pack ~4.3 ms` max-rank total) while the Python per-expert math loop is the wall (`~1.21 s`). Route stats still show learned-router imbalance: mean `12.326x` and max `21.264x` padding multiplier if the owner VJP were padded. Read: reuse compact forward dispatch state, but do not save blind padded slabs; the production target is AITER-forward-compatible saved compact state plus a fused grouped owner-rank training backward and earlier/overlapped dX return.

## 2026-06-05 AITER Compact Owner VJP Negative Promotion

The AITER grouped-GEMM owner-VJP branch is recorded in [`run_artifacts/torchtitan_mori_aiter_compact_vjp_materialization_20260605.json`](run_artifacts/torchtitan_mori_aiter_compact_vjp_materialization_20260605.json). It is executable at the target `4xMI350`, `S=4096`, `GBS=128`, `MBS=4`, no-AC shape, but it does not replace the retained expert-loop baseline.

Uncached `local_vjp_mode=aiter_compact` completed six steps but averaged only `11,564.67` tok/GPU/s and `218.82 TFLOPs/GPU` over steps 4-6, with `125.25 GiB` peak logged memory. A transposed-weight LRU cache improved the branch to `12,246.0` tok/GPU/s and `231.93 TFLOPs/GPU` over steps 4-6, but peak memory rose to `134.10 GiB`, and it still trails the retained expert-loop baseline's `12,570.33` tok/GPU/s, `237.19 TFLOPs/GPU`, and `115.04 GiB`.

The profile diagnosis is useful: `aiter_compact` spends its local-VJP median time in backward forward-recompute (`~10.8 ms`), wgrad (`~11.4 ms`), xgrad (`~7.8 ms`), and prepare/layout work (`~7.9 ms`). The transposed-weight cache did not materially reduce `prepare_ops`, likely because FSDP/local expert tensor identity or data pointer changes across calls. Decision: keep `aiter_compact` as a diagnostic branch, not a promoted baseline. The production direction is still AITER-forward-compatible compact saved intermediates or a fused grouped owner-rank training backward, not a Python adapter that recomputes all forward intermediates and prepares weight layouts per owner VJP.

## 2026-06-03 EP8 Additive Wall-Time Decomposition

The first rank-aware additive timing pass for the retained 8xMI350 TorchTitan DSv4-style candidate is recorded in [`run_artifacts/torchtitan_additive_wall_time_decomposition_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_additive_wall_time_decomposition_ep8_s4096_gbs128_20260603.json). The follow-up backward/deepbreak pass is recorded in [`run_artifacts/torchtitan_additive_deepbreak_backward_decomposition_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_additive_deepbreak_backward_decomposition_ep8_s4096_gbs128_20260603.json). Shape: `TP1 PP1 CP1 FSDP8 EP8`, `MBS=1`, `S=4096`, `GBS=128`, full activation checkpointing, MORI routed backward, padded owner VJP, Triton SwiGLU, CE2, checkpoint disabled.

The clean finalize-marker additive run completed at step 2 with `2,831` tok/GPU/s, `53.56` TFLOPs/GPU, and `21.61 s` max-rank step wall. The additive outer waterfall is almost entirely `step.train_step`; inside it, the 16-microbatch forward/backward loop is the wall at about `21.24 s`. Other additive buckets are small: accumulated-loss reduce about `0.31 s`, fetch about `0.03 s`, grad clip about `0.02 s`, optimizer about `0.02 s`, and metrics/valid-token reductions are single-digit milliseconds.

Nested microbatch diagnostics show backward as the largest named child: the clean finalize run has backward about `13.93 s`, forward about `2.25 s`, loss about `0.47 s`, and `post_backward_finalize` only about `0.0005 s`. The finer deepbreak run added labels for `train_context_enter`, `train_context_exit`, `drop_pred`, and `return_loss`; these are all sub-ms totals and do not explain the remaining `4.47-4.55 s` same-rank wrapper residual. That residual repeats at about `0.29-0.33 s` for microbatches 1-15, while microbatch 0 is near zero. Next wrapper target is therefore lower-level autograd/FSDP/checkpoint scheduling work or timer-boundary attribution, not `pred_to_local`, train-context exit, or post-backward finalization.

The deepbreak backward split ranks the current optimization targets. In the sidecar run, step 2 logged `2,789` tok/GPU/s and `52.78` TFLOPs/GPU. `forward_backward.backward` was `14.31 s` max-rank under profiling. MORI routed MoE backward/return is the largest confirmed bucket at about `6.79 s` max-rank immediate-stage GPU time: `backward.local_vjp.padded` about `2.64 s`, `backward.combine_dx.call` about `2.37 s`, `backward.dispatch_x.call` about `0.25 s`, and the remaining dispatch/combine pieces are smaller. Inside local padded VJP, the leading pieces are `w13_xgrad_gemm_reduce` about `0.54 s`, `w13_forward_gemm` about `0.50 s`, `w13_wgrad_gemm` about `0.44 s`, `w2_dgrad_gemm` about `0.26 s`, `w2_forward_gemm` about `0.25 s`, and `w2_wgrad_gemm` about `0.22 s`.

Dense projection backward is not the main wall in this EP8 run: chunked CE/lm-head internals are about `0.46 s`, attention projection linear backward about `0.26 s`, shared-expert/dense-FFN linear backward about `0.06 s`, and router gate backward about `0.01 s`. The rough remaining backward remainder after those named buckets is about `7.4 s`; it likely includes activation-checkpoint recompute, attention core backward, FSDP/autograd hooks, and uninstrumented ops. Use this as a ranking surface, not a clean replacement for the no-profiler throughput anchor.

The sync-gap plus activation-checkpoint block follow-up is recorded in [`run_artifacts/torchtitan_additive_syncgap_acblocks_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_additive_syncgap_acblocks_ep8_s4096_gbs128_20260603.json). This profiling run logged step 2 at `2,842` tok/GPU/s, `53.78` TFLOPs/GPU, and `21.524 s` max-rank `step.train_step`, so it should be used for attribution rather than as the throughput anchor. The outer `train_step.microbatch_forward_backward` phase is `21.160 s`; its named children are `2.341 s` forward, `0.477 s` loss, and `13.865 s` backward, leaving a `4.534 s` wrapper residual. The sync split rules out a hidden CUDA sync at the wrapper exit: post-sync is only `3.31 ms` max. The residual is almost entirely `gap_after__last_forward_backward_child`, `4.519 s` max and `4.488 s` mean, after `forward_backward.return_loss` and before the outer wrapper exits. The next A/B target is TorchTitan structured logging/decorator unwind around `@sl.log_trace_span("fwd_bwd")`, not model math.

The same follow-up refines the backward buckets. Activation-checkpoint recompute is now directly visible at `2.560 s` max-rank, while the original checkpointed block forward costs `2.068 s`. MORI routed MoE backward remains the largest named backward subsystem: local owner VJP is `2.637 s` max, `combine_dx` is `2.094 s`, and dispatch-grad inputs are `0.309 s`. Inside the local VJP, the main pieces are recompute forward `0.815 s`, wgrad `0.668 s`, xgrad/reduce `0.555 s`, W2 dgrad plus SwiGLU `0.336 s`, and pack/scatter `0.243 s`. Dense projection linear backward is still small in comparison: attention projection `0.255 s`, shared expert `0.062 s`, and router `0.012 s`. CE2/lm-head work is under the `loss` phase, not the later `backward` phase; the chunked loss sidecar shows lm-head forward `0.090 s + 0.035 s`, cross entropy forward `0.017 s + 0.017 s`, and CE chunk backward `0.107 s + 0.214 s`.

Structured-logging A/B: [`run_artifacts/torchtitan_structlog_off_ab_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_structlog_off_ab_ep8_s4096_gbs128_20260603.json) reruns the same `8xMI350`, `S=4096`, `GBS=128`, MORI routed-backward diagnostic with `CANARY_ENABLE_STRUCTURED_LOGGING=false` / `--debug.no-enable-structured-logging`. The first attempt used the wrong tyro spelling, `--debug.enable_structured_logging=False`, and failed at config parse before training; the successful `20260603b` run logged step 2 at `2,764` tok/GPU/s, `52.29` TFLOPs/GPU, and `10.07720` loss. Disabling structured logging does not remove the wrapper gap: `train_step.microbatch_forward_backward` is `21.815 s` max, forward `2.382 s`, loss `0.476 s`, backward `14.534 s`, wrapper residual `4.552 s`, and `gap_after__last_forward_backward_child` `4.545 s`. The wrapper post-sync remains tiny at `3.14 ms`. This rules out structured JSONL trace overhead as the 4.5 s residual; the next branch is autograd/FSDP/checkpoint scheduling or an unlabelled operation at the function-unwind boundary after `forward_backward.return_loss`.

Call-boundary release diagnostic: [`run_artifacts/torchtitan_callboundary_release_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_callboundary_release_ep8_s4096_gbs128_20260603.json) adds explicit `release_tensors`, `train_step.forward_backward_call`, `train_step.forward_backward_call_returned`, and `loss_detach_append` markers, keeps structured logging disabled, and filters model/MORI profile sidecars to step 2. The run logged step 2 at `2,860` tok/GPU/s, `54.12` TFLOPs/GPU, and `9.50722` loss. The additive max-rank step is `21.315 s`; `train_step.microbatch_forward_backward` is `20.947 s`, with named children forward `2.302 s`, loss `0.470 s`, and backward `13.654 s`.

The wrapper gap is now localized more tightly. The old major-child residual is `4.617 s` max-rank, and the refined residual inside `train_step.forward_backward_call` is `4.605 s`. After the fwd/bwd call returns, the outer residual is only `2.51 ms`; `forward_backward_call_returned` is only `0.90 ms`, `loss_detach_append` is `1.09 ms`, and explicit `release_tensors` is `0.47 ms`. Microbatch 0 has only `0.24 ms` call-boundary residual, while microbatches 1-15 show `277-338 ms` each, p50 `305.7 ms`, which sums to the full `4.5 s` gap. So the wall is not normal tensor release, not after-call bookkeeping, and not structured logging; it is synchronous work hidden at the `forward_backward_step` return/unwind boundary after `forward_backward.return_loss`.

The same run gives the current step-2 backward breakdown. `forward_backward.backward` is `13.654 s` max-rank. Named second-step buckets are MORI routed MoE backward top-level `3.536 s`, activation-checkpoint recompute `2.500 s`, and dense linear backward `0.325 s`; the same-rank residual is still about `7.72 s`, so the next profile pass must instrument attention-core backward, FSDP/autograd hooks, and checkpoint-engine internals rather than only MORI. Inside MORI, the biggest max-rank top-level stages are local padded VJP `2.498 s`, `combine_dx` `1.713 s`, and dispatch-x `0.234 s`. Inside local padded VJP, the leading pieces remain W1/W3 xgrad+reduce `0.509 s`, W1/W3 recompute forward GEMM `0.471 s`, W1/W3 wgrad `0.408 s`, W2 dgrad `0.240 s`, W2 recompute forward GEMM `0.237 s`, and W2 wgrad `0.210 s`.

Torch profiler follow-up: [`run_artifacts/torchtitan_torchprof_decomposition_v3_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_torchprof_decomposition_v3_ep8_s4096_gbs128_20260603.json) reruns the same `8xMI350`, `S4096`, `GBS128`, CE2, MORI routed padded VJP path with Kineto active on step 2. It logged `2,976 tok/GPU/s`, `56.31 TFLOPs/GPU`, loss `8.82204`, and a profiled `ProfilerStep#1` wall of `22.02 s`. The parser drops `320` FSDP async `finished` marker ranges per rank; without that filter, FSDP all-gather/reduce-scatter annotations falsely cover most of the trace.

The corrected autograd split explains most of the backward residual: `MmBackward0` is `8.92-9.05 s/rank`, `_MoriAiterForwardBackwardBackward` is `4.28-4.39 s/rank`, and `RegisterPostBackwardFunctionBackward` is only `0.36-0.45 s/rank`. User/kernel attribution shows MORI is still the largest named GPU subsystem (`moe.mori_aiter_bridge` `3.32-3.47 s/rank` kernel time), with `combine_dx.reorder_to_grad_order` up to `2.17 s/rank`, `dispatch_topk_dummy.total_recv_item` up to `1.90 s/rank`, and filtered FSDP post-backward ranges in the tens to low hundreds of ms. Dense attention in this guarded path is not the leading GPU wall in this trace: `dsv4.attention.dense_inner_attention` is only about `40-61 ms/rank` of attributed kernel time.

The important new wrapper clue is allocator/runtime synchronization. HIP runtime totals are large: `hipFree` is `9.33-9.56 s/rank`, `hipStreamSynchronize` `3.98-5.15 s/rank`, `hipMemcpyWithStream` `1.70-2.65 s/rank`, and `hipMalloc` `1.20-1.40 s/rank`. Autograd attribution splits `hipFree` almost exactly between `MmBackward0` (`4.68-4.80 s/rank`) and unattributed outer-step runtime (`4.59-4.76 s/rank`). That outer unattributed half lines up with the earlier `4.50-4.61 s` `forward_backward_step` wrapper residual after `return_loss`: the current best explanation is FSDP/autograd/checkpoint unwind causing synchronizing allocator frees, not ordinary Python bookkeeping or named model math.

Read: the next optimization branch should not only chase MoE kernels. The backward wall has two lead surfaces: (1) MORI routed backward skew/VJP, already identified, and (2) allocator/runtime synchronization dominated by `hipFree`, partly inside `MmBackward0` and partly at the wrapper unwind boundary. The next diagnostic A/B should target the allocator/free path directly, for example by changing the activation/checkpoint save/release contract or PyTorch HIP allocator behavior, and then rerunning the same call-boundary plus torch-profiler parser.

Persisted Torch-profiler parser: [`tools/dsv4_runtime_probe/summarize_torchprof_decomposition.py`](tools/dsv4_runtime_probe/summarize_torchprof_decomposition.py) now turns the raw Chrome traces into compact runtime/user/autograd/kernel attribution. It filters the long FSDP async all-gather/reduce-scatter marker spans that otherwise make user annotations look hundreds of seconds long. Reparsed baseline artifact: [`run_artifacts/torchtitan_torchprof_decomposition_parser_v2_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_torchprof_decomposition_parser_v2_ep8_s4096_gbs128_20260603.json).

Allocator A/B: [`run_artifacts/torchtitan_torchprof_decomposition_alloc_expand_v2_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_torchprof_decomposition_alloc_expand_v2_ep8_s4096_gbs128_20260603.json) reruns the same `8xMI350`, `S4096`, `GBS128`, `FSDP8`, `EP8`, CE2, MORI routed padded VJP profiler shape with `CANARY_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It logged step 2 at `3,108` tok/GPU/s, `58.82` TFLOPs/GPU, loss `9.63620`, and `ProfilerStep#1` wall `21.084 s`.

The allocator A/B does not move the wrapper culprit: `hipFree` changes only `9.425 s -> 9.390 s` mean per rank (`-0.37%`). The split remains the same: inside `MmBackward0`, `hipFree` is `4.732 s -> 4.730 s`; unattributed outer-step `hipFree` is `4.692 s -> 4.660 s`. That unattributed half is still the best profiler match for the additive `4.50-4.61 s` fwd/bwd wrapper residual after `forward_backward.return_loss`. So `expandable_segments` is not enough; the next free-path work should target FSDP/autograd/checkpoint tensor lifetime or reuse, not just allocator fragmentation.

The same A/B run did reduce MORI/wait-like runtime, probably from run-to-run skew rather than the allocator knob: `_MoriAiterForwardBackwardBackward` eval `4.349 s -> 3.440 s`, `dispatch_topk_dummy.total_recv_item` `1.876 s -> 1.072 s`, and `combine_dx.reorder_to_grad_order` `1.790 s -> 1.480 s`. `hipMemcpyWithStream` also fell `2.152 s -> 1.359 s`, while `hipStreamSynchronize` fell `4.489 s -> 4.128 s`. This explains the better profiled-step throughput, but it does not close the wrapper residual because `hipFree` is unchanged.

No-activation-checkpoint A/B: [`run_artifacts/torchtitan_noac_backward_wrapper_ab_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_noac_backward_wrapper_ab_ep8_s4096_gbs128_20260603.json), [`run_artifacts/torchtitan_additive_timing_noac_v2_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_additive_timing_noac_v2_ep8_s4096_gbs128_20260603.json), and [`run_artifacts/torchtitan_torchprof_decomposition_noac_v2_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_torchprof_decomposition_noac_v2_ep8_s4096_gbs128_20260603.json) rerun the same `8xMI350`, `S4096`, `GBS128`, CE2, MORI routed padded VJP profiler shape with `CANARY_ACTIVATION_CHECKPOINT_MODE=none`. The run completed cleanly and logged step 2 at about `4,042` tok/GPU/s, `76.49` TFLOPs/GPU, `3.31%` MFU, and loss `12.23414 -> 9.06199`; memory fit, but the rank peak range was high at `70.87-210.48 GiB`.

Clean no-activation-checkpoint performance pass: [`run_artifacts/torchtitan_dsv4_noac_perf_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_dsv4_noac_perf_ep8_s4096_gbs128_20260603.json) reruns the retained no-profiler `8xMI350`, `S4096`, `GBS128`, `FSDP8`, `EP8`, launcher-default CE2 shape with activation checkpointing disabled and profiling/structured/additive timing off. It completed three steps and destroyed process groups cleanly. Step 2 logged `4,492` tok/GPU/s, `85.00` TFLOPs/GPU, `3.68%` MFU, and loss `12.26291 -> 9.41511`; versus the retained full-AC CE2 anchor at `3,103` tok/GPU/s and `58.71` TFLOPs/GPU, that is about `+44.8%` TFLOPs uplift and a `21.12s -> 14.59s` implied step-time reduction. Caveat: step 3 slowed to `3,376` tok/GPU/s and `63.88` TFLOPs/GPU while peak logged rank memory rose to `249.36 GiB` with two allocation-retry warnings, so no-AC is a promising throughput lever but not yet a stable promoted baseline.

No-AC stability repeat: [`run_artifacts/torchtitan_dsv4_noac_stability4_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_dsv4_noac_stability4_ep8_s4096_gbs128_20260603.json) runs the same no-profiler/no-structured/no-additive `8xMI350`, `S4096`, `GBS128`, CE2 shape for four steps without allocator overrides. It confirms the step-2 burst at `4,523` tok/GPU/s and `85.58` TFLOPs/GPU, but then step 3 drops to `73.80` TFLOPs/GPU and step 4 drops to `56.20` TFLOPs/GPU while peak logged memory rises to `249.49 GiB` with two allocation-retry warnings. Mean steps 2-4 are `71.86` TFLOPs/GPU; mean steps 3-4 are `65.00` TFLOPs/GPU. Read: no-AC is a real latency lever but not currently a stable retained baseline; the `85` TFLOPs/GPU value is a burst diagnostic, and the next no-AC branch should target memory/lifetime or allocator behavior before another longer canary.

No-AC allocator-stabilized A/B: [`run_artifacts/torchtitan_dsv4_noac_alloc_expand_stability4_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_dsv4_noac_alloc_expand_stability4_ep8_s4096_gbs128_20260603.json) repeats the four-step no-AC shape with both allocator aliases set to `expandable_segments:True`. This materially improves the no-AC branch: step 2 reaches `4,832` tok/GPU/s and `91.44` TFLOPs/GPU with only `68.48 GiB` peak logged rank memory, and step 4 remains at `3,516` tok/GPU/s and `66.53` TFLOPs/GPU with `104.21 GiB` peak memory and no allocation-retry warnings. Versus default allocator no-AC, step 4 improves `56.20 -> 66.53` TFLOPs/GPU and peak memory drops `249.49 -> 104.21 GiB`. Versus the retained full-AC anchor, step 4 is `+13.3%` TFLOPs and mean steps 3-4 are `68.14` TFLOPs/GPU (`+15.9%`). Read: allocator-expanded no-AC is the first plausible retained no-AC branch, but it still needs a longer no-profile run because throughput decays from the `91.44` step-2 burst to the `66-70` TFLOPs/GPU late-step band.

No-AC allocator-expanded 6-step gate: [`run_artifacts/torchtitan_dsv4_noac_alloc_expand_stability6_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_dsv4_noac_alloc_expand_stability6_ep8_s4096_gbs128_20260603.json) reruns the same no-profiler, structured-logging-off `8xMI350`, `S4096`, `GBS128`, CE2, no-activation-checkpoint branch with both allocator aliases set to `expandable_segments:True`. It completes six steps and destroys process groups cleanly. The step-2 burst is real at `4,857` tok/GPU/s, `91.90` TFLOPs/GPU, and `13.49s` implied step time, but the late steps are `70.65`, `52.29`, `63.00`, and `60.67` TFLOPs/GPU for steps 3-6. Mean steps 2-6 are `67.70` TFLOPs/GPU (`+15.3%` over the retained full-AC anchor), but mean steps 4-6 are only `58.65` TFLOPs/GPU, essentially the retained full-AC anchor's `58.71` TFLOPs/GPU. Read: no-AC plus expandable allocator is useful evidence for memory/free behavior and the checkpoint-recompute wall, but this exact branch should not replace the retained full-AC baseline until late-step throughput is stable above the anchor.

The no-AC additive split answers the immediate backward/wrapper question. In the comparable step-2 additive waterfall, full-AC call-boundary attribution had forward `2.302 s`, loss `0.470 s`, backward `13.654 s`, and fwd/bwd wrapper residual `4.605 s`. No-AC changes that to forward `2.570 s`, loss `0.469 s`, backward `6.750 s`, and wrapper residual `4.575 s`. So disabling activation checkpointing removes about `6.9 s` from the named backward phase, but it does not materially change the `4.5-4.6 s` fwd/bwd wrapper wall. The corrected additive parser also confirms this wrapper is not hidden phase pre-sync: step-2 no-AC inner pre-sync is only `2.40 ms` max-rank across all microbatches.

Detached-loss return-boundary negative control: [`run_artifacts/torchtitan_return_detached_loss_probe_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_return_detached_loss_probe_ep8_s4096_gbs128_20260603.json) adds `TORCHTITAN_DSV4_RETURN_DETACHED_LOSS=1` while keeping the same full-AC `8xMI350`, `S4096`, `GBS128`, CE2, MORI routed padded VJP diagnostic shape. The run completes two steps and logs step 2 at `2,817` tok/GPU/s, `53.30` TFLOPs/GPU, and loss `12.29612 -> 9.69264`, so it does not produce a throughput uplift. More importantly, `forward_backward.return_loss` remains sub-ms (`0.896 ms` max-rank), while the same-rank `forward_backward_call` residual remains about `4.916 s` (`p50` per-microbatch residual `306.2 ms`). Read: the live returned loss object is not the wrapper wall; the separate return/unwind free-path branch remains FSDP/autograd/checkpoint lifetime or allocator/runtime synchronization rather than ordinary loss-return bookkeeping.

Drop-original-loss lifetime localization: [`run_artifacts/torchtitan_drop_original_loss_probe_ep8_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_drop_original_loss_probe_ep8_s4096_gbs128_20260603.json) adds `TORCHTITAN_DSV4_DROP_ORIGINAL_LOSS_BEFORE_RETURN=1` on top of the detached-loss probe and explicitly deletes the original graph-bearing `loss` inside a named `forward_backward.drop_original_loss` phase. The same full-AC `8xMI350`, `S4096`, `GBS128`, CE2, MORI routed padded VJP shape completes two steps, with step 2 at `2,852` tok/GPU/s, `53.97` TFLOPs/GPU, and loss `12.15477 -> 9.20598`. The diagnostic read is clean: `drop_original_loss` is `4.823 s` mean / `4.886 s` max-rank, while the remaining `forward_backward_call` residual collapses to `4.832 ms` mean / `5.390 ms` max-rank. So the old `4.6-4.9 s` wrapper wall is the graph-bearing loss lifetime release, not scalar return bookkeeping. This names the wall but does not remove it; the production lever is reducing or avoiding that FSDP/autograd/checkpoint graph-free path.

Torch-profiler attribution gives the mechanism split. Full AC has `MmBackward0` at `~9.004 s/rank` autograd eval, `_MoriAiterForwardBackwardBackward` at `~4.349 s/rank`, and `hipFree` at `~9.425 s/rank`, split into `~4.732 s` inside `MmBackward0` and `~4.692 s` unattributed. No-AC drops `MmBackward0` to `~0.102 s/rank` and removes the `MmBackward0` `hipFree` half, but the unattributed `hipFree` remains `~4.664 s/rank`. This makes the ladder sharper: checkpointed dense/recompute/autograd lifetime is one backward branch, MORI routed VJP/return is another backward branch, and the fwd/bwd wrapper is a separate return/unwind free-path branch.

Current bottleneck ladder from the additive + Torch-profiler bridge: (1) fwd/bwd wrapper hidden free/unwind, about `4.6 s`, matched by unattributed `hipFree` and unchanged by no-AC or `expandable_segments`; (2) activation-checkpointed `MmBackward0`, about `9.0 s` under full AC and nearly gone under no-AC, with about `4.7 s` of synchronizing `hipFree` inside it; (3) MORI routed backward skew/return, about `3.4-5.4 s` in Torch-profiler eval depending on run skew and still the main named MoE branch; (4) loss/lm-head and attention are secondary in this guarded canary trace. Dense/sparse attention final-path work still matters for production, but it is not the lead wall in this current EP8 guarded systems run.

## Current Targets

Ascend-inspired parity targets for the single full DSv4 layer are based on `47.3k` layer-tokens/s with BF16 input/output and full forward + loss + backward:

| Shape | Target ms | Current best ms | Current tokens/s | Current TFLOPs | Gap |
| --- | ---: | ---: | ---: | ---: | --- |
| `B=1 S=1024` | `21.6` | not current | not current | not current | trend/debug only |
| `B=1 S=2048` | `43.3` | `47.0589` | `43519.91` | `206.892852` | `1.09x` time gap |
| `B=1 S=4096` | `86.6` | `68.2089` | `60050.82` | `310.415755` | clears target, `0.79x` target time |

## 2026-05-31 Retained-Path Resample

Command family:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen <2048-or-4096> --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir <profile-dir>
```

Results from `do-sonle-kernel` in container `b9d33e6e8227`:

| Shape | Total ms | Target ms | Tokens/s | TFLOPs | Forward ms | Backward ms | Forward attention ms | Forward MoE ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B=1 S=2048` | `47.2087` | `43.3` | `43,381.80` | `206.2363` | `15.0589` | `32.0575` | `4.9593` | `9.3408` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_resample_20260531/summary.json` |
| `B=1 S=4096` | `68.4030` | `86.6` | `59,880.42` | `309.5349` | `22.5905` | `45.6676` | `10.8870` | `10.5828` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s4096_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_resample_20260531/summary.json` |

S2048 segmented backward attribution used the same retained path with `--profile-backward-blocks --profile-attention-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1`.

| Block | ms |
| --- | ---: |
| `moe.routed.dgrad` | `23.2755` |
| `moe.routed.wgrad` | `23.0698` |
| `attention.wgrad` | `7.1151` |
| `attention.dgrad` | `6.8804` |
| `moe.routed.w1_w3.dgrad` | `6.0829` |
| `moe.routed.w2.wgrad` | `4.9636` |
| `moe.routed.w1.wgrad` | `4.8052` |
| `moe.routed.w3.wgrad` | `4.4015` |
| `attention.sparse_mla.sink_wgrad` | `3.1856` |
| `attention.sparse_mla.dgrad_qkv` | `3.1672` |

Read: the resample supports treating the current fixed-router packed-sim layer as usable for system-integration canaries, but not as the final DSv4 training kernel. The next single-layer performance work should target real packed-FP4 routed MoE training GEMM/backward first for S2048, while preserving the BF16-contract sparse MLA path that already clears the S4096 target in this harness.

## 2026-05-31 Learned-Router Smoke

The retained ceiling path above intentionally uses `--fixed-router`, so router gradients are bypassed and routed expert assignments are uniform/cyclic. To bound the current gap to learned routing, two `B=1 S=1024` full-dim smokes were run with the real router enabled.

Generic selected-weight grouped path:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-token-sum --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_learned_router_grouped_token_sum_bf16_contract_saved_accum_1iter_20260531
```

Result: failed before timing with `torch.OutOfMemoryError` while selecting routed expert weights. The failed allocation was `252.00 GiB` for the selected W1/W3-style expert table at `S=1024`, with about `233.66 GiB` free. This confirms the generic assignment-batched path is not viable for full-dim learned routing because it materializes selected weights as `[tokens * top_k, expert_dim, hidden]`.

Loop reference path:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl loop --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_learned_router_loop_bf16_contract_saved_accum_warm1_1iter_20260531
```

Result: completed with real router gradients, but only as a slow correctness/reference surface: total `12,847.0178 ms`, forward `94.9729 ms`, backward `12,751.9660 ms`, `79.71 tokens/s`, and `0.3624 TFLOPs`. Forward routing/MoE dominated the measured forward pass: `moe.routed_experts 89.6744 ms`, `moe.router 0.8283 ms`, while attention forward was only `3.3814 ms`. Artifact: `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s1024_fp4_packed_sim_learned_router_loop_bf16_contract_saved_accum_warm1_1iter_20260531/summary.json`.

Read: fixed-router packed-sim remains the right single-layer math/perf ceiling and 8x system-canary baseline, but learned-router single-layer training is not solved. The next aligned implementation surface is dynamic expert batching/dispatch that keeps full expert weights resident and gathers tokens to experts, then launches grouped packed-FP4/BF16 training GEMMs without selected-weight materialization and with dense checkpointable router/expert gradients.

## 2026-05-31 Dynamic Learned-Router Expert Batches

Added `grouped-expert-batches-dynamic-dense-wgrad`, a PyTorch packed-sim bridge for non-uniform learned routing. It sorts token assignments by expert, pads each expert to the maximum touched-token count for grouped `torch.bmm`, keeps full expert weights resident, returns dense expert wgrads, and propagates gradients through `top_scores` back to `moe.gate`.

Correctness command on `do-sonle-kernel` container `b9d33e6e8227`:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_dense_wgrad_match,moe_expert_batches_dynamic_learned_router_match --json
```

Result: passed. The learned-router check compared against the `loop` oracle with no `router_override`, watched `moe.gate` plus expert/shared gradients, and reported zero output error with maximum gradient relative L2 `1.05e-16`.

Full-dim learned-router smoke command family:

```bash
python3 dsv4_layer_microbench/run_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dynamic-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen <1024|2048|4096> --dtype bfloat16 --device cuda --warmup 1 --iters 1 --rmsnorm-save-policy saved-accum --json
```

| Shape | Total ms | Target ms | Tokens/s | TFLOPs | `moe.gate` grad norm | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `B=1 S=1024` | `44.3061` | `21.6` | `23,111.92` | `105.0752` | `2.4348e-4` | no OOM, real router grad |
| `B=1 S=2048` | `54.5535` | `43.3` | `37,541.10` | `178.4697` | `1.7442e-4` | no OOM, real router grad |
| `B=1 S=4096` | `78.4155` | `86.6` | `52,234.60` | `270.0120` | `1.2332e-4` | clears primary 4K target with learned router |

Route skew with seed `1234`:

| Shape | Assignments | Touched experts | Mean per expert | Max per expert | Padded rows | Padding multiplier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S=1024` | `6,144` | `384` | `16.0` | `29` | `11,136` | `1.8125x` |
| `S=2048` | `12,288` | `384` | `32.0` | `52` | `19,968` | `1.6250x` |
| `S=4096` | `24,576` | `384` | `64.0` | `95` | `36,480` | `1.4844x` |

S2048 forward block timing with `--profile-blocks --warmup 1 --iters 2`: total `55.3648 ms`, forward `moe.routed_experts 10.8267 ms`, `moe.router 0.8723 ms`, `attn.sparse_mla 2.3778 ms`, `attn.q_proj 1.0194 ms`.

S2048 segmented backward attribution with `--profile-backward-blocks --profile-attention-blocks --backward-attribution-iters 1`: normal phase total `56.4729 ms`, forward `18.8385 ms`, backward `37.5471 ms`; routed MoE dominates segmented attribution with `moe.routed.dgrad 28.8211 ms` and `moe.routed.wgrad 28.5577 ms`, while attention is `6.7808 ms` dgrad and `7.0120 ms` wgrad. Artifact: `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s2048_dynamic_learned_router_backward_attribution_20260531/summary.json`.

Updated the dynamic backward xgrad path to accumulate the W3 contribution into the W1 `grad_x_group` buffer with `torch.baddbmm(..., out=grad_x_group)`, avoiding one routed-xgrad temporary. Correctness after the change:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_learned_router_match --json
```

Result: passed, with zero output error, zero `moe.gate` gradient error, and maximum gradient relative L2 `1.08e-16`.

Post-change full-dim learned-router timing:

| Shape | Total ms | Target ms | Tokens/s | TFLOPs | `moe.gate` grad norm | Command |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `B=1 S=2048` | `53.4689` | `43.3` | `38,302.66` | `182.0901` | `1.7442e-4` | `--warmup 2 --iters 5` |
| `B=1 S=4096` | `77.4362` | `86.6` | `52,895.12` | `273.4264` | `1.2332e-4` | `--warmup 2 --iters 3` |

Contiguous expert-chunk variants were added as negative controls. They keep resident weights and use contiguous `narrow()` slices, so they avoid selected-weight materialization while reducing padding. The theoretical padded-row multipliers are:

| Shape | Global padded | Chunk128 padded | Chunk32 padded |
| --- | ---: | ---: | ---: |
| `S=2048` | `1.6250x` | `1.5313x` | `1.3776x` |
| `S=4096` | `1.4844x` | `1.4219x` | `1.2839x` |

Correctness gates passed for `grouped-expert-batches-dynamic-chunk32-dense-wgrad` and `grouped-expert-batches-dynamic-chunk128-dense-wgrad`, including the learned-router `moe.gate` gradient check. Performance regressed despite the lower padded row count:

| Variant | Shape | Total ms | Tokens/s | TFLOPs | Command |
| --- | --- | ---: | ---: | ---: | --- |
| `dynamic-chunk32-dense-wgrad` | `S=2048` | `81.7969` | `25,037.63` | `119.0284` | `--warmup 2 --iters 5` |
| `dynamic-chunk32-dense-wgrad` | `S=4096` | `107.2938` | `38,175.54` | `197.3377` | `--warmup 2 --iters 3` |
| `dynamic-chunk128-dense-wgrad` | `S=2048` | `81.9203` | `24,999.92` | `118.8491` | `--warmup 2 --iters 5` |

Read: learned-router training is now usable in the full-dim PyTorch packed-sim layer and the small `baddbmm` xgrad edit gives a modest win, but S2048 remains above the development target. The route-skew data says the padded bridge is doing `1.6x` ideal routed rows at S2048, and backward attribution confirms routed MoE dgrad/wgrad is the limiter. However, PyTorch-side chunking is the wrong fix: the smaller grouped-BMM shapes and launch fragmentation overwhelm the reduced padding. Keep `grouped-expert-batches-dynamic-dense-wgrad` as the retained learned-router bridge. The next performance step is replacing padded PyTorch `bmm` with real dynamic grouped GEMM/dispatch that consumes compact per-expert token counts in one or a few efficient kernels.

## 2026-05-31 AITER Dynamic Grouped-GEMM Probe

Added `probe_aiter_gmm_dynamic_moe.py` to isolate AITER `gmm`, `ptgmm`, and `nptgmm` on compact non-uniform learned-router-like expert groups. This is not an integrated MoE path yet: it excludes token dispatch/return, SiLU/gating, score weighting, dense expert-gradient scatter, and packed-FP4 quantization. It tests whether AITER can consume per-expert `group_sizes` directly enough to replace the padded PyTorch `bmm` bridge.

Small correctness command:

```bash
python3 dsv4_layer_microbench/probe_aiter_gmm_dynamic_moe.py --tokens 16 --hidden 32 --intermediate 24 --experts 8 --topk 2 --warmup 1 --iters 2 --check --time-torch --json
```

Result: passed. All eight compact non-uniform contractions matched per-expert PyTorch references with zero observed error: `gmm_fwd_w1`, `gmm_fwd_w2`, `gmm_dgrad_w2`, `gmm_xgrad_w1w3_pair`, `ptgmm_wgrad_w2`, `nptgmm_wgrad_w2`, `ptgmm_wgrad_w1`, and `nptgmm_wgrad_w1`. The test included one zero-token expert group.

Full-shape timing command family:

```bash
python3 dsv4_layer_microbench/probe_aiter_gmm_dynamic_moe.py --tokens <2048-or-4096> --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 1 --iters <3-or-2> --json --summary-path profiles/<run>/summary.json
```

| Shape | Route padding if padded to max | AITER W2 dgrad GMM | AITER W1/W3 xgrad pair | AITER W2 wgrad PTGMM | AITER W1 wgrad PTGMM | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `S=2048` | `1.4688x` | `6.8231 ms`, `79.31 TFLOPs` | `13.8718 ms`, `78.02 TFLOPs` | `7.2412 ms`, `74.73 TFLOPs` | `7.2124 ms`, `75.03 TFLOPs` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/aiter_dynamic_gmm_s2048_20260531/summary.json` |
| `S=4096` | `1.3750x` | `6.9268 ms`, `156.25 TFLOPs` | `14.2125 ms`, `152.31 TFLOPs` | `8.4410 ms`, `128.22 TFLOPs` | `8.1392 ms`, `132.98 TFLOPs` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/aiter_dynamic_gmm_s4096_20260531/summary.json` |

For `S=2048`, the AITER dgrad-shaped work is approximately `gmm_dgrad_w2 + gmm_xgrad_w1w3_pair = 20.69 ms`, compared with the current segmented PyTorch dynamic bridge's `moe.routed.dgrad 28.82 ms`. The AITER wgrad-shaped work using persistent TGMM is approximately `ptgmm_wgrad_w2 + 2 * ptgmm_wgrad_w1 = 21.67 ms`, compared with current `moe.routed.wgrad 28.56 ms`. This is a useful candidate for backward integration. Forward is less obviously favorable: separate compact GMMs for W1, W3, and W2 would be about `21.44 ms` at `S=2048`, while the current measured dynamic routed-expert forward block is `10.83 ms`; integration should therefore start with backward contractions or combine forward kernels before replacing the existing forward bridge.

Follow-up integration probe: added `grouped-expert-batches-dynamic-aiter-bwd-dense-wgrad`, which keeps the current padded PyTorch forward bridge but uses compact AITER `gmm`/`ptgmm` in the dynamic backward path when CUDA BF16/FP16 shapes are compatible. A small `preset=small`, `S=16`, BF16 MoE-only comparison against the loop oracle passed with matching loss, output relative L2 `2.87e-3`, and watched gradient relative L2 below `7.65e-3`. Full-dim timing was negative:

| Variant | Shape | Total ms | Tokens/s | TFLOPs | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `dynamic-aiter-bwd-dense-wgrad` | `S=2048` | `65.2770` | `31,374.00` | `149.1514` | slower than retained dynamic bridge at `53.4689 ms` |
| `dynamic-aiter-bwd-dense-wgrad` | `S=4096` | `90.1813` | `45,419.64` | `234.7840` | slower than retained dynamic bridge at `77.4362 ms`, and misses `86.6 ms` target |

Read: the isolated AITER contraction kernels are correct and fast enough to keep as a design input, but this first autograd-level integration is not the right replacement. Compact materialization, extra index-select/scatter work, and separate AITER calls erase the raw GEMM/TGMM savings. Keep `grouped-expert-batches-dynamic-dense-wgrad` as the retained learned-router bridge; the next real implementation needs a fused compact dispatch plus grouped GEMM/backward path, not a Python-side compact wrapper around the existing padded forward.

Current-code attribution rerun with identical `S=2048`, `--profile-backward-blocks --profile-attention-blocks --profile-moe-blocks`, and `--backward-attribution-iters 1` confirms the regression is inside routed backward:

| Impl | Phase total | Forward | Backward | Routed dgrad | Routed wgrad | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dynamic-dense-wgrad` | `55.9147 ms` | `18.5475 ms` | `37.2755 ms` | `28.4840 ms` | `28.1315 ms` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s2048_dynamic_retained_backward_attribution_current_20260531/summary.json` |
| `dynamic-aiter-bwd-dense-wgrad` | `68.4740 ms` | `18.8345 ms` | `49.5446 ms` | `40.6271 ms` | `39.6812 ms` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s2048_dynamic_aiter_bwd_backward_attribution_20260531/summary.json` |

Read: AITER contraction-only timings are not predictive once the current autograd wrapper has to compact saved padded activations, run multiple independent grouped GEMMs, materialize compact gradients, then scatter/reduce back to token order. The next fused kernel contract should consume the sorted assignment metadata directly and fuse at least compact activation gather, W2 dgrad, W1/W3 xgrad accumulation, W1/W2/W3 TGMM wgrad, and token-gradient index-add/reduce around one resident expert-weight layout.

Added `probe_dynamic_moe_bridge_ops.py` to time the retained dynamic PyTorch bridge's internal operations outside the full layer. This uses the same learned-router-style non-uniform route, pads to per-shape max expert count, and times route build, padded gather/mask, routed BMMs, score-gradient restore, output reduce, and input reduce. It does not include attention, RMSNorm, shared expert, or full autograd scheduling.

Command family:

```bash
python3 dsv4_layer_microbench/probe_dynamic_moe_bridge_ops.py --tokens <2048-or-4096> --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --dtype bfloat16 --warmup 2 --iters <5-or-3> --json --summary-path profiles/<run>/summary.json
```

| Shape | Padding | Route build | X gather/mask | Output scale/reduce | W2 dgrad | W2 dgrad `out=` | W1 wgrad | W2 wgrad | W3 wgrad | W1/W3 xgrad | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S=2048` | `1.6250x` | `0.3151 ms` | `0.2134 ms` | `0.8742 ms` | `3.6303 ms` | `3.4936 ms` | `5.3106 ms` | `5.4456 ms` | `5.3730 ms` | `6.8232 ms` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/dynamic_moe_bridge_ops_s2048_20260531/summary.json` |
| `S=4096` | `1.2813x` | `0.3434 ms` | `0.3332 ms` | `1.4185 ms` | `4.1241 ms` | `3.9336 ms` | `6.0346 ms` | `6.1750 ms` | `5.9335 ms` | `7.4711 ms` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/dynamic_moe_bridge_ops_s4096_20260531/summary.json` |

Read: the retained bridge's Python route bookkeeping and tensor movement are measurable but not the main gap: route build plus gather plus output reduce are roughly `1.4 ms` at S2048 and `2.1 ms` at S4096. The wall is still the padded routed GEMM family, especially the three wgrad BMMs and W1/W3 xgrad pair. A local `out=` tweak for W2 dgrad saves only about `0.14-0.19 ms` in isolation, so it is not worth promoting without a full-layer win. The next meaningful step is still a fused dynamic grouped-MoE training kernel that avoids max-padding and collapses dispatch/reduce with the grouped GEMM work.

Added selectable negative-control variant `grouped-expert-batches-dynamic-bmm-out-dgrad-dense-wgrad` to test that small `out=` W2-dgrad signal in the full layer without changing the retained path. Correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_bmm_out_dgrad_dense_wgrad_match,moe_expert_batches_dynamic_bmm_out_dgrad_learned_router_match --json
```

Result: passed. The learned-router check matched the loop oracle exactly at output and `moe.gate` gradient, with maximum watched-gradient relative L2 `1.05e-16`.

Full-layer adjacent timing:

| Impl | Shape | Total ms | Tokens/s | TFLOPs | Read |
| --- | --- | ---: | ---: | ---: | --- |
| `dynamic-bmm-out-dgrad-dense-wgrad` | `S=2048` | `53.1505` | `38,532.08` | `183.1808` | adjacent retained was `53.2086 ms` |
| `dynamic-dense-wgrad` | `S=2048` | `53.2086` | `38,489.98` | `182.9807` | retained baseline |
| `dynamic-bmm-out-dgrad-dense-wgrad` | `S=4096` | `77.1339` | `53,102.48` | `274.4983` | adjacent retained was `77.2244 ms` |
| `dynamic-dense-wgrad` | `S=4096` | `77.2244` | `53,040.21` | `274.1764` | retained baseline |

Read: the variant is correctness-safe and directionally positive, but the full-layer gain is only `0.06-0.09 ms`, too small to treat as a new retained path. Keep it selectable as a micro-probe and keep `grouped-expert-batches-dynamic-dense-wgrad` as the retained learned-router bridge unless a broader grouped-GEMM/backward change makes this allocation pattern matter.

Added two additional dynamic learned-router bridge probes:

- `grouped-expert-batches-dynamic-no-mask-dense-wgrad`: skips the padded-row `valid_mask` elementwise passes in forward/output reduce/backward input reduce. The invalid padded rows already carry zero `score_padded`, and backward `grad_y_unscaled` is therefore zero, so the masks are redundant for finite inputs.
- `grouped-expert-batches-dynamic-no-mask-bmm-out-dgrad-dense-wgrad`: combines the no-mask contract with the previous W2 dgrad `torch.bmm(..., out=...)` micro-probe.

Correctness commands:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_no_mask_dense_wgrad_match,moe_expert_batches_dynamic_no_mask_learned_router_match --json
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_no_mask_bmm_out_dgrad_learned_router_match --json
```

Result: passed. The no-mask and combined learned-router checks matched the loop oracle exactly at output and `moe.gate` gradient. The combined check's maximum watched-gradient relative L2 was `1.08e-16`.

Full-layer adjacent timing with the same command family as above:

| Impl | Shape | Total ms | Min ms | Tokens/s | TFLOPs | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `dynamic-dense-wgrad` | `S=2048` | `53.4000` | `53.2439` | `38,352.04` | `182.3249` | adjacent retained baseline |
| `dynamic-no-mask-dense-wgrad` | `S=2048` | `52.9487` | `52.7896` | `38,678.95` | `183.8790` | best S2048 of this probe |
| `dynamic-no-mask-bmm-out-dgrad-dense-wgrad` | `S=2048` | `53.0539` | `52.8943` | `38,602.26` | `183.5144` | improves retained, but trails no-mask-only |
| `dynamic-dense-wgrad` | `S=4096` | `76.9556` | `76.9273` | `53,225.49` | `275.1341` | adjacent retained baseline |
| `dynamic-no-mask-dense-wgrad` | `S=4096` | `77.0486` | `76.3967` | `53,161.25` | `274.8021` | neutral/slightly negative at 4K |
| `dynamic-no-mask-bmm-out-dgrad-dense-wgrad` | `S=4096` | `76.8752` | `76.5971` | `53,281.18` | `275.4220` | best 4K of this probe, but only `0.08 ms` ahead |

Read: the redundant mask removal is correctness-safe and gives a useful development-shape signal, but it does not change the overall diagnosis. The combined no-mask plus W2-dgrad `out=` path is the only one that is positive on both S2048 and S4096 in this adjacent sample, yet the 4K gain is only about `0.1%`. Keep both selectable; use `grouped-expert-batches-dynamic-no-mask-bmm-out-dgrad-dense-wgrad` as the provisional fastest learned-router bridge for 4K smoke runs, while continuing to prioritize a fused compact grouped-MoE training kernel for material progress.

Added `grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad`, which keeps the no-mask plus W2-dgrad `out=` contract and also compacts the forward output `index_add_` and backward input-gradient `index_add_` down to valid routed assignments. The BMM shapes are unchanged; this only avoids reducing padded zero rows.

Correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_compact_reduce_bmm_out_dgrad_dense_wgrad_match,moe_expert_batches_dynamic_compact_reduce_bmm_out_dgrad_learned_router_match --json
```

Result: passed. The learned-router check matched the loop oracle exactly at output and `moe.gate` gradient, with maximum watched-gradient relative L2 `1.05e-16`.

Adjacent timing:

| Impl | Shape | Total ms | Min ms | Tokens/s | TFLOPs | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `dynamic-no-mask-bmm-out-dgrad-dense-wgrad` | `S=2048` | `53.0390` | `52.8509` | `38,613.11` | `183.5660` | adjacent previous provisional |
| `dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad` | `S=2048` | `52.8658` | `52.5156` | `38,739.57` | `184.1672` | new best learned-router bridge in this harness |
| `dynamic-no-mask-bmm-out-dgrad-dense-wgrad` | `S=4096` | `76.7516` | `76.4385` | `53,366.97` | `275.8655` | adjacent previous provisional |
| `dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad` | `S=4096` | `76.7364` | `76.1159` | `53,377.56` | `275.9202` | tiny 4K win |

Read: compacting reductions is worthwhile enough to keep as the provisional fastest learned-router PyTorch bridge: it is positive on both target shapes and moves S2048 to `38.74k` layer-tokens/s. The 4K improvement is tiny, so this does not change the main bottleneck conclusion: remaining material progress still requires replacing the padded routed GEMM family with a fused compact grouped-MoE training kernel.

Refined the same `dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad` path so the compact forward output reduction multiplies only valid `y_unscaled` rows by sorted router scores, and the router-score gradient computes the `grad_output * y_unscaled` row reductions only for valid routed assignments. This keeps the padded BMM contract unchanged but removes another padded elementwise/reduction pass. CPU and CUDA learned-router correctness both passed against the loop oracle after the refinement:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_compact_reduce_bmm_out_dgrad_learned_router_match --json
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_compact_reduce_bmm_out_dgrad_dense_wgrad_match,moe_expert_batches_dynamic_compact_reduce_bmm_out_dgrad_learned_router_match --json
```

| Shape | Warmup/iters | Total ms | Min ms | Tokens/s | TFLOPs | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `S=2048` | `3/7` | `52.5944` | `52.1409` | `38,939.51` | `185.1177` | best learned-router bridge sample so far |
| `S=4096` | `3/5` | `76.3632` | `76.0684` | `53,638.37` | `277.2684` | best learned-router bridge sample so far |

Read: the valid-row score/output specialization is a larger win than the original compact-reduction edit, especially at 4K. This is still a PyTorch bridge with padded routed GEMMs, but it is now the retained/provisional learned-router bridge for single-layer smoke runs while fused compact grouped-MoE training kernels are developed.

Added `grouped-expert-batches-dynamic-compact-scatter-bmm-out-dgrad-dense-wgrad` as a negative-control probe. It keeps the compact-reduce forward/output and W2-dgrad `out=` contracts, but in backward builds the padded `grad_y_unscaled` BMM input by scaling only valid routed rows and scattering them into a zero padded slab. The intent was to test whether valid-row gather plus scatter beats the full padded `grad_output_group * score_padded` multiply.

Correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_compact_scatter_bmm_out_dgrad_learned_router_match --json
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dynamic_compact_scatter_bmm_out_dgrad_dense_wgrad_match,moe_expert_batches_dynamic_compact_scatter_bmm_out_dgrad_learned_router_match --json
```

Result: passed. CPU learned-router, CUDA dense-router, and CUDA learned-router checks matched the loop oracle with maximum watched-gradient relative L2 at or below `1.05e-16`.

Adjacent timing with the same current code snapshot:

| Impl | Shape | Warmup/iters | Total ms | Min ms | Tokens/s | TFLOPs | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dynamic-compact-scatter-bmm-out-dgrad-dense-wgrad` | `S=2048` | `3/7` | `52.9529` | `52.6301` | `38,675.86` | `183.8643` | slower than adjacent compact-reduce |
| `dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad` | `S=2048` | `3/7` | `52.8996` | `52.4100` | `38,714.86` | `184.0497` | retained |
| `dynamic-compact-scatter-bmm-out-dgrad-dense-wgrad` | `S=4096` | `3/5` | `76.8241` | `76.2872` | `53,316.57` | `275.6050` | slower than adjacent compact-reduce |
| `dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad` | `S=4096` | `3/5` | `76.4850` | `76.0246` | `53,552.97` | `276.8270` | retained |

Read: scatter-building the padded backward BMM input adds more memory traffic than it removes. Keep the variant selectable as a correctness-safe negative control, but keep `grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad` as the retained learned-router PyTorch bridge. The next useful MoE step remains a fused compact grouped-MoE training kernel that consumes sorted assignment metadata directly instead of materializing padded PyTorch BMM inputs.

A follow-up S2048 profile of the retained compact-reduce path confirmed the normal phase shape but exposed a profiler caveat. With `--profile-backward-blocks --profile-attention-blocks`, the measured normal phase was total `54.4142 ms`, forward `17.3860 ms`, backward `36.9342 ms`, with forward attention `4.9630 ms` and forward MoE `11.6202 ms`. However, the segmented `autograd.grad` attribution path reported routed MoE dgrad `158.1072 ms` and routed MoE wgrad `12,849.8748 ms`, far above the normal backward phase. Treat these routed dgrad/wgrad segmented numbers as a profiler/harness artifact for the compact dynamic autograd path, not as kernel evidence. Artifact: `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s2048_dynamic_compact_reduce_backward_attribution_nomoe_20260531/summary.json`.

Updated `profile_microbench.py` so `--profile-moe-blocks` can rebuild the dynamic learned-router expert-batch graph for the retained compact-reduce bridge and split the routed expert path into score-gradient restore, output reduce, W2 dgrad, SwiGLU dgrad, W1/W3 xgrad, input reduce, and the three routed wgrads. A CPU smoke inside the ROCm container passed cleanly with `preset=tiny`, dynamic compact-reduce, `--profile-backward-blocks --profile-moe-blocks`, and `attention_impl=dense-mask`.

Real S2048 MI350 attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --rmsnorm-save-policy saved-accum --trace-dir profiles/full_b1s2048_dynamic_compact_reduce_moe_subblock_attr_20260531 --no-torch-trace --profile-backward-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1
```

Normal phase from that run: total `59.4128 ms`, forward `22.3186 ms`, backward `36.9907 ms`. Dynamic routed MoE subblock timings:

| Block | ms |
| --- | ---: |
| `moe.routed.output_scale_reduce.dgrad` | `0.7323` |
| `moe.routed.score_grad` | `0.4021` |
| `moe.routed.w2.dgrad` | `3.1178` |
| `moe.routed.swiglu.dgrad` | `0.4096` |
| `moe.routed.w1_w3.dgrad` | `6.5315` |
| `moe.routed.input_gather_reduce.dgrad` | `0.6234` |
| `moe.routed.w1.wgrad` | `6.3312` |
| `moe.routed.w2.wgrad` | `6.5082` |
| `moe.routed.w3.wgrad` | `5.6959` |

Read: this makes the dynamic bridge attribution line up with the isolated op probe: routed expert GEMM-family work, not route bookkeeping, is the wall. The subblock reconstruction detaches `top_scores`, so its routed input-gradient comparison intentionally excludes the router-score-to-input path; the observed `moe_subblock_input_grad_rel_l2_diff=0.0729` is not a kernel mismatch. The old whole-routed segmented buckets are still not useful for compact dynamic learned-router attribution: in this same run they reported `moe.routed.dgrad=162.6882 ms` and `moe.routed.wgrad=12,724.1371 ms`, far beyond the normal `36.9907 ms` backward phase. Use the subblock rows above and the isolated op probe as the actionable breakdown. Artifact: `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_b1s2048_dynamic_compact_reduce_moe_subblock_attr_20260531/summary.json`.

Updated `probe_dynamic_moe_bridge_ops.py` to directly model the retained compact dynamic bridge instead of only the older dense masked path. The probe now accepts `--variant dense|no-mask|compact-reduce|compact-scatter`, defaults to `compact-reduce`, includes `--check` to compare the variant's output reduction, `grad_y_unscaled` build, router-score gradient, and input-gradient reduction against the dense masked reference, supports `--check-only` for no-timing correctness smokes, and separately times `bwd_input_reduce_only` versus `bwd_input_reduce_including_xgrad`. Timed runs also emit `derived_median_sum_ms` groups for isolated forward GEMMs, forward glue, backward dgrad path using W2 `bmm(out=)`, routed wgrad GEMMs, router-score restore, and total isolated train-path work. These sums are attribution aids, not end-to-end fused latency predictions. Full S2048/S4096 timing on canonical `do-sonle-kernel` is still pending a clean GPU window because CK flatmm and TileLang attention/value-tail probes were active on the node during the update.

Command family:

```bash
python3 dsv4_layer_microbench/probe_dynamic_moe_bridge_ops.py --tokens <2048-or-4096> --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --dtype bfloat16 --variant compact-reduce --check --check-atol 1.0 --check-rtol 0.025 --warmup 2 --iters <5-or-3> --json --summary-path profiles/dynamic_moe_bridge_ops_compact_reduce_s<2048-or-4096>_20260531/summary.json
```

The probe also supports `--device cpu` for small reference checks while the MI350 is occupied, plus `--check-atol/--check-rtol` for full BF16 reductions where compact valid-row `index_add_` is numerically equivalent but not bitwise identical to dense padded `index_add_`. Remote container `--check-only` CPU checks passed exactly at `tokens=8 hidden=16 intermediate=8 experts=4 topk=2 dtype=bfloat16` for all four variants: `dense`, `no-mask`, `compact-reduce`, and `compact-scatter`. A small CUDA compact-reduce check passed exactly on fallback GPU0 of `do-sonle5-mi350-gpu` with the same container image. This is not performance evidence; it only validates the probe's variant bookkeeping and reference equivalence.

Because canonical `do-sonle-kernel` remained busy, fallback single-GPU timings were taken on idle GPU0 of `do-sonle5-mi350-gpu` with image `rocm/sgl-dev:rocm720-mi35x-8c3b5aa-20260521-DSv4`, PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015`, and staged path `/local/data/sonle5/dsv4_layer_microbench_fallback_20260531`. Treat these as MI350 evidence, not the canonical single-GPU-node record. Full-shape exact checks fail only for the two BF16 reduction-order comparisons: at S2048, output reduce `max_abs=0.5 rel_l2=0.00420` and input reduce `max_abs=0.75 rel_l2=0.00419`; grad-y build and router-score gradient are exact. The tolerance run uses `--check-atol 1.0 --check-rtol 0.025` and reports `check_ok=true`.

Retained compact-reduce isolated op sums on fallback GPU0:

| Shape | Padding multiplier | Forward total ms | Backward dgrad path ms | Backward wgrad GEMM ms | Router-score ms | Backward total ms | Train isolated ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S=2048` | `1.6250` | `10.8565` | `10.7305` | `15.8446` | `0.2768` | `26.8519` | `37.7084` | `/local/data/sonle5/dsv4_layer_microbench_fallback_20260531/profiles/dynamic_moe_bridge_ops_compact_reduce_s2048_fallback_8xnode_20260531/summary.json` |
| `S=4096` | `1.2813` | `14.2542` | `12.2021` | `17.7657` | `0.5360` | `30.5038` | `44.7580` | `/local/data/sonle5/dsv4_layer_microbench_fallback_20260531/profiles/dynamic_moe_bridge_ops_compact_reduce_s4096_fallback_8xnode_20260531/summary.json` |

Representative median sub-ops:

| Shape | Fwd W1/W3/W2 BMM ms | W2 dgrad `bmm(out=)` ms | Xgrad W1+W3 `baddbmm` ms | Input reduce-only ms | W2/W1/W3 wgrad BMM ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S=2048` | `3.495 / 3.327 / 3.303` | `3.405` | `6.564` | `0.384` | `5.431 / 5.208 / 5.206` |
| `S=4096` | `4.346 / 4.324 / 4.360` | `3.745` | `7.153` | `0.756` | `6.085 / 5.836 / 5.845` |

Read: the route build/reduce glue is measurable but not the main target. The padded BMM GEMM family still dominates, especially routed wgrad. Larger S improves padding from `1.625x` to `1.281x`, explaining why the 4K isolated train-path sum grows only modestly, but this also means more progress requires a compact/fused grouped-MoE training kernel rather than another small reduction edit.

Ran the retained compact-reduce learned-router path through the full single-layer `run_microbench.py` harness to tie the isolated MoE evidence back to the actual layer contract `[B,S,7168] -> [B,S,7168]`, BF16, full forward+loss+backward+gradients. After fallback GPU0 results on `do-sonle5-mi350-gpu`, canonical `do-sonle-kernel` became idle and was used for the official single-GPU confirmation.

Command family:

```bash
python3 dsv4_layer_microbench/run_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen <2048-or-4096> --dtype bfloat16 --device cuda --warmup 2 --iters <5-or-3> --rmsnorm-save-policy saved-accum --json
```

| Shape | Iters | Total ms | Min ms | Tokens/s | TFLOPs | Loss | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S=2048` canonical `do-sonle-kernel` | 5 | `52.6859` | `52.3743` | `38,871.88` | `184.7962` | `1.0037039` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_layer_compact_reduce_canonical_20260531/s2048_summary.json` |
| `S=4096` canonical `do-sonle-kernel` | 3 | `76.4464` | `76.2999` | `53,580.04` | `276.9669` | `1.0032604` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/full_layer_compact_reduce_canonical_20260531/s4096_summary.json` |
| `S=2048` | 5 | `52.2637` | `52.0838` | `39,185.89` | `186.2890` | `1.0037039` | `/local/data/sonle5/dsv4_layer_microbench_fallback_20260531/profiles/full_layer_compact_reduce_fallback_8xnode_20260531/s2048_summary.json` |
| `S=4096` | 3 | `75.4483` | `75.3341` | `54,288.86` | `280.6309` | `1.0032605` | `/local/data/sonle5/dsv4_layer_microbench_fallback_20260531/profiles/full_layer_compact_reduce_fallback_8xnode_20260531/s4096_summary.json` |

The canonical S4096 run is below the Ascend proxy single-layer time target of `86.6 ms` by about `10.15 ms`, while canonical S2048 is still above the `43.3 ms` development proxy by about `9.39 ms`. This is a PyTorch packed-sim microbench on one MI350 and not yet an 8x TorchTitan end-to-end result. Watched gradients were present for input, attention projections, router gate, routed expert weights, and shared expert weights.

One canonical S4096 `--profile-blocks` pass with the same full-layer command reported total `78.9972 ms` under block-timer synchronization overhead. Forward block means from that pass:

| Block | ms |
| --- | ---: |
| `moe.routed_experts` | `13.9964` |
| `moe.router` | `1.1969` |
| `moe.shared_expert` | `0.7660` |
| `attn.sparse_mla` | `7.0322` |
| `attn.q_proj` | `1.8352` |
| `attn.out_proj` | `1.1406` |
| `attn.compress_csa` | `0.4584` |
| `attn.compress_hca` | `0.3488` |
| `attn.kv_proj` | `0.2451` |
| `layer.attn_norm + layer.moe_norm` | `0.5825` |
| `layer.attn_residual + layer.moe_residual` | `0.0986` |

Canonical S4096 isolated compact-reduce MoE-op probe:

| Shape | Padding multiplier | Forward total ms | Backward dgrad path ms | Backward wgrad GEMM ms | Router-score ms | Backward total ms | Train isolated ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S=4096` | `1.2813` | `14.7225` | `12.7435` | `18.5385` | `0.5331` | `31.8151` | `46.5376` | `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/dynamic_moe_bridge_ops_compact_reduce_s4096_canonical_20260531/summary.json` |

Representative canonical S4096 MoE-op medians: fwd W1/W3/W2 BMM `4.579 / 4.409 / 4.473 ms`, W2 dgrad `bmm(out=)` `3.874 ms`, W1+W3 xgrad `baddbmm` `7.547 ms`, input reduce-only `0.760 ms`, and W2/W1/W3 wgrad BMM `6.356 / 6.097 / 6.086 ms`. BF16 reduction-order checks used `--check-atol 1.0 --check-rtol 0.025` and passed; grad-y build and router-score gradient remained exact.

Read: canonical full-layer confirmation shows the retained learned-router bridge is now under the 4K single-layer proxy time in this harness. The remaining high-leverage work shifts from small PyTorch bridge edits to true packed-FP4/fused compact grouped-MoE training kernels, sparse MLA fused forward/backward, and then the 8x TorchTitan DSv4-style integration target.

Canonical S4096 selected-region rocprofv3 capture with ROCTx forward block ranges:

```bash
/opt/rocm/bin/rocprofv3 --selected-regions --marker-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --output-directory profiles/rocprof_s4096_dynamic_compact_reduce_canonical_region_1iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --rmsnorm-save-policy saved-accum --no-torch-trace --roctx-profile-region --roctx-block-ranges --trace-dir profiles/rocprof_s4096_dynamic_compact_reduce_canonical_region_1iter/torch_summary
```

`summarize_rocprof_blocks.py` attributed `77.7266 ms` of `77.7572 ms` kernel time across `494` dispatches. Top attributed blocks:

| ROCTx block | Marker ms | Kernel ms | Dispatches | Main read |
| --- | ---: | ---: | ---: | --- |
| `phase.backward` | `50.7848` | `50.5566` | `269` | backward dominates; top GEMM-family kernel buckets were `19.2484 ms` across 3 calls and `9.6726 ms` across 3 calls |
| `forward.moe.routed_experts` | `14.5446` | `14.1063` | `52` | routed MoE forward is mostly three GEMM calls, `12.3239 ms` total |
| `forward.attn.sparse_mla` | `7.3334` | `7.2097` | `19` | masked softmax was `2.3968 ms`; remaining time is GEMM/copy/reduce-style attention kernels |
| `forward.attn.q_proj` | `1.9116` | `1.6797` | `13` | q RMSNorm plus projection |
| `forward.attn.out_proj` | `1.2715` | `1.1540` | `3` | output projection GEMMs |
| `forward.moe.router` | `1.2299` | `0.9166` | `36` | router GEMM/top-k/reductions |

Read: the kernel trace agrees with the timer and isolated-op story. Forward range attribution is detailed, while `phase.backward` is still coarse because the normal `loss.backward` path has no internal ROCTx ranges. Within that coarse bucket, GEMM-family routed MoE backward work is the largest visible kernel mass; attention softmax backward is visible but much smaller. Artifact root: `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/rocprof_s4096_dynamic_compact_reduce_canonical_region_1iter/`, including `trace_kernel_trace.csv`, `trace_marker_api_trace.csv`, `block_summary.json`, and `torch_summary/summary.json`.

Checked the existing compact AITER grouped-GEMM probe on the same fallback GPU. It uses learned-router-like non-uniform group sizes without dispatch/combine/activation. Correctness passed for all `gmm`, `ptgmm`, and `nptgmm` ops against per-expert PyTorch references, but timings are negative versus the retained padded PyTorch BMM bridge:

| Shape | AITER fwd W1 ms | AITER fwd W2 ms | AITER W2 dgrad ms | AITER W1+W3 xgrad pair ms | AITER ptgmm W2/W1 wgrad ms | AITER nptgmm W2/W1 wgrad ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S=2048` | `6.986` | `6.748` | `6.688` | `13.561` | `7.088 / 6.934` | `7.288 / 7.195` | `/local/data/sonle5/dsv4_layer_microbench_fallback_20260531/profiles/aiter_gmm_dynamic_moe_s2048_fallback_8xnode_20260531/summary.json` |
| `S=4096` | `7.120` | `6.774` | `6.821` | `13.882` | `8.066 / 8.026` | `8.795 / 8.628` | `/local/data/sonle5/dsv4_layer_microbench_fallback_20260531/profiles/aiter_gmm_dynamic_moe_s4096_fallback_8xnode_20260531/summary.json` |

Read: AITER's current Triton grouped-GEMM wrappers are correctness-clean for these compact dynamic groups but are not a drop-in performance replacement for the padded ROCm batched-BMM path. Keep them as a correctness reference and possible integration surface, but do not switch the retained learned-router bridge to AITER without a lower-level kernel change.

Added `--blas-backend {current,default,hipblas,hipblaslt,ck}` to `run_microbench.py` so full-layer runs can record the effective `torch.backends.cuda.preferred_blas_library` selector in JSON. Rechecked the retained learned-router bridge on the MI350 with default hipBLASLt/Cublaslt versus forced `hipblas`.

Command family:

```bash
python3 dsv4_layer_microbench/run_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dynamic-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen <2048-or-4096> --dtype bfloat16 --device cuda --warmup 2 --iters <5-or-3> --rmsnorm-save-policy saved-accum --blas-backend <current-or-hipblas> --json
```

| Shape | Backend request | Effective backend | Iters | Total ms | Min ms | Tokens/s | TFLOPs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `S=2048` | `current` | `Cublaslt` | 5 | `53.2378` | `53.0904` | `38,468.90` | `182.8804` |
| `S=2048` | `hipblas` | `Cublas` | 5 | `57.2407` | `54.7707` | `35,778.72` | `170.0914` |
| `S=4096` | `current` | `Cublaslt` | 3 | `77.2151` | `76.9584` | `53,046.64` | `274.2096` |
| `S=4096` | `hipblas` | `Cublas` | 3 | `78.2521` | `78.0515` | `52,343.61` | `270.5755` |

Read: forcing `hipblas` is negative for the dynamic learned-router full layer: about `+4.00 ms` at S2048 and `+1.04 ms` at S4096. This matches the earlier isolated BMM selector probe where `hipblas` was mixed and CK was not a drop-in backend for these grouped-BMM contracts. Keep the PyTorch ROCm default hipBLASLt/Cublaslt selector for the retained dynamic bridge; the remaining gap still needs a real compact/fused grouped-MoE training kernel rather than a backend override.

## Environment

- Node: `do-sonle-kernel` (`sonle5-kernel-dev-mi350x1`)
- GPU: AMD Instinct MI350X VF, `gfx950`
- Container: `b9d33e6e8227`, image `rocm/sgl-dev:rocm720-mi35x-8c3b5aa-20260521-DSv4`
- PyTorch: `2.9.1+rocm7.2.0.git7e1940d4`
- Staged path: `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530`

## Correctness Gates

Command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --json
```

Result: pass.

| Gate | Result | Notes |
| --- | --- | --- |
| `rope` | pass | Round-trip max error `2.22e-16` |
| `swiglu` | pass | Float64 gradcheck |
| `attention` | pass | Float64 gradcheck, gradcheck preset `[1, 4, 8]` |
| `moe` | pass | Float64 gradcheck with fixed router |
| `layer` | pass | Composed layer float64 gradcheck with fixed router |
| `fp4_scale` | pass | Gradcheck through FP4 dequant scale handling; scale grad norm observed |
| `fp4_chunked_match` | pass | Forced chunked FP4 quant/dequant matched unchunked quant/dequant exactly |
| `fp4_layer_match` | pass | FP4 fake-quant layer vs reference with fixed router |
| `fp4_packed_sim_match` | pass | Packed-sim FP4 exactly matched fake FP4 outputs and watched grads |
| `moe_grouped_match` | pass | Grouped routed MoE matched loop routed MoE outputs and watched grads |
| `moe_token_sum_match` | pass | Token-sum grouped MoE matched loop routed MoE outputs and watched grads |
| `moe_contiguous_match` | pass | Fixed-router contiguous grouped MoE matched loop routed MoE outputs and watched grads |
| `moe_contiguous_fused_match` | pass | Fused contiguous packed-sim path matched loop routed MoE outputs and watched grads |
| `moe_expert_batches_match` | pass | Equal-count grouped expert-batch MoE matched loop routed MoE outputs and watched grads |
| `moe_expert_batches_uniform_dense_wgrad_match` | pass | Fixed-router uniform-score expert-batch path matched loop routed MoE outputs and watched grads |
| `moe_expert_batches_cyclic_uniform_dense_wgrad_match` | pass | Fixed-router cyclic uniform expert-batch path matched loop routed MoE outputs and watched grads |
| `moe_expert_batches_cyclic_uniform_topk_wgrad_baddbmm_xgrad_match` | pass | Fixed-router cyclic uniform grouped-top-k wgrad path matched loop routed MoE outputs and watched grads |
| `moe_expert_batches_cyclic_uniform_shared_x_dense_wgrad_match` | pass | Fixed-router cyclic uniform shared-x expert-batch path matched loop routed MoE outputs and watched grads |

FP4 layer match from the passing run:

- Output max abs error: `0.0023357889149338007`
- Output relative L2: `0.0011688831408293444`
- Max watched gradient relative L2: `0.43903726764258577`
- Reference loss: `0.9675203694916306`
- FP4 fake-quant loss: `0.9679165096595618`

Packed-sim match from the passing run:

- Output max abs error vs `fp4-fake`: `0.0`
- Output relative L2 vs `fp4-fake`: `0.0`
- Max watched gradient relative L2 vs `fp4-fake`: `0.0`

Grouped MoE match from the passing run:

- Output max abs error vs loop MoE: `0.0`
- Output relative L2 vs loop MoE: `0.0`
- Max watched gradient relative L2 vs loop MoE: `2.0830982360529346e-16`

Focused token-sum and contiguous grouped MoE match runs:

- `moe_token_sum_match`: output max abs error `0.0`; max watched gradient relative L2 `1.7298927068285576e-16`.
- `moe_contiguous_match`: output max abs error `0.0`; max watched gradient relative L2 `1.7298927068285576e-16`.

Full-suite rerun after adding packed-sim cache owner validation and the contiguous batched matvec helper:

- `moe_token_sum_match`: output max abs error `0.0`; max watched gradient relative L2 `1.5940681828123781e-16`.
- `moe_contiguous_match`: output max abs error `0.0`; max watched gradient relative L2 `1.3447406271813868e-16`.

Full-suite rerun after adding sparse selected-row wgrad for the contiguous packed-sim path:

- `moe_token_sum_match`: output max abs error `0.0`; max watched gradient relative L2 `1.5940681828123781e-16`.
- `moe_contiguous_match`: output max abs error `0.0`; max watched gradient relative L2 `1.3447406271813868e-16`.
- `moe_contiguous_fused_match`: output max abs error `0.0`; max watched gradient relative L2 `1.9338834054140394e-16`.

Full-suite rerun after adding chunked packed-sim FP4 materialization:

- `fp4_chunked_match`: forced one-row chunks matched unchunked quant/dequant with output max abs error `0.0` and relative L2 `0.0`.
- `fp4_packed_sim_match`: output max abs error `0.0`; max watched gradient relative L2 `0.0`.
- `moe_contiguous_match`: output max abs error `0.0`; max watched gradient relative L2 `1.25805230108868e-16`.
- `moe_contiguous_fused_match`: output max abs error `0.0`; max watched gradient relative L2 `1.5314550468916379e-16`.

Focused grouped expert-batch MoE match run:

- `moe_expert_batches_match`: output max abs error `0.0`; max watched gradient relative L2 `8.8929235628238e-17`.

Focused cyclic uniform grouped-top-k wgrad match run:

- CUDA command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_cyclic_uniform_topk_wgrad_baddbmm_xgrad_match --json`
- CUDA result: pass. Output max abs error `0.0`; max watched gradient relative L2 `6.449652832387695e-17`.

Focused route-cache validation after caching grouped expert assignment metadata:

- CPU command: `python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_match,moe_expert_batches_fused_w13_match --json`
- CPU result: pass. Non-fused output max abs error `0.0`, max watched gradient relative L2 `3.6501615255140047e-16`; fused W13 output max abs error `1.0842021724855044e-19`, max watched gradient relative L2 `1.576641797663692e-16`.
- CUDA command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_match,moe_expert_batches_fused_w13_match --json`
- CUDA result: pass. Non-fused output max abs error `0.0`, max watched gradient relative L2 `8.8929235628238e-17`; fused W13 output max abs error `0.0`, max watched gradient relative L2 `8.123493959409061e-17`.

Focused sparse-MLA metadata-cache validation after caching static positions and invalid attention masks:

- CPU commands: `python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks attention,layer --json` and `python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks fp4_layer_match,fp4_packed_sim_match --json`
- CPU result: pass. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002444660756736994`, max watched gradient relative L2 `0.4627755005321434`; packed-sim matched fake FP4 exactly.
- CUDA command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer,fp4_layer_match,fp4_packed_sim_match --json`
- CUDA result: pass. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002455961424857378`, max watched gradient relative L2 `0.3540374210961061`; packed-sim matched fake FP4 exactly.

Focused RoPE-table cache validation after caching fixed-shape cosine/sine tables and reusing compressed group-end positions:

- CPU command: `python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks rope,attention,layer,fp4_layer_match,fp4_packed_sim_match --json`
- CPU result: pass. RoPE round-trip max error `2.636779683484747e-16`; attention and layer gradcheck passed; FP4 layer match output max abs error `0.0045593855902552605`, max watched gradient relative L2 `0.40758122682002895`; packed-sim matched fake FP4 exactly.
- CUDA command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks rope,attention,layer,fp4_layer_match,fp4_packed_sim_match --json`
- CUDA result: pass. RoPE round-trip max error `2.220446049250313e-16`; attention and layer gradcheck passed; FP4 layer match output max abs error `0.001556938630528748`, max watched gradient relative L2 `0.6793706009395092`; packed-sim matched fake FP4 exactly.

Focused custom RMSNorm validation after replacing decomposed PyTorch RMSNorm autograd with an explicit saved-rstd backward:

- CPU command: `python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks attention,moe,layer,fp4_layer_match,fp4_packed_sim_match --json`
- CPU result: pass. Attention, MoE, and layer gradcheck passed; FP4 layer match output max abs error `0.005716319661587477`, max watched gradient relative L2 `0.7121041122277622`; packed-sim matched fake FP4 exactly.
- CUDA command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,moe,layer,fp4_layer_match,fp4_packed_sim_match --json`
- CUDA result: pass. Attention, MoE, and layer gradcheck passed; FP4 layer match output max abs error `0.0032427513506263494`, max watched gradient relative L2 `0.25099587947965296`; packed-sim matched fake FP4 exactly.

Focused memory-light RMSNorm rerun after saving original `x`, original `weight`, and `rstd` instead of saved accumulated tensors:

- CUDA command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,moe,layer,fp4_layer_match,fp4_packed_sim_match --json`
- CUDA result: pass. Attention, MoE, and layer gradcheck passed; FP4 layer match output max abs error `0.0032427513506263494`, max watched gradient relative L2 `0.25099587947965296`; packed-sim matched fake FP4 exactly.
- S1024 no-trace command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_memlight_perf_2iter`
- S1024 result: `53.8056 ms`, `19031.47 tokens/s`, `86.523982 TFLOPs`; forward attention aggregate `8.0671 ms`, forward MoE aggregate `9.1614 ms`, boundary aggregate `0.4134 ms`.
- S2048 no-trace command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_memlight_perf_2iter`
- S2048 result: `97.6832 ms`, `20965.74 tokens/s`, `99.670762 TFLOPs`; forward attention aggregate `23.7880 ms`, forward MoE aggregate `9.7828 ms`, boundary aggregate `0.3939 ms`.
- S2048 warmed attribution command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --profile-attention-blocks --profile-moe-blocks --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_memlight_attr_1iter_warm`
- S2048 warmed attribution result: normal phase total `98.1873 ms`, `20858.09 tokens/s`, `99.158987 TFLOPs`; attention dgrad `36.5691 ms` / `38.2880 TFLOPs`, attention wgrad `36.9228 ms` / `37.9212 TFLOPs`, MoE dgrad `24.8591 ms` / `76.1925 TFLOPs`, MoE wgrad `25.1152 ms` / `75.4158 TFLOPs`.
- Read: the memory-light saved tensor policy reduces RMSNorm saved activation footprint, but the measured full-layer timing is slightly below the saved-accum RMSNorm run (`100.297167 TFLOPs` at S2048). This is a memory policy win with a small steady-state perf cost in the current PyTorch custom-autograd baseline.

Focused RMSNorm saved-tensor policy A/B after making the policy explicit in the harness:

- CPU commands: `python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks attention,moe,layer,fp4_layer_match,fp4_packed_sim_match --rmsnorm-save-policy memory-light --json` and the same command with `--rmsnorm-save-policy saved-accum`.
- CPU result: pass for both policies. Attention, MoE, and layer gradcheck passed; FP4 layer match output max abs error `0.005716319661587477`, max watched gradient relative L2 `0.7121041122277622`; packed-sim matched fake FP4 exactly.
- CUDA commands: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,moe,layer,fp4_layer_match,fp4_packed_sim_match --rmsnorm-save-policy memory-light --json` and the same command with `--rmsnorm-save-policy saved-accum`.
- CUDA result: pass for both policies. Attention, MoE, and layer gradcheck passed; FP4 layer match output max abs error `0.0032427513506263494`, max watched gradient relative L2 `0.25099587947965296`; packed-sim matched fake FP4 exactly.
- S2048 no-trace memory-light command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_memory_light_policy_2iter`
- S2048 no-trace saved-accum command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_saved_accum_policy_2iter`
- S2048 A/B result: memory-light `97.6065 ms`, `20982.22 tokens/s`, `99.749066 TFLOPs`; saved-accum `97.6442 ms`, `20974.11 tokens/s`, `99.710536 TFLOPs`.
- Static RMSNorm saved-tensor estimate at S2048: memory-light saves `67,684,352` input/weight bytes plus `34,880` rstd bytes; saved-accum saves `135,368,704` input/weight bytes plus `34,880` rstd bytes. Timing is flat in this PyTorch custom-autograd baseline, so memory-light remains the better default unless a fused backward kernel wants pre-cast accum tensors explicitly.

## Smoke Timings

Reference and FP4 paths are PyTorch implementations. The FP4 path currently simulates block-scaled E2M1/UE8M0 quantization and dequantization in Python/PyTorch with straight-through weight gradients; it is a correctness baseline, not an optimized packed FP4 kernel.

| Impl | Shape | Router | Avg ms | Tokens/s | Est. TFLOPs | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `reference` | tiny `B=2 S=8 d=64` | fixed | `11.05` | `1447.35` | `0.000251` | Full fwd+bwd, bf16 |
| `fp4-fake` | tiny `B=2 S=8 d=64` | fixed | `22.44` | `713.09` | `0.000124` | Full fwd+bwd, bf16 |
| `fp4-fake` | full `B=1 S=1 d=7168` | fixed | `8250.82` | `0.1212` | `0.000514` | Full real-dim fwd+bwd, bf16 |

## Profiling Snapshot

Trace artifacts were copied out of the repo to:

- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_tiny_fp4_fixed_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_tiny_fp4_fixed_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/tiny_fp4_fixed_backward_attr_2iter_v2/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1_fp4_backward_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/tiny_fp4_packed_sim_backward_attr_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1_fp4_packed_sim_backward_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s8_fp4_packed_sim_backward_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s8_fp4_packed_sim_grouped_moe_backward_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s16_fp4_packed_sim_grouped_moe_backward_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s16_fp4_packed_sim_grouped_moe_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s16_fp4_packed_sim_grouped_moe_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_full_b1s16_fp4_packed_sim_grouped_moe_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s16_fp4_packed_sim_grouped_token_sum_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s16_fp4_packed_sim_grouped_token_sum_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s16_fp4_packed_sim_grouped_contiguous_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s16_fp4_packed_sim_grouped_contiguous_batched_matvec_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_batched_matvec_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s16_fp4_packed_sim_grouped_contiguous_sparse_wgrad_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_sparse_wgrad_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s16_fp4_packed_sim_grouped_contiguous_fused_pair_w13_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_fused_pair_w13_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s32_fp4_packed_sim_grouped_contiguous_sparse_wgrad_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s64_fp4_packed_sim_grouped_contiguous_sparse_wgrad_chunked_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s64_fp4_packed_sim_grouped_contiguous_sparse_wgrad_chunked_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s64_fp4_packed_sim_grouped_expert_batches_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s128_fp4_packed_sim_grouped_expert_batches_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s128_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_full_b1s128_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s256_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s256_fp4_packed_sim_grouped_expert_batches_split_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s256_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s512_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_split_attr_1iter_retry/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/torch_full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter_clean/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_grouped_expert_batches_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/smoke_attn_attr_gradcheck_s4_diff/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_attention_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_attention_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s64_fp4_packed_sim_grouped_expert_batches_moe_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_moe_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_moe_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_moe_shared_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_moe_shared_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s64_fp4_packed_sim_grouped_expert_batches_fused_w13_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_fused_w13_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_fused_w13_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_route_cache_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_route_cache_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_attn_meta_cache_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_attn_meta_cache_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_rope_cache_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rope_cache_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s1024_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_memlight_perf_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_memlight_perf_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_memlight_attr_1iter_warm/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_memory_light_policy_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_saved_accum_policy_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_shared_linear_pair_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_shared_fused_w13_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_shared_linear_pair_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_shared_fused_w13_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_gradout_sorted_token_2iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_gradout_sorted_token_attr_1iter/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/traffic_fp4_storage_cpu_smoke/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/backward_shape_contract_cpu_smoke_20260530/`
- `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_route_precheck_removed_2iter/`

PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset tiny --impl fp4-fake --batch 2 --seqlen 8 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir /local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530/profiles/torch_tiny_fp4_fixed_2iter
```

Result:

- Forward phase: `24.4819 ms`
- Loss phase: `0.1394 ms`
- Backward phase: `8.8962 ms`
- End-to-end phase total: `33.5175 ms`
- Tokens/s: `477.36`
- Estimated TFLOPs: `0.000083`

Forward block timing from the PyTorch-profiled run:

| Block | Avg ms |
| --- | ---: |
| `moe.routed_experts` | `12.7533` |
| `moe.shared_expert` | `2.9701` |
| `attn.q_proj` | `1.7459` |
| `attn.compress_hca` | `1.6297` |
| `attn.compress_csa` | `1.5519` |
| `attn.kv_proj` | `1.0813` |
| `attn.out_proj` | `0.7012` |
| `attn.sparse_mla` | `0.6957` |

rocprofv3 command:

```bash
/opt/rocm/bin/rocprofv3 --runtime-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_tiny_fp4_fixed_2iter/summary.txt --output-directory profiles/rocprof_tiny_fp4_fixed_2iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset tiny --impl fp4-fake --batch 2 --seqlen 8 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/rocprof_tiny_fp4_fixed_2iter/torch_summary
```

Result:

- Forward phase: `42.6091 ms`
- Loss phase: `0.0978 ms`
- Backward phase: `14.5997 ms`
- End-to-end phase total: `57.3066 ms`
- Tokens/s: `279.20`
- Estimated TFLOPs: `0.000049`

rocprof kernel statistics show that this fake-FP4 baseline is dominated by PyTorch quantize/dequantize overhead rather than packed FP4 GEMM. Top entries include `__amd_rocclr_copyBuffer`, `MaxNan` reductions, `AbsFunctor`, elementwise kernels, index/argmin kernels, and only a small CK/hipBLASLt GEMM share. This is useful as a correctness and instrumentation baseline, but it is not a performance target.

## Backward Dgrad/Wgrad Attribution

`profile_microbench.py` now supports `--profile-backward-blocks`. This keeps the normal performance timing on `loss.backward()`, then runs a separate segmented `autograd.grad` attribution pass to split major blocks into dgrad and wgrad buckets. With `--profile-attention-blocks`, the attention module is also recomputed with the real upstream `d_attn_out` and split into output projection, inverse RoPE, sparse MLA, q path, kv path, CSA, and HCA backward edges. With `--profile-moe-blocks`, the grouped expert-batch routed MoE path is recomputed with the real upstream `d_moe_out` and split into output scale/reduce, W2 dgrad/wgrad, SwiGLU dgrad, W1/W3 dgrad/wgrad, and input gather/reduce. The shared expert is also split into shared W2 dgrad/wgrad, shared SwiGLU dgrad, and shared W1/W3 dgrad/wgrad. The attribution mode is useful for ranking work, but it is not a replacement for rocprof kernel attribution because dgrad and wgrad are issued as separate `autograd.grad` calls.

Profiler summaries now also include `traffic_estimate`, `performance_by_block`, and `subsystem_summary`. `traffic_estimate` is a static shape byte estimate for boundary activations, attention activations, MoE activations, gradient edges, and parameter gradients. `performance_by_block` divides approximate static FLOPs by measured phase and sub-block milliseconds so each run directly reports per-block tokens/s and estimated TFLOPs; router FLOPs are set to zero when `--fixed-router` bypasses the router GEMM. `subsystem_summary` rolls the block list up to attention, MoE, and boundary buckets, plus dgrad/wgrad buckets when segmented backward attribution is enabled. This is reporting metadata, not a replacement for rocprof memory counters or kernel-level utilization.

Tiny FP4 attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset tiny --impl fp4-fake --batch 2 --seqlen 8 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --profile-backward-blocks --backward-attribution-iters 2 --backward-attribution-warmup 1 --no-torch-trace --trace-dir profiles/tiny_fp4_fixed_backward_attr_2iter_v2
```

Full real-dim FP4 attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-fake --batch 1 --seqlen 1 --dtype bfloat16 --device cuda --warmup 0 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s1_fp4_backward_attr_1iter
```

Normal fwd+bwd phase timing:

| Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny `B=2 S=8 d=64` | `16.7212` | `5.0470` | `21.8313` | `732.89` | `0.000127` |
| full `B=1 S=1 d=7168` | `4588.7258` | `3616.4930` | `8205.2995` | `0.1219` | `0.000517` |

Segmented backward attribution:

| Block | Tiny ms | Full B1S1 ms |
| --- | ---: | ---: |
| `loss.dout` | `0.2589` | `0.5164` |
| `moe.dgrad` | `2.0554` | `134.8780` |
| `moe.wgrad` | `1.8553` | `180.1438` |
| `layer.moe_norm.dgrad` | `0.2318` | `0.4979` |
| `layer.moe_norm.wgrad` | `0.1346` | `0.2255` |
| `attention.dgrad` | `2.0646` | `2.5141` |
| `attention.wgrad` | `2.1996` | `2.5082` |
| `layer.attn_norm.dgrad` | `0.2455` | `0.2803` |
| `layer.attn_norm.wgrad` | `0.0906` | `0.1207` |

Full-shape gradient norms from the normal `loss.backward()` path:

- Input grad norm: `0.02476002834737301`
- `attention.wq_a`: `0.5608614683151245`
- `attention.wq_b`: `0.5484263896942139`
- `attention.wkv`: `0.48841559886932373`
- `attention.wo_b`: `0.666267454624176`
- `moe.w1`: `0.07908806204795837`
- `moe.w2`: `0.07523109763860703`
- `moe.shared_w1`: `0.20564144849777222`
- `moe.gate`: `null` in fixed-router mode by design

## Packed-Sim FP4 Path

`fp4-packed-sim` is a performance-oriented path beside `fp4-fake`. It precomputes cached dequantized FP4 weights keyed by parameter pointer/version and uses a custom straight-through linear backward, so measured iterations behave closer to a packed-weight training kernel while still producing input and weight gradients. It is not a real packed FP4 GEMM kernel.

Correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks fp4_packed_sim_match --json
```

Result: pass. Packed-sim matched `fp4-fake` exactly for output, loss, input grad, attention weight grads, MoE weight grads, and shared expert weight grads on the gradcheck preset. Max output abs error and max watched grad relative L2 were both `0.0`.

Packed-sim timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset tiny --impl fp4-packed-sim --batch 2 --seqlen 8 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --profile-backward-blocks --backward-attribution-iters 2 --backward-attribution-warmup 1 --no-torch-trace --trace-dir profiles/tiny_fp4_packed_sim_backward_attr_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --batch 1 --seqlen 1 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s1_fp4_packed_sim_backward_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --batch 1 --seqlen 8 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s8_fp4_packed_sim_backward_attr_1iter
```

Normal fwd+bwd phase timing:

| Impl | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fp4-fake` | tiny `B=2 S=8 d=64` | `16.7212` | `5.0470` | `21.8313` | `732.89` | `0.000127` |
| `fp4-packed-sim` | tiny `B=2 S=8 d=64` | `5.8498` | `5.4531` | `11.3530` | `1409.32` | `0.000245` |
| `fp4-fake` | full `B=1 S=1 d=7168` | `4588.7258` | `3616.4930` | `8205.2995` | `0.1219` | `0.000517` |
| `fp4-packed-sim` | full `B=1 S=1 d=7168` | `6.3637` | `174.5166` | `180.9378` | `5.5268` | `0.023436` |
| `fp4-packed-sim` | full `B=1 S=8 d=7168` | `16.8998` | `1568.7762` | `1585.7303` | `5.0450` | `0.021425` |

Packed-sim full-shape backward attribution:

| Block | Full B1S1 ms | Full B1S8 ms |
| --- | ---: | ---: |
| `loss.dout` | `0.4553` | `0.8663` |
| `moe.dgrad` | `134.7029` | `147.8240` |
| `moe.wgrad` | `172.9504` | `1556.0629` |
| `layer.moe_norm.dgrad` | `0.3644` | `0.6902` |
| `layer.moe_norm.wgrad` | `0.1904` | `0.2150` |
| `attention.dgrad` | `2.5717` | `3.1164` |
| `attention.wgrad` | `2.5274` | `2.5615` |
| `layer.attn_norm.dgrad` | `0.2779` | `0.2729` |
| `layer.attn_norm.wgrad` | `0.1247` | `0.1224` |

Current read: once per-call FP4 quantize/dequantize is removed from the timed loop, the full `B=1 S=8` loop-MoE path is dominated by MoE weight-gradient work. The reference MoE still loops over selected experts and emits many tiny expert GEMMs, so the next optimization target was routed expert grouping.

## Grouped Routed MoE Path

`--moe-impl grouped-assignments` adds a swappable routed MoE implementation beside the default loop reference. It gathers the selected expert weights for the current token assignments, runs assignment-batched `bmm` for `w1`, `w3`, and `w2`, and scatters the weighted outputs back with `index_add_`. For `fp4-packed-sim`, selected expert weights are cached after FP4 dequantization instead of caching all 384 experts at once.

Grouped MoE correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_grouped_match --json
```

Result: pass. Grouped MoE matched loop MoE with output max abs error `0.0`; max watched gradient relative L2 was `1.7298927068285576e-16` in the focused run and `2.0830982360529346e-16` in the final default correctness run.

Grouped MoE timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-assignments --batch 1 --seqlen 8 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s8_fp4_packed_sim_grouped_moe_backward_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-assignments --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s16_fp4_packed_sim_grouped_moe_backward_attr_1iter
```

Normal fwd+bwd phase timing:

| Impl | MoE impl | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `fp4-packed-sim` | `loop` | full `B=1 S=8 d=7168` | `16.8998` | `1568.7762` | `1585.7303` | `5.0450` | `0.021425` |
| `fp4-packed-sim` | `grouped-assignments` | full `B=1 S=8 d=7168` | `16.8945` | `30.8729` | `47.8221` | `167.2866` | `0.710421` |
| `fp4-packed-sim` | `grouped-assignments` | full `B=1 S=16 d=7168` | `27.4133` | `51.9644` | `79.4409` | `201.4075` | `0.856908` |

Grouped MoE backward attribution:

| Block | B1S8 ms | B1S16 ms |
| --- | ---: | ---: |
| `loss.dout` | `0.3922` | `0.5759` |
| `moe.dgrad` | `136.2952` | `133.9670` |
| `moe.wgrad` | `29.3680` | `49.8858` |
| `layer.moe_norm.dgrad` | `0.2855` | `0.2833` |
| `layer.moe_norm.wgrad` | `0.2081` | `0.1822` |
| `attention.dgrad` | `2.6067` | `2.5234` |
| `attention.wgrad` | `2.4422` | `2.5675` |
| `layer.attn_norm.dgrad` | `0.2470` | `0.2811` |
| `layer.attn_norm.wgrad` | `0.1145` | `0.1172` |

The attribution pass now also splits routed and shared MoE backward buckets on `B=1 S=16`:

| Block | B1S16 split ms |
| --- | ---: |
| `moe.routed.dgrad` | `139.0773` |
| `moe.routed.wgrad` | `49.7387` |
| `moe.shared.dgrad` | `0.3961` |
| `moe.shared.wgrad` | `0.3451` |

Steady-state PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-assignments --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s16_fp4_packed_sim_grouped_moe_2iter
```

Result:

- Forward phase: `28.5025 ms`
- Backward phase: `52.8713 ms`
- End-to-end phase total: `81.4548 ms`
- Tokens/s: `196.43`
- Estimated TFLOPs: `0.835721`

Top steady-state PyTorch profiler entries:

| Entry | Self CUDA |
| --- | ---: |
| `aten::index_add_` | `74.036 ms` |
| `forward.moe.routed_experts` | `46.166 ms` |
| `aten::bmm` | `22.466 ms` |
| `aten::fill_` | `14.941 ms` |
| `aten::sub` | `12.868 ms` |
| `aten::add` | `12.790 ms` |
| `aten::gather` | `9.818 ms` |

rocprofv3 command:

```bash
/opt/rocm/bin/rocprofv3 --runtime-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_full_b1s16_fp4_packed_sim_grouped_moe_2iter/summary.txt --output-directory profiles/rocprof_full_b1s16_fp4_packed_sim_grouped_moe_2iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-assignments --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/rocprof_full_b1s16_fp4_packed_sim_grouped_moe_2iter/torch_summary
```

The rocprof phase summary was `189.06 tokens/s` and `0.804392` estimated TFLOPs. Its kernel stats include warmup/cache materialization because rocprof wraps the full Python process; as expected, the top kernel was an FP4 prepack `ArgMin` reduce. Use the PyTorch profiler above for steady-state ranked work and the rocprof artifacts for raw kernel traces.

Current read: grouped assignments remove the loop-mode routed MoE wgrad bottleneck for the measured single-GPU token blocks. The main remaining issue is routed MoE scatter/gather and selected-weight materialization around the assignment-batched path: steady-state profiling is dominated by `index_add_`/large-index scatter, with `bmm`, gather, and huge temporary selected-weight tensors behind it. A real grouped FP4 kernel should avoid the selected-weight materialization traffic and fold scatter/reduce into the expert kernel path.

### Token-Sum and Contiguous Grouping Probes

`--moe-impl grouped-token-sum` changes the output combine from `index_add_` to a token-major `view(...).sum(dim=1)` while keeping selected expert weight gathers. It matched the loop MoE exactly, but steady-state profiling still showed `aten::index_add_` as the top CUDA entry because weight `index_select` backward scatters gradients back into the expert table.

Token-sum split-attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-token-sum --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s16_fp4_packed_sim_grouped_token_sum_split_attr_1iter
```

Token-sum timing:

- Forward phase: `27.6620 ms`
- Backward phase: `52.3138 ms`
- End-to-end phase total: `80.0406 ms`
- Tokens/s: `199.90`
- Estimated TFLOPs: `0.850488`
- Split attribution: `moe.routed.dgrad` `135.0594 ms`; `moe.routed.wgrad` `49.7378 ms`; `moe.shared.dgrad` `0.5593 ms`; `moe.shared.wgrad` `0.3679 ms`

Token-sum steady-state PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-token-sum --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s16_fp4_packed_sim_grouped_token_sum_2iter
```

Token-sum profiler result: `86.8690 ms` total, `184.19 tokens/s`, `0.783635` estimated TFLOPs. Top CUDA entries included `aten::index_add_` at `85.362 ms`, `forward.moe.routed_experts` at `46.303 ms`, `aten::bmm` at `22.646 ms`, and `aten::gather` at `9.312 ms`.

`--moe-impl grouped-contiguous` is a fixed-router layout ceiling path for the balanced canary where expert assignments are exactly `0..assignments-1`. It slices contiguous expert ranges instead of gathering selected expert weights and uses token-major sum for the output combine; it falls back to token-sum if the assignment layout is not contiguous.

Contiguous split-attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s16_fp4_packed_sim_grouped_contiguous_split_attr_1iter
```

Contiguous timing:

- Forward phase: `22.8925 ms`
- Backward phase: `20.2407 ms`
- End-to-end phase total: `43.1982 ms`
- Tokens/s: `370.39`
- Estimated TFLOPs: `1.575844`
- Forward `moe.routed_experts`: `18.8978 ms`
- Split attribution: `moe.routed.dgrad` `139.8147 ms`; `moe.routed.wgrad` `17.9696 ms`; `moe.shared.dgrad` `0.4045 ms`; `moe.shared.wgrad` `0.3414 ms`

Contiguous steady-state PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_2iter
```

Contiguous profiler result: `45.3995 ms` total, `352.43 tokens/s`, `1.499434` estimated TFLOPs. `aten::index_add_` was removed from the top entries. Remaining top CUDA work was selected-weight materialization/STE traffic and batched matvec work: vectorized elementwise kernels around `26.265 ms` and `14.807 ms`, `aten::bmm` at `22.898 ms`, `aten::fill_` at `14.929 ms`, `aten::add`/`aten::sub` around `13 ms`, `aten::copy_` at `12.478 ms`, DtoD memcpy at `11.011 ms`, and CK GEMM at `10.952 ms`.

Current read from these probes: output-combine scatter was not the only issue. The indexed selected-weight path also scatters weight gradients. The contiguous canary removes generic indexed gradient scatter and roughly doubles throughput over grouped assignments, but still pays PyTorch temporary tensor and STE materialization overhead. The next useful kernel target is a grouped FP4 expert matvec/GEMM path that consumes packed expert weights directly, computes dgrad/wgrad without materializing selected dequantized weights, and handles the output reduce inside the expert kernel.

### Packed-Sim Batched Matvec Update

The contiguous path now uses a custom packed-sim batched matvec helper for `[assignment, out, in] @ [assignment, in]`. It keeps the same straight-through weight gradient semantics, but avoids constructing `weight + (packed - weight).detach()` tensors before the routed expert `bmm` calls. Packed-sim caches now also validate weak tensor owners so pointer reuse across short-lived correctness models cannot return stale dequantized weights.

Updated contiguous split-attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s16_fp4_packed_sim_grouped_contiguous_batched_matvec_split_attr_1iter
```

Updated contiguous timing:

- Forward phase: `8.5997 ms`
- Backward phase: `20.5504 ms`
- End-to-end phase total: `29.2136 ms`
- Tokens/s: `547.69`
- Estimated TFLOPs: `2.330199`
- Forward `moe.routed_experts`: `4.6060 ms`
- Split attribution: `moe.routed.dgrad` `144.0287 ms`; `moe.routed.wgrad` `19.3763 ms`; `moe.shared.dgrad` `0.3932 ms`; `moe.shared.wgrad` `0.3267 ms`

Updated contiguous steady-state PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_batched_matvec_2iter
```

Updated contiguous profiler result: `31.1805 ms` total, `513.14 tokens/s`, `2.183212` estimated TFLOPs. Top CUDA entries are now `aten::bmm` at `20.694 ms`, `phase.forward` at `18.187 ms`, `aten::fill_` at `15.003 ms`, vectorized elementwise kernels around `14.880 ms`, `aten::copy_` at `12.727 ms`, DtoD memcpy at `11.240 ms`, and `forward.moe.routed_experts` at `9.227 ms` across two iterations. The earlier `aten::add`/`aten::sub` STE materialization entries fell out of the top profile rows.

Current read after this update: the harness has now isolated two separate costs. Indexed assignment routing is expensive because selected expert weight gradients scatter back into the expert table, and even a contiguous ceiling path still spends most time in PyTorch `bmm`, zero/fill, copies, and DtoD traffic. The next aligned implementation step is a real grouped expert kernel or custom autograd path that fuses the three routed expert projections/SwiGLU/reduce and writes compact wgrad/dgrad products without PyTorch temporaries.

### Sparse Selected-Row Wgrad Update

The contiguous packed-sim path now has a prefix batched matvec helper that takes the full expert table but returns sparse selected-row gradients for the touched contiguous expert prefix. This is still a PyTorch/autograd probe, not a production optimizer contract, but it models the compact wgrad product that a grouped expert kernel should emit before any global expert-table reduction.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_contiguous_match --json
```

Focused correctness result: pass. Contiguous grouped MoE matched the loop reference with output max abs error `0.0`; max watched gradient relative L2 was `1.7298927068285576e-16`.

Sparse-wgrad split-attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s16_fp4_packed_sim_grouped_contiguous_sparse_wgrad_split_attr_1iter
```

Sparse-wgrad timing:

- Forward phase: `8.7039 ms`
- Backward phase: `8.2836 ms`
- End-to-end phase total: `17.0523 ms`
- Tokens/s: `938.29`
- Estimated TFLOPs: `3.992056`
- Forward `moe.routed_experts`: `4.5274 ms`
- Split attribution: `moe.routed.dgrad` `143.4262 ms`; `moe.routed.wgrad` `6.3663 ms`; `moe.shared.dgrad` `0.4457 ms`; `moe.shared.wgrad` `0.3547 ms`

Sparse-wgrad steady-state PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_sparse_wgrad_2iter
```

Sparse-wgrad profiler result: `18.5585 ms` total, `862.14 tokens/s`, `3.668048` estimated TFLOPs. Top CUDA entries are now mostly matmul work: `aten::bmm` at `20.681 ms`, `phase.forward` at `18.519 ms`, `forward.moe.routed_experts` at `9.200 ms`, CK GEMM kernels at `8.805 ms`, `6.765 ms`, and `4.767 ms`, and `aten::copy_` at only `1.789 ms`. `aten::fill_` dropped from `15.003 ms` in the dense-gradient contiguous run to `0.228 ms`, and DtoD memcpy dropped from `11.240 ms` to `0.134 ms`.

Current read after sparse-wgrad: dense expert-table gradient materialization was a major hidden cost in the PyTorch ceiling path. Returning compact selected-row wgrad gets the single-layer full `B=1 S=16` path to `3.67-3.99` estimated TFLOPs, and the remaining visible work is dominated by the routed expert `bmm` calls themselves plus the forward pass. The next aligned kernel target is no longer generic scatter/fill elimination; it is fusing the three routed expert batched matvecs with SwiGLU and the token reduce, then replacing PyTorch `bmm` with an FP4 grouped GEMM/matvec kernel that produces compact selected-row dgrad/wgrad products directly.

### Fused Contiguous Routed-MoE Probe

`--moe-impl grouped-contiguous-fused` adds a separate packed-sim custom autograd path for the contiguous canary. It fuses the routed expert Python/autograd surface around `w1`, `w3`, SwiGLU, `w2`, score scaling, and token reduce while keeping compact sparse selected-row wgrad. A paired `w1/w3` projection was also tested to reduce batched-matvec launch count by concatenating the packed `w1` and `w3` prefixes along the expert hidden dimension.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_contiguous_fused_match --json
```

Focused correctness result: pass. Fused contiguous grouped MoE matched the loop reference with output max abs error `0.0`; max watched gradient relative L2 was `1.7298927068285576e-16`.

Fused paired-`w1/w3` split-attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous-fused --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s16_fp4_packed_sim_grouped_contiguous_fused_pair_w13_split_attr_1iter
```

Fused paired-`w1/w3` timing:

- Forward phase: `8.9052 ms`
- Backward phase: `12.0426 ms`
- End-to-end phase total: `21.0111 ms`
- Tokens/s: `761.50`
- Estimated TFLOPs: `3.239891`
- Forward `moe.routed_experts`: `4.6558 ms`
- Split attribution: `moe.routed.dgrad` `141.6575 ms`; `moe.routed.wgrad` `6.3859 ms`; `moe.shared.dgrad` `0.3768 ms`; `moe.shared.wgrad` `0.3422 ms`

Fused paired-`w1/w3` steady-state PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous-fused --batch 1 --seqlen 16 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s16_fp4_packed_sim_grouped_contiguous_fused_pair_w13_2iter
```

Fused paired-`w1/w3` profiler result: `22.4506 ms` total, `712.67 tokens/s`, `3.032143` estimated TFLOPs. This is slower than the unfused sparse-wgrad path (`18.5585 ms`, `862.14 tokens/s`, `3.668048` estimated TFLOPs). The paired projection reduced `aten::bmm` calls from `36` to `30` across two iterations, but it raised `aten::copy_` to `9.599 ms` and added a large vectorized elementwise band around `8.611 ms`. Current read: pairing `w1/w3` at the PyTorch tensor level is a false lead for this shape; the launch-count reduction is outweighed by combined-prefix materialization and less favorable CK shapes. The best current PyTorch ceiling remains `grouped-contiguous` with sparse selected-row wgrad.

### Contiguous Scaling and Full-Expert Envelope

`grouped-contiguous` was scaled beyond the original `B=1 S=16` canary. With top-6 routing and 384 routed experts, `B=1 S=64` is the largest fixed-router contiguous shape before the balanced router wraps and the implementation falls back to token-sum. It touches every routed expert exactly once.

The first `B=1 S=64` attempt exposed a packed-sim prepack OOM:

- Failure site: `_cached_packed_sim_weight(...)` while quantizing the full `384 x 3072 x 7168` routed expert prefix.
- Failure mode: `torch.OutOfMemoryError`, trying to allocate `126.00 GiB` for the FP4 E2M1 distance tensor with `96.00 GiB` free.
- Fix: packed-sim materialization now uses `quantize_dequantize_fp4_e2m1_chunked(...)` for large 3D tensors, so expert prefixes are packed in row chunks instead of one huge temporary.

Scaling split-attribution commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 32 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s32_fp4_packed_sim_grouped_contiguous_sparse_wgrad_split_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 64 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s64_fp4_packed_sim_grouped_contiguous_sparse_wgrad_chunked_split_attr_1iter
```

Scaling timing:

| Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms | Routed wgrad attr ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=16` | `8.7039` | `8.2836` | `17.0523` | `938.29` | `3.992056` | `4.5274` | `6.3663` |
| `B=1 S=32` | `13.1050` | `14.1494` | `27.3169` | `1171.44` | `5.002417` | `9.2327` | `12.3542` |
| `B=1 S=64` | `22.9561` | `25.7176` | `48.7374` | `1313.16` | `5.648932` | `18.5247` | `23.9564` |

Steady-state `B=1 S=64` PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-contiguous --batch 1 --seqlen 64 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s64_fp4_packed_sim_grouped_contiguous_sparse_wgrad_chunked_2iter
```

Steady-state `B=1 S=64` profiler result: `50.4244 ms` total, `1269.23 tokens/s`, `5.459933` estimated TFLOPs. Top CUDA entries are dominated by routed expert `bmm`: `aten::bmm` at `84.600 ms` across two iterations, `phase.forward` at `46.134 ms`, `forward.moe.routed_experts` at `37.011 ms`, and CK GEMM kernels at `36.629 ms`, `25.442 ms`, and `22.060 ms`. Copy/fill traffic stays low: `aten::copy_` `1.866 ms`, `aten::fill_` `0.229 ms`, and DtoD memcpy `0.143 ms`.

Current read after scaling: the sparse-wgrad ceiling path improves from `3.67` profiler TFLOPs at `S=16` to `5.46` profiler TFLOPs at the full contiguous `S=64` expert envelope. The remaining work is overwhelmingly grouped expert matvec/GEMM; attention, norms, residuals, fill, scatter, and copy traffic are now small in comparison. The next real performance step is a lower-level FP4 grouped expert kernel, not further PyTorch graph rearrangement.

### Grouped Expert-Batches Probe

`--moe-impl grouped-expert-batches` adds a PyTorch custom-autograd path for fixed-router packed-sim shapes where every touched expert has the same positive assignment count. Instead of treating each routed assignment independently, it sorts assignments by expert, forms `x_group [experts, count, hidden]`, runs expert-batched projections, applies SwiGLU, runs the expert-batched down projection, then unsorts and reduces the top-k outputs back to tokens. It falls back to the generic grouped path when the assignment counts are not balanced.

This is still not a real packed FP4 grouped GEMM kernel, but it is closer to the intended kernel contract than the contiguous prefix canary: one batch item per expert with a small local token count, compact expert-local wgrad products, and no global dense expert-table gradient fill in the hot path.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_match --json
```

Focused correctness result: pass. Expert-batch grouped MoE matched the loop reference with output max abs error `0.0`; max watched gradient relative L2 was `8.8929235628238e-17`.

Grouped expert-batches split-attribution commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 64 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s64_fp4_packed_sim_grouped_expert_batches_split_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 128 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s128_fp4_packed_sim_grouped_expert_batches_split_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 256 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s256_fp4_packed_sim_grouped_expert_batches_split_attr_1iter
```

Grouped expert-batches timing:

| Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms | Routed wgrad attr ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=64` | `12.8066` | `24.5923` | `37.4760` | `1707.76` | `7.346417` | `8.8305` | `21.2284` |
| `B=1 S=128` | `12.9612` | `25.0189` | `38.0558` | `3363.48` | `14.680583` | `8.9566` | `22.3480` |
| `B=1 S=256` | `13.1092` | `25.9220` | `39.1025` | `6546.90` | `28.745140` | `8.8129` | `21.7199` |

The best non-attribution `B=1 S=256` no-trace run used two measured iterations:

- Forward phase: `12.9298 ms`
- Backward phase: `25.6346 ms`
- End-to-end phase total: `38.6347 ms`
- Tokens/s: `6626.18`
- Estimated TFLOPs: `29.093216`
- Forward `moe.routed_experts`: `8.6798 ms`

Larger expert-batch scaling commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 512 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s512_fp4_packed_sim_grouped_expert_batches_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s4096_fp4_packed_sim_grouped_expert_batches_1iter
```

Larger expert-batch scaling result:

| Shape | Iters | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=512` | 2 | `14.3278` | `27.8318` | `42.2378` | `12121.84` | `53.851949` | `1.4187` | `8.8200` |
| `B=1 S=1024` | 2 | `18.9741` | `35.8928` | `54.9366` | `18639.68` | `84.742737` | `5.2867` | `8.9439` |
| `B=1 S=2048` | 2 | `35.9577` | `62.6480` | `98.6942` | `20750.97` | `98.649735` | `20.4037` | `9.3392` |
| `B=1 S=4096` | 1 | `91.5726` | `160.2154` | `251.9298` | `16258.50` | `84.043716` | `71.3725` | `10.5250` |

Current scaling read: the expert-batch MoE path keeps amortizing through `S=2048`, where each expert has 32 routed assignments. Past that point the sparse-MLA attention path becomes dominant: `attn.sparse_mla` grows from `20.4 ms` at `S=2048` to `71.4 ms` at `S=4096`, and the end-to-end TFLOPs drops despite more tokens.

Split attribution for the larger expert-batch runs:

| Block | B1S128 split ms | B1S256 split ms | B1S2048 split ms |
| --- | ---: | ---: | ---: |
| `moe.routed.dgrad` | `169.1288` | `160.9795` | `157.8123` |
| `moe.routed.wgrad` | `22.3480` | `21.7199` | `23.8506` |
| `moe.shared.dgrad` | `0.4172` | `0.4530` | `1.0849` |
| `moe.shared.wgrad` | `0.3504` | `0.3777` | `0.9118` |
| `attention.dgrad` | `2.5723` | `2.7957` | `36.6727` |
| `attention.wgrad` | `2.5196` | `2.5107` | `37.0502` |

The first `B=1 S=2048` split-attribution attempt was contaminated by a concurrent torch-profiler run and OOMed while materializing routed expert wgrad (`15.75 GiB` allocation). It was discarded. The retained `full_b1s2048...split_attr_1iter_retry` run was rerun alone and passed; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was ignored by this ROCm build as unsupported, so the successful result does not rely on that allocator knob.

Steady-state `B=1 S=128` PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 128 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s128_fp4_packed_sim_grouped_expert_batches_2iter
```

Steady-state `B=1 S=128` profiler result: `39.1416 ms` total, `3270.18 tokens/s`, `14.273328` estimated TFLOPs. Top CUDA entries are dominated by routed expert `bmm`: `aten::bmm` at `60.728 ms` across two iterations and 36 calls, `phase.forward` at `26.839 ms`, `forward.moe.routed_experts` at `17.758 ms`, and CK GEMM kernels at `26.990 ms`, `11.009 ms`, `10.817 ms`, `5.694 ms`, and `5.684 ms`. Non-matmul traffic is now much smaller: `aten::copy_` `2.188 ms`, `aten::mul` `1.345 ms`, and DtoD memcpy `0.180 ms`.

Steady-state `B=1 S=256` PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 256 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s256_fp4_packed_sim_grouped_expert_batches_2iter
```

Steady-state `B=1 S=256` profiler result: `40.3344 ms` total, `6346.95 tokens/s`, `27.867221` estimated TFLOPs. The ranked work remains dominated by the expert-batched matmuls: `aten::bmm` at `61.613 ms` across two iterations and 36 calls, `phase.forward` at `27.689 ms`, `forward.moe.routed_experts` at `17.813 ms`, and CK GEMM kernels at `26.650 ms`, `11.077 ms`, `10.831 ms`, `5.701 ms`, and `5.678 ms`. Non-matmul traffic stays secondary: `aten::copy_` `2.580 ms`, `aten::mul` `1.657 ms`, DtoD memcpy `0.223 ms`, and `aten::fill_` `0.268 ms`.

Steady-state `B=1 S=2048` PyTorch profiler command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --trace-dir profiles/torch_full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter_clean
```

Steady-state `B=1 S=2048` profiler result: `100.5697 ms` total, `20363.98 tokens/s`, `96.809994` estimated TFLOPs. The top CUDA rows are no longer purely MoE: `aten::bmm` is `132.224 ms` across two iterations and 36 calls, `phase.forward` is `73.439 ms`, `forward.attn.sparse_mla` is `41.294 ms`, `forward.moe.routed_experts` is `18.816 ms`, `aten::copy_` is `15.843 ms`, `aten::mul` is `11.753 ms`, `aten::cat` is `6.433 ms`, softmax forward/backward are about `4.4 ms` and `4.2 ms`, and DtoD memcpy is `2.858 ms`.

Whole-process `B=1 S=128` rocprofv3 command:

```bash
/opt/rocm/bin/rocprofv3 --runtime-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_full_b1s128_fp4_packed_sim_grouped_expert_batches_2iter/summary.txt --output-directory profiles/rocprof_full_b1s128_fp4_packed_sim_grouped_expert_batches_2iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 128 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/rocprof_full_b1s128_fp4_packed_sim_grouped_expert_batches_2iter/torch_summary
```

The rocprof-wrapped phase summary was `44.6303 ms` total, `2868.01 tokens/s`, and `12.517976` estimated TFLOPs. The kernel stats are intentionally retained as raw evidence, but like the earlier rocprof runs they include Python process setup, warmup, and packed-sim cache materialization. The top entry was the FP4 prepack `ArgMin` reduction at `1.7358 s` total (`58.28%`) before the steady-state GEMM kernels. An attach-mode attempt with `--ready-file`, `--trigger-file`, and `--sleep-after-iters-s` made the target process controllable, but this rocprofv3 attach build reported attach success without writing trace files, so the retained rocprof artifact is the whole-process wrapper.

Whole-process `B=1 S=2048` rocprofv3 command:

```bash
/opt/rocm/bin/rocprofv3 --runtime-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter/summary.txt --output-directory profiles/rocprof_full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/rocprof_full_b1s2048_fp4_packed_sim_grouped_expert_batches_2iter/torch_summary
```

The S2048 rocprof-wrapped phase summary was `104.4967 ms` total, `19598.70 tokens/s`, and `93.171871` estimated TFLOPs. As with S128, the whole-process kernel stats are dominated by packed-sim FP4 prepack: top `ArgMin` reduction is `1.7352 s` (`54.73%`). The retained trace is still useful as raw kernel evidence; the steady-state work should be read primarily from the torch-profiler run above.

Current read after expert-batching: `grouped-expert-batches` improves the full `B=1 S=64` no-trace path from `48.7374 ms`, `1313.16 tokens/s`, and `5.648932` TFLOPs in the sparse-wgrad contiguous canary to `37.4760 ms`, `1707.76 tokens/s`, and `7.346417` TFLOPs. Before route metadata caching, the strongest no-trace point was `B=1 S=2048` at `98.65` estimated TFLOPs and `20750.97 tokens/s`. The routed expert forward remains near `9 ms` from `S=128` through `S=2048`, so the next kernel target for MoE is still a real grouped FP4 expert GEMM; however, the larger training-shaped block also exposes sparse MLA as the next major system limiter.

### Grouped Expert-Batches Route Cache

The grouped expert-batch path now caches fixed-router assignment metadata keyed by the `top_indices` tensor owner and version. This avoids rebuilding `assignment_tokens`, `argsort`, `bincount`, `inverse_order`, and `token_sorted` on every timed iteration. The cache is intentionally tied to the tensor owner/version so dynamic router outputs still rebuild when their storage changes.

Route-cache timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_route_cache_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_route_cache_2iter
```

Route-cache result compared with the previous no-cache expert-batch baseline:

| Shape | Variant | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=1024` | no cache | `18.9741` | `35.8928` | `54.9366` | `18639.68` | `84.742737` | `5.2867` | `8.9439` |
| `B=1 S=1024` | route cache | `19.2893` | `35.3217` | `54.6834` | `18725.99` | `85.135145` | `5.2182` | `8.7590` |
| `B=1 S=2048` | no cache | `35.9577` | `62.6480` | `98.6942` | `20750.97` | `98.649735` | `20.4037` | `9.3392` |
| `B=1 S=2048` | route cache | `35.3724` | `62.6254` | `98.0832` | `20880.23` | `99.264232` | `20.2535` | `9.0287` |

Read: route caching is a modest but positive fixed-router cleanup. The new best no-trace point is `B=1 S=2048` at `99.26` estimated TFLOPs and `20880.23 tokens/s`, with the same caveat that this is still PyTorch packed-sim rather than a real packed FP4 grouped GEMM.

### Sparse MLA Metadata Cache

The sparse MLA reference path now caches static positions, compressed-key positions, and the invalid attention mask for each `(device, seqlen, window, CSA ratio, HCA ratio)` tuple. This removes repeated `arange`, `cat`, key-kind construction, causal/window mask construction, and mask inversion from steady-state fixed-shape layer iterations. The profiling recompute path uses the same cached metadata so attention sub-block attribution stays aligned with normal forward timing.

Sparse MLA metadata-cache timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_attn_meta_cache_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_attn_meta_cache_2iter
```

Sparse MLA metadata-cache result compared with the previous route-cache baseline:

| Shape | Variant | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=1024` | route cache | `19.2893` | `35.3217` | `54.6834` | `18725.99` | `85.135145` | `5.2182` | `8.7590` |
| `B=1 S=1024` | attention metadata cache | `18.5395` | `35.7577` | `54.3685` | `18834.44` | `85.628208` | `5.3192` | `8.7213` |
| `B=1 S=2048` | route cache | `35.3724` | `62.6254` | `98.0832` | `20880.23` | `99.264232` | `20.2535` | `9.0287` |
| `B=1 S=2048` | attention metadata cache | `35.1966` | `62.6564` | `97.9413` | `20910.48` | `99.408031` | `20.2537` | `9.0932` |

Read: sparse MLA metadata caching is a small positive harness cleanup rather than a real sparse MLA kernel improvement. It mainly trims static metadata work around the attention path; the measured `attn.sparse_mla` core remains flat. The new best no-trace point is `B=1 S=2048` at `99.41` estimated TFLOPs and `20910.48 tokens/s`.

### RoPE Table Cache

The RoPE path now caches fixed-shape cosine/sine tables by position tensor owner/version, RoPE dimension, device, and accumulation dtype. The attention metadata path also passes cached CSA/HCA group-end position tensors into the compressors so compressed RoPE calls can hit the same table cache across steady-state iterations. This removes repeated `inv_freq`, angle, `cos`, and `sin` construction from q, kv, compressed CSA/HCA, and inverse output RoPE calls.

RoPE-cache timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_rope_cache_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rope_cache_2iter
```

RoPE-cache result compared with the previous attention metadata-cache baseline:

| Shape | Variant | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=1024` | attention metadata cache | `18.5395` | `35.7577` | `54.3685` | `18834.44` | `85.628208` | `5.3192` | `8.7213` |
| `B=1 S=1024` | RoPE table cache | `17.9817` | `35.4500` | `53.4995` | `19140.35` | `87.018996` | `5.2243` | `8.7290` |
| `B=1 S=2048` | attention metadata cache | `35.1966` | `62.6564` | `97.9413` | `20910.48` | `99.408031` | `20.2537` | `9.0932` |
| `B=1 S=2048` | RoPE table cache | `34.5017` | `62.5707` | `97.1591` | `21078.82` | `100.208341` | `20.1809` | `9.0237` |

Read: RoPE table caching is still a harness/reference cleanup, but it removes enough repeated table construction around attention to push the best no-trace S2048 point over `100` estimated TFLOPs. The new best no-trace point is `B=1 S=2048` at `100.21` estimated TFLOPs and `21078.82 tokens/s`.

### Custom RMSNorm Backward Boundary

`RMSNorm` now uses a local custom autograd function that saves the forward accumulation input, weight, and reciprocal RMS statistic, then computes dgrad and wgrad explicitly in backward. This creates the boundary needed for future RMSNorm statistic-passing/fusion work across the attention-to-MoE transition. It is still implemented with PyTorch tensor ops, so it is not a final fused RMSNorm kernel.

Custom RMSNorm timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_2iter
```

Custom RMSNorm result compared with the previous RoPE-cache baseline:

| Shape | Variant | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=1024` | RoPE table cache | `17.9817` | `35.4500` | `53.4995` | `19140.35` | `87.018996` | `5.2243` | `8.7290` |
| `B=1 S=1024` | custom RMSNorm | `17.6924` | `35.4856` | `53.2464` | `19231.35` | `87.432710` | `5.1186` | `8.7064` |
| `B=1 S=2048` | RoPE table cache | `34.5017` | `62.5707` | `97.1591` | `21078.82` | `100.208341` | `20.1809` | `9.0237` |
| `B=1 S=2048` | custom RMSNorm | `34.6903` | `62.2914` | `97.0731` | `21097.51` | `100.297167` | `20.1283` | `9.0405` |

Major-block attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_rmsnorm_custom_attr_1iter
```

Custom RMSNorm attribution at `B=1 S=2048`:

| Block | Time ms |
| --- | ---: |
| `layer.attn_norm.dgrad` | `0.4134` |
| `layer.attn_norm.wgrad` | `0.3528` |
| `layer.moe_norm.dgrad` | `0.3502` |
| `layer.moe_norm.wgrad` | `0.3409` |

Read: custom RMSNorm is a small positive for the full normal `loss.backward()` path, but the segmented norm-only attribution does not make it look like a strong standalone optimization. Keep it mainly as a useful boundary for later statistic-passing or fused RMSNorm-backward work. The new best no-trace point is `B=1 S=2048` at `100.30` estimated TFLOPs and `21097.51 tokens/s`.

## Attention Backward Sub-Block Attribution

The attention sub-block attribution path was added after the S2048 scaling run. It recomputes the attention module from the saved normalized attention input, drives it with the real upstream `d_attn_out`, then verifies that the sum of q/kv/CSA/HCA input gradients matches the top-level attention dgrad from the full layer attribution.

Smoke correctness command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset gradcheck --impl reference --batch 1 --seqlen 4 --dtype float32 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-attention-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/smoke_attn_attr_gradcheck_s4_diff
```

Smoke result: pass. The recomposed attention input gradient matched the top-level attention dgrad with max abs diff `2.91e-11` and relative L2 diff `7.77e-08`.

Full-shape attribution commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-attention-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_attention_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-attention-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_attention_attr_1iter
```

Normal phase timing from the same runs:

| Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=1024` | `19.5344` | `35.4932` | `55.1025` | `18583.57` | `84.487641` | `5.3671` | `9.0489` |
| `B=1 S=2048` | `36.0144` | `62.5293` | `98.6293` | `20764.62` | `98.714597` | `20.2956` | `9.3658` |

Attention sub-block backward attribution:

| Block | B1S1024 ms | B1S2048 ms |
| --- | ---: | ---: |
| `attention.dgrad` | `11.6109` | `36.7210` |
| `attention.wgrad` | `11.8101` | `37.3520` |
| `attention.sparse_mla.dgrad_qkv` | `8.0235` | `30.4822` |
| `attention.sparse_mla.sink_wgrad` | `2.8983` | `11.1490` |
| `attention.q_path.dgrad` | `2.1457` | `3.7996` |
| `attention.q_path.wgrad` | `2.1468` | `3.7466` |
| `attention.out_proj.dgrad` | `0.8216` | `1.4893` |
| `attention.out_proj.wgrad` | `0.7004` | `0.8982` |
| `attention.output_rope_inv.dgrad` | `0.4283` | `0.7282` |
| `attention.kv_path.dgrad` | `0.4808` | `0.5779` |
| `attention.kv_path.wgrad` | `0.4639` | `0.4695` |
| `attention.csa.dgrad` | `0.5954` | `0.6795` |
| `attention.csa.wgrad` | `0.5914` | `0.6141` |
| `attention.hca.dgrad` | `0.5944` | `0.5894` |
| `attention.hca.wgrad` | `0.6001` | `0.5806` |

Full-shape gradient split check:

| Shape | Top-level attention dgrad norm | Recomposed sub-block dgrad norm | Max abs diff | Rel L2 diff |
| --- | ---: | ---: | ---: | ---: |
| `B=1 S=1024` | `5.037139e-05` | `5.037140e-05` | `3.73e-09` | `3.82e-03` |
| `B=1 S=2048` | `3.042020e-05` | `3.042012e-05` | `9.31e-10` | `3.87e-03` |

Read: at the S2048 best-throughput point, attention backward is almost entirely the sparse MLA softmax/einsum region. The q path is the next material attention backward component; kv projection, CSA, HCA, output projection, and inverse RoPE are all sub-2 ms in this PyTorch attribution mode. This makes sparse MLA backward the first attention kernel target after the grouped FP4 expert GEMM path.

## MoE Backward Sub-Block Attribution

The MoE sub-block attribution path was added to avoid over-reading the coarse `moe.routed.dgrad` number for the custom grouped expert-batch autograd function. The coarse custom-function dgrad call still sees all routed expert weights as requiring grad, so it is useful as a stress signal but not a clean dgrad-only number. The new path recomputes the same grouped expert-batch math with standard PyTorch edges and the real upstream `d_moe_out`, then checks that the recomposed routed MoE input gradient matches the top-level routed MoE dgrad.

MoE attribution commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 64 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-moe-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s64_fp4_packed_sim_grouped_expert_batches_moe_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-moe-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_moe_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-moe-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_moe_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-moe-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_moe_shared_attr_1iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --profile-backward-blocks --profile-moe-blocks --backward-attribution-iters 1 --backward-attribution-warmup 0 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_moe_shared_attr_1iter
```

Normal phase timing from the same runs:

| Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms | Forward `attn.sparse_mla` ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=64` | `12.7472` | `24.6429` | `37.4672` | `1708.16` | `7.348132` | `8.7679` | `0.4049` |
| `B=1 S=1024` | `19.1297` | `35.5482` | `54.7584` | `18700.32` | `85.018434` | `9.0968` | `5.2784` |
| `B=1 S=2048` | `36.1573` | `62.5899` | `98.8346` | `20721.49` | `98.509589` | `9.3985` | `20.4474` |
| `B=1 S=2048` shared split rerun | `35.9530` | `62.6704` | `98.7088` | `20747.90` | `98.635121` | `9.3334` | `20.3679` |

MoE routed expert-batch sub-block backward attribution:

| Block | B1S64 ms | B1S1024 ms | B1S2048 ms |
| --- | ---: | ---: | ---: |
| `moe.routed.output_scale_reduce.dgrad` | `0.2804` | `0.5480` | `1.0413` |
| `moe.routed.w2.dgrad` | `2.8626` | `2.9036` | `3.0272` |
| `moe.routed.w2.wgrad` | `4.5336` | `4.6667` | `4.8504` |
| `moe.routed.swiglu.dgrad` | `0.1093` | `0.1681` | `0.3069` |
| `moe.routed.w1_w3.dgrad` | `5.6187` | `5.8605` | `6.2422` |
| `moe.routed.w1.wgrad` | `4.2438` | `4.4998` | `4.7546` |
| `moe.routed.w3.wgrad` | `4.1633` | `4.1383` | `4.5767` |
| `moe.routed.input_gather_reduce.dgrad` | `0.1545` | `0.2405` | `0.4555` |

MoE gradient split check:

| Shape | Top-level routed MoE dgrad norm | Recomposed sub-block dgrad norm | Max abs diff | Rel L2 diff |
| --- | ---: | ---: | ---: | ---: |
| `B=1 S=64` | `8.221478e-05` | `8.221391e-05` | `3.73e-09` | `3.55e-03` |
| `B=1 S=1024` | `2.046650e-05` | `2.046629e-05` | `2.33e-10` | `3.55e-03` |
| `B=1 S=2048` | `1.446416e-05` | `1.446401e-05` | `2.33e-10` | `3.55e-03` |

Shared expert sub-block backward attribution from the shared split reruns:

| Block | B1S1024 ms | B1S2048 ms |
| --- | ---: | ---: |
| `moe.shared.dgrad` | `0.6651` | `1.0969` |
| `moe.shared.wgrad` | `0.5975` | `0.9078` |
| `moe.shared.w2.dgrad` | `0.2380` | `0.3488` |
| `moe.shared.w2.wgrad` | `0.2021` | `0.2963` |
| `moe.shared.swiglu.dgrad` | `0.1005` | `0.1323` |
| `moe.shared.w1_w3.dgrad` | `0.3107` | `0.4980` |
| `moe.shared.w1.wgrad` | `0.1607` | `0.2575` |
| `moe.shared.w3.wgrad` | `0.1543` | `0.2545` |

Shared expert gradient split check:

| Shape | Top-level shared MoE dgrad norm | Recomposed sub-block dgrad norm | Max abs diff | Rel L2 diff |
| --- | ---: | ---: | ---: | ---: |
| `B=1 S=1024` | `5.090743e-05` | `5.090743e-05` | `0.0` | `0.0` |
| `B=1 S=2048` | `3.597360e-05` | `3.597360e-05` | `0.0` | `0.0` |

Read: for equal-count grouped expert batches, MoE routed expert cost is dominated by the three expert-batched projection backward edges. W1/W3 dgrad is the largest single clean sub-block, followed by W2 wgrad and W1/W3 wgrads. Output scaling/reduce, SwiGLU dgrad, input gather/reduce, and all shared-expert internals are small relative to the routed expert matmuls. This keeps the MoE optimization target focused on a real grouped FP4 expert GEMM covering routed W1, W3, and W2 forward/backward, rather than routing traffic, shared expert, or elementwise SwiGLU first.

## Fused W1/W3 Expert-Batch Negative Control

`--moe-impl grouped-expert-batches-fused-w13` fuses the routed expert W1 and W3 projections into one packed-sim table shaped `[experts, 2 * intermediate, hidden]` and one batched matmul, then splits the gate/up halves before SwiGLU. This preserves the same fixed-router equal-count contract as `grouped-expert-batches`.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_fused_w13_match --json
```

Focused correctness result: pass. Fused W1/W3 expert batches matched the loop reference with output max abs error `0.0`; max watched gradient relative L2 was `8.8929235628238e-17`.

Fused W1/W3 timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-fused-w13 --batch 1 --seqlen 64 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s64_fp4_packed_sim_grouped_expert_batches_fused_w13_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-fused-w13 --batch 1 --seqlen 1024 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s1024_fp4_packed_sim_grouped_expert_batches_fused_w13_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-fused-w13 --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_fused_w13_2iter
```

Timing result compared with the current non-fused expert-batch baseline:

| Shape | Impl | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=1 S=64` | non-fused | `12.8066` | `24.5923` | `37.4760` | `1707.76` | `7.346417` | `8.8305` |
| `B=1 S=64` | fused W1/W3 | `12.9174` | `38.4887` | `51.4849` | `1243.08` | `5.347475` | `8.7485` |
| `B=1 S=1024` | non-fused | `18.9741` | `35.8928` | `54.9366` | `18639.68` | `84.742737` | `8.9439` |
| `B=1 S=1024` | fused W1/W3 | `18.7407` | `50.0138` | `68.8228` | `14878.80` | `67.644426` | `8.9930` |
| `B=1 S=2048` | non-fused | `35.9577` | `62.6480` | `98.6942` | `20750.97` | `98.649735` | `9.3392` |
| `B=1 S=2048` | fused W1/W3 | `35.6230` | `77.5000` | `113.2118` | `18089.99` | `85.999460` | `9.3213` |

Read: this fused W1/W3 PyTorch packed-sim variant is a negative control. It slightly reduces or preserves routed expert forward time, but backward regresses at every measured shape. The likely issue is that the larger fused W13 table and combined backward matmul lose more than they save in launch count for this PyTorch/CK path. Keep the non-fused grouped expert-batch path as the current PyTorch ceiling until a real grouped FP4 expert kernel exists.

## Fused Shared Expert Negative Control

`--shared-expert-impl fused-w13` fuses the shared expert W1 and W3 projections into one packed-sim table shaped `[2 * shared_hidden, hidden]`, then applies an explicit SwiGLU backward before the shared W2 projection. This leaves the routed expert implementation independently selectable with `--moe-impl`.

Focused correctness commands:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks attention,moe,layer,fp4_packed_sim_match,moe_shared_fused_w13_match --rmsnorm-save-policy memory-light --json
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_shared_fused_w13_match --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass. The fused shared expert matched the default shared expert in the full layer with output max abs error `0.0`; CPU max watched gradient relative L2 was `2.3950786080640395e-16`, CUDA max watched gradient relative L2 was `1.6191415080372311e-16`.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_shared_linear_pair_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl fused-w13 --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_shared_fused_w13_2iter
```

Timing result:

| Shared impl | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.shared_expert` ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear-pair` | `34.8129` | `62.6158` | `97.5160` | `21001.68` | `99.841581` | `0.5291` |
| `fused-w13` | `34.9371` | `62.7792` | `97.8042` | `20939.80` | `99.547441` | `0.4656` |

One-iteration segmented backward attribution:

| Shared impl | Forward `moe.shared_expert` ms | `moe.shared.dgrad` ms | `moe.shared.wgrad` ms | `moe.routed.dgrad` ms | `moe.routed.wgrad` ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `linear-pair` | `0.5239` | `1.0822` | `0.8823` | `23.6753` | `23.7463` |
| `fused-w13` | `0.5013` | `1.0828` | `0.8786` | `24.0831` | `24.0373` |

Read: fused shared W1/W3 is a small forward-block win, but it does not improve full-layer timing in the current PyTorch packed-sim baseline. The shared expert remains too small relative to routed expert W1/W3/W2 backward to prioritize ahead of a real grouped FP4 routed expert GEMM.

## Grouped Expert-Batch Backward Grad-Output Gather Cleanup

The grouped expert-batch packed-sim autograd path now saves the sorted token ids from the route cache and gathers `grad_output` directly as `grad_output.index_select(0, token_sorted)`. The previous path expanded the upstream gradient to `[tokens * top_k, hidden]` and then selected by sorted assignment order. This is the same mathematical contract, but it avoids sorting through a larger logical source tensor during routed MoE backward.

Focused correctness commands:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_match,moe_expert_batches_fused_w13_match,fp4_packed_sim_match --rmsnorm-save-policy memory-light --json
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_match,moe_expert_batches_fused_w13_match,fp4_packed_sim_match --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass. CUDA grouped expert-batch output max abs error `0.0`; max watched gradient relative L2 `8.8929235628238e-17`. CUDA grouped expert-batch fused W1/W3 output max abs error `0.0`; max watched gradient relative L2 `8.123493959409061e-17`. Packed-sim still matched fake FP4 exactly.

S2048 no-trace timing command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_gradout_sorted_token_2iter
```

Timing result: `97.5118 ms`, `21002.59 tokens/s`, `99.845903 TFLOPs`; forward routed MoE block `9.3176 ms`, forward shared MoE block `0.4905 ms`.

One-iteration segmented backward attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_gradout_sorted_token_attr_1iter
```

Attribution result: normal phase total `98.2506 ms`, `20844.65 tokens/s`, `99.095084 TFLOPs`; routed MoE dgrad `23.6986 ms`, routed MoE wgrad `23.6255 ms`, shared MoE dgrad `1.0061 ms`, shared MoE wgrad `0.8962 ms`, attention dgrad `36.3180 ms`, attention wgrad `37.0354 ms`.

Read: this cleanup is correctness-safe and lands roughly neutral to slightly positive on whole-layer timing compared with adjacent S2048 packed-sim runs. It is a sensible data-movement cleanup for the current PyTorch autograd ceiling, but not a substitute for the required real grouped FP4 routed expert GEMM.

Focused grouped expert-batch route-precheck cleanup:

- Change: remove the duplicate `torch.bincount`/Python `bool` equal-count precheck from the `RoutedMoERef` grouped expert-batch wrappers. The packed-sim route builder now validates and caches the route before packed weight materialization; the wrapper falls back to token-sum only when the route builder rejects an unbalanced layout.
- CUDA correctness command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_match,moe_expert_batches_fused_w13_match,moe_expert_batches_unbalanced_fallback_match,fp4_packed_sim_match --rmsnorm-save-policy memory-light --json`
- CUDA correctness result: pass. Balanced grouped expert-batch max output error `0.0`, max watched gradient relative L2 `8.8929235628238e-17`; fused W13 max output error `0.0`, max watched gradient relative L2 `8.123493959409061e-17`; unbalanced fallback against grouped-token-sum max output error `0.0`, max watched gradient relative L2 `0.0`; packed-sim still matched fake FP4 exactly.
- S2048 no-trace command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_route_precheck_removed_2iter`
- S2048 result: `97.2368 ms`, `21061.98 tokens/s`, `100.128266 TFLOPs`; phase split forward `34.3500 ms`, backward `62.8018 ms`, loss `0.0850 ms`. Forward routed MoE block `8.9245 ms`, shared MoE block `0.4939 ms`, sparse MLA block `20.1379 ms`.
- Read: removing the wrapper precheck trims one fixed-router synchronization/recount from every grouped expert-batch forward. The whole-layer number is slightly better than the adjacent grad-output cleanup run (`97.5118 ms`, `99.845903 TFLOPs`), while still a PyTorch packed-sim ceiling path rather than a real packed FP4 GEMM.

## FP4 Packed Storage Contract

The FP4 helper layer now exposes an explicit byte-level storage oracle for future grouped expert GEMM kernels:

- E2M1 values are stored as sign+magnitude nibbles, two FP4 values per `uint8`.
- UE8M0 block scales are stored as one biased exponent byte per 2D block.
- `pack_fp4_block_scaled_weight` returns packed value bytes, scale bytes, original shape, blocked storage shape, and logical value count.
- `dequantize_packed_fp4_block_scaled_weight` reconstructs the exact same dequantized tensor as the existing FP4 fake-quant oracle.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cpu --dtype float64 --checks fp4_packed_storage_roundtrip --json
```

Focused correctness result: pass in the ROCm container on `do-sonle-kernel`. The gate covered an odd-nibble case, a rank-3 expert-weight case, and a block-aligned case. All cases had exact nibble code/sign roundtrip, exact packed byte unpack, exact UE8M0 scale decode, and dequantized output max abs error `0.0` versus the existing FP4 reference.

| Case | Input shape | Blocked shape | FP4 values | Packed value bytes | Scale bytes |
| --- | --- | --- | ---: | ---: | ---: |
| `odd_nibble_count` | `[1, 1]` | `[1, 1, 3, 3]` | `9` | `5` | `1` |
| `rank3_expert_weight` | `[5, 3, 5]` | `[5, 2, 2, 2, 3]` | `120` | `60` | `20` |
| `block_aligned` | `[4, 4]` | `[1, 1, 4, 4]` | `16` | `8` | `1` |

CUDA correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --checks fp4_packed_storage_roundtrip --json
```

CUDA correctness result: pass on `do-sonle-kernel` once the external CODA/TileLang profiling job cleared. The same three storage cases had exact nibble code/sign roundtrip, exact packed byte unpack, exact UE8M0 scale decode, and dequantized output max abs error `0.0` versus the existing FP4 reference.

Focused profile-summary smoke command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset gradcheck --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 4 --dtype float32 --device cpu --warmup 0 --iters 1 --fixed-router --no-torch-trace --trace-dir profiles/traffic_fp4_storage_cpu_smoke
```

Profile smoke result: pass. `summary.json` now includes `traffic_estimate.packed_fp4_weight_storage_bytes`, split into attention linear, MoE router, routed MoE, shared MoE, current FP4 path totals, and per-weight records.

Focused backward shape-contract smoke command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset gradcheck --impl fp4-packed-sim --moe-impl grouped-expert-batches --batch 1 --seqlen 4 --dtype float32 --device cpu --warmup 0 --iters 1 --fixed-router --profile-backward-blocks --profile-attention-blocks --profile-moe-blocks --backward-attribution-warmup 0 --backward-attribution-iters 1 --no-torch-trace --trace-dir profiles/backward_shape_contract_cpu_smoke_20260530
```

Shape-contract smoke result: pass. `summary.json` now includes `backward_shape_contract`, a single-GPU `[B, S, M]` contract for the top-level backward path plus attention output projection, sparse MLA, q/kv/compressed attention paths, grouped routed MoE, and shared expert subblocks. The gradcheck-sized smoke emitted `B=1`, `S=4`, `T=4`, `M=8`, `H=2`, `D=4`, `KV_ALL=7`, and balanced routed expert batches with `A_PER_E_BALANCED=2`.

Full-shape static packed-weight estimate for `B=1 S=2048`, full DSv4 dims, `128x128` blocks:

| Group | Logical elements | FP4 value bytes | UE8M0 scale bytes | Total bytes |
| --- | ---: | ---: | ---: | ---: |
| attention FP4 linear weights | `188,743,680` | `94,371,840` | `11,520` | `94,383,360` |
| MoE router | `2,752,512` | `1,376,256` | `168` | `1,376,424` |
| routed MoE W1/W2/W3 | `25,367,150,592` | `12,683,575,296` | `1,548,288` | `12,685,123,584` |
| shared MoE W1/W2/W3 | `66,060,288` | `33,030,144` | `4,032` | `33,034,176` |
| current FP4-covered path | `25,624,707,072` | `12,812,353,536` | `1,564,008` | `12,813,917,544` |

Read: routed MoE dominates packed weight storage just as it dominates FLOPs. Each routed expert matrix (`moe.w1`, `moe.w2`, `moe.w3`) is `4,228,374,528` bytes with `128x128` E2M1/UE8M0 storage. This reinforces that the next real kernel target is grouped routed FP4 GEMM over W1/W2/W3, not shared expert or attention-side bookkeeping.

## Sparse MLA Physical-Key Gather

The sparse MLA reference now has an explicit `--attention-impl sparse-gather` path. It gathers only the physically valid local-window plus CSA/HCA keys before score/value matmuls instead of scoring every vanilla token and masking most of them afterward. For `B=1 S=2048`, this changes the attention score axis from `KV_ALL=2576` to `ATTN_KEYS=656` (`window=128`, `CSA=512`, `HCA=16`) while still carrying `kv_all [1, 2576, 512]` for the combined vanilla/compressed gradient destination.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match,moe_expert_batches_match --attention-impl sparse-gather --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. Dense-mask and sparse-gather full-layer outputs matched exactly (`max_abs=0.0`), max watched gradient relative L2 was `1.7209425079949776e-16`, sparse attention/layer gradcheck passed, FP4 packed-sim matched fake FP4 exactly under `attention_impl=sparse-gather`, and grouped expert-batch MoE parity stayed green with max watched gradient relative L2 `1.1961225334218e-16`.

S2048 no-trace timing command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --attention-impl sparse-gather --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_2iter
```

Timing result:

| Attention impl | Score keys/query | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward attention ms | Forward `attn.sparse_mla` ms | Forward routed MoE ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense-mask` route-precheck baseline | `2576` physical score keys | `34.3500` | `62.8018` | `97.2368` | `21061.98` | `100.128266` | n/a | `20.1379` | `8.9245` |
| `sparse-gather` | `656` gathered sparse keys | `21.2038` | `49.6556` | `70.9470` | `28866.62` | `137.231365` | `10.3030` | `6.5783` | `9.0138` |

Sparse-gather static traffic for `B=1 S=2048`: `kv_selected [1, 2048, 656, 512]` is `1,375,731,712` bf16 bytes, scores `[1, 128, 2048, 657]` are `688,914,432` fp32 bytes, and probs `[1, 128, 2048, 656]` are `687,865,856` fp32 bytes. The materialized gather is large, but it still beats scoring the dense vanilla key axis in this PyTorch reference.

Read: this is a real reference-path correction, not the final sparse MLA kernel. It aligns the benchmark with the intended sparse physical-key contract and lifts the current PyTorch packed-sim ceiling from roughly `21.1k` to `28.9k tokens/s` at S2048. The next attention target is to replace the materialized `kv_selected` gather plus PyTorch einsums/softmax with a true sparse MLA forward/backward kernel that accumulates directly into `d_kv_all`.

Sparse-gather one-iteration segmented backward attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --attention-impl sparse-gather --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --profile-attention-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_attr_1iter
```

Attribution result: normal phase total `71.0979 ms`, `28805.35 tokens/s`, `136.940083 TFLOPs`; phase split forward `21.2441 ms`, backward `49.7611 ms`, loss `0.0927 ms`. Segmented backward blocks were nearly balanced: attention dgrad `23.7653 ms`, attention wgrad `24.0969 ms`, MoE dgrad `24.5118 ms`, MoE wgrad `24.5833 ms`. The largest attention subblock was `attention.sparse_mla.dgrad_qkv` at `17.6612 ms`; the largest routed MoE subblocks were `moe.routed.w1_w3.dgrad` at `6.0444 ms`, `moe.routed.w1.wgrad` at `4.9333 ms`, `moe.routed.w2.wgrad` at `4.8440 ms`, `moe.routed.w3.wgrad` at `4.3429 ms`, and `moe.routed.w2.dgrad` at `3.0992 ms`.

Read: after physical sparse-key gathering, the harness is no longer dominated by only routed MoE. The next real performance work should either (1) implement the true sparse MLA backward kernel that fuses score/prob/value backward and writes directly into `d_kv_all`, or (2) replace the PyTorch grouped expert-batch `bmm` path with a real grouped packed-FP4 GEMM. Small Python-level MoE gather/reduce cleanups are unlikely to move the phase total much.

## Grouped Expert-Batch Index-Add Negative Control

`--moe-impl grouped-expert-batches-index-add` is a swappable candidate that reduces expert-sorted routed outputs and routed input gradients with `index_add_` directly against token ids. This avoids restoring original assignment order and summing `[tokens, top_k, hidden]`, but keeps the same grouped expert-batch matmul sequence and FP4 packed-sim contract.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_index_add_match,fp4_packed_sim_match --attention-impl sparse-gather --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. Index-add grouped expert batches matched the loop reference with output max abs error `0.0`; max watched gradient relative L2 was `8.8929235628238e-17`. FP4 packed-sim still matched fake FP4 exactly under `attention_impl=sparse-gather`.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-index-add --shared-expert-impl linear-pair --attention-impl sparse-gather --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_index_add_sparse_gather_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --attention-impl sparse-gather --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_adjacent_baseline_2iter
```

Timing result:

| MoE impl | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms | Forward attention ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grouped-expert-batches-index-add` | `21.1357` | `49.5434` | `70.7662` | `28940.38` | `137.582035` | `9.2059` | `10.0763` |
| `grouped-expert-batches` adjacent baseline | `21.0335` | `49.3357` | `70.4526` | `29069.18` | `138.194334` | `9.0832` | `10.1055` |

Read: this is a negative control. Direct `index_add_` reduction is correctness-safe, but the adjacent baseline is faster overall and has a faster routed MoE forward block. Keep the default PyTorch ceiling path on `grouped-expert-batches`; do not spend more time on token-level scatter/reduce variants unless a lower-level kernel can fuse the scatter with expert GEMM output.

## Sparse MLA Manual Backward Surface

`--attention-impl sparse-gather-manual-bwd` is now a swappable sparse MLA path with an explicit custom autograd Function for the physical-key gather attention core. Forward still materializes `kv_selected [B, S, ATTN_KEYS, D]` and PyTorch einsums/softmax, but backward now spells out `dP`, softmax backward, `dQ`, gathered `dKV`, `d_attn_sink`, and the `index_add_` accumulation into `d_kv_all`. This is a kernel-under-test surface for a future fused sparse MLA backward kernel, not the final kernel itself.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. Dense-mask and manual sparse-gather full-layer outputs matched exactly (`max_abs=0.0`), max watched gradient relative L2 was `1.7209425079949776e-16`, sparse attention/layer gradcheck passed, and FP4 packed-sim matched fake FP4 exactly under `attention_impl=sparse-gather-manual-bwd`.

S2048 no-trace timing command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_manual_bwd_2iter
```

Timing result:

| Attention impl | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward attention ms | Forward `attn.sparse_mla` ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sparse-gather` adjacent baseline | `21.0335` | `49.3357` | `70.4526` | `29069.18` | `138.194334` | `10.1055` | `6.5136` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_adjacent_baseline_2iter/summary.json` |
| `sparse-gather-manual-bwd` | `21.8794` | `45.7066` | `67.6727` | `30263.29` | `143.871119` | `10.6857` | `6.6814` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_manual_bwd_2iter/summary.json` |

One-iteration segmented backward attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --profile-attention-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_manual_bwd_attr_1iter
```

Attribution result: normal phase total `67.4723 ms`, `30353.19 tokens/s`, `143.947793 TFLOPs`; phase split forward `21.6151 ms`, backward `45.7699 ms`, loss `0.0873 ms`. Segmented backward blocks: attention dgrad `19.6231 ms`, attention wgrad `19.9019 ms`, MoE dgrad `25.0827 ms`, MoE wgrad `24.6955 ms`. Attention subblocks: output projection dgrad/wgrad `1.2082/0.9683 ms`, q path dgrad/wgrad `3.7528/3.6761 ms`, `attention.sparse_mla.dgrad_qkv` `13.6413 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_manual_bwd_attr_1iter/summary.json`.

Read: the manual backward surface is correctness-safe and improves the normal whole-layer backward phase versus the adjacent autograd sparse-gather baseline. It is worth keeping as the next attention kernel contract because the explicit backward reduces the sparse MLA dgrad/qkv attribution from `17.6612 ms` to `13.6413 ms`. Caveat: isolated `attention.sparse_mla.sink_wgrad` attribution invokes the full custom Function backward, so it is not a true sink-only cost in this mode.

## Grouped Expert-Batch Dense WGrad Candidate

`--moe-impl grouped-expert-batches-dense-wgrad` is a swappable variant of the grouped expert-batch packed-sim MoE path. It keeps the same balanced expert route, three grouped `bmm` sequence, SwiGLU math, and routed output reduction, but returns dense `d_w1`, `d_w2`, and `d_w3` tensors when all 384 experts are touched instead of wrapping those full-row gradients in sparse COO tensors. This is closer to the full-gradient training contract and removes sparse-gradient wrapper work, but still uses PyTorch packed-sim dequantized weights rather than a real packed FP4 grouped GEMM.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_dense_wgrad_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. Dense-wgrad grouped expert batches matched the loop reference with output max abs error `0.0`; max watched gradient relative L2 was `8.8929235628238e-17`. FP4 packed-sim still matched fake FP4 exactly under `attention_impl=sparse-gather-manual-bwd`.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_manual_bwd_adjacent_2iter
```

Timing result:

| MoE impl | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `grouped-expert-batches` adjacent baseline | `21.1272` | `46.0692` | `67.2837` | `30438.27` | `144.702979` | `9.0199` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_sparse_gather_manual_bwd_adjacent_2iter/summary.json` |
| `grouped-expert-batches-dense-wgrad` | `21.0339` | `45.6176` | `66.7408` | `30685.89` | `145.880135` | `8.9029` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_2iter/summary.json` |

One-iteration segmented attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_attr_1iter
```

Attribution result: normal phase total `67.3382 ms`, `30413.62 tokens/s`, `143.989687 TFLOPs`; phase split forward `21.4840 ms`, backward `45.7566 ms`, loss `0.0976 ms`. Segmented backward blocks: attention dgrad `19.9780 ms`, attention wgrad `19.9145 ms`, MoE dgrad `24.6933 ms`, MoE wgrad `24.8924 ms`. Routed MoE subblocks: `moe.routed.w1_w3.dgrad` `6.3342 ms`, `moe.routed.w2.dgrad` `3.0537 ms`, `moe.routed.w1.wgrad` `4.9042 ms`, `moe.routed.w2.wgrad` `4.9027 ms`, `moe.routed.w3.wgrad` `4.4587 ms`, `moe.routed.output_scale_reduce.dgrad` `1.1021 ms`, `moe.routed.input_gather_reduce.dgrad` `0.4254 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_attr_1iter/summary.json`.

Read: dense wgrads are a small positive control, not the main kernel solution. The adjacent full-layer pair improves by `0.5429 ms` (`67.2837 -> 66.7408 ms`) and lifts throughput from `30.44k` to `30.69k tokens/s`. Keep this as the current best PyTorch packed-sim ceiling path, while the real performance target remains a grouped packed-FP4 routed expert GEMM that avoids dequantized `bmm` and writes the full expert gradients directly.

## Grouped Expert-Batch Fused W13 Dense WGrad Candidate

`--moe-impl grouped-expert-batches-fused-w13-dense-wgrad` combines two previously separate surfaces: the grouped expert-batch fused W1/W3 projection and dense full-row routed expert gradients. It keeps the same balanced fixed-router contract and returns dense `d_w1`, `d_w2`, and `d_w3` tensors when every expert is touched.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_fused_w13_dense_wgrad_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. The fused-W13 dense-wgrad path matched the loop reference with output max abs error `0.0`; max watched gradient relative L2 was `8.8929235628238e-17`. FP4 packed-sim still matched fake FP4 exactly under `attention_impl=sparse-gather-manual-bwd`.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-fused-w13-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_fused_w13_dense_wgrad_sparse_gather_manual_bwd_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_adjacent2_2iter
```

Timing result:

| MoE impl | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `grouped-expert-batches-dense-wgrad` adjacent baseline | `21.0905` | `45.7165` | `66.8982` | `30613.67` | `145.536821` | `9.0069` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_adjacent2_2iter/summary.json` |
| `grouped-expert-batches-fused-w13-dense-wgrad` | `20.7693` | `60.3069` | `81.1608` | `25233.86` | `119.961307` | `8.9195` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_fused_w13_dense_wgrad_sparse_gather_manual_bwd_2iter/summary.json` |

Read: this is a negative control. Fusing W1/W3 gives a tiny routed forward-block improvement (`9.0069 -> 8.9195 ms`), but the full backward regresses badly (`45.7165 -> 60.3069 ms`), so the total layer throughput drops from `30.61k` to `25.23k tokens/s`. Keep `grouped-expert-batches-dense-wgrad` as the general non-uniform PyTorch packed-sim baseline. A production fused W1/W3 path needs a real packed FP4 kernel that avoids the large concatenated PyTorch gradient materialization rather than merely concatenating the dequantized packed-sim weights.

## Grouped Expert-Batch Uniform Dense WGrad Candidate

`--moe-impl grouped-expert-batches-uniform-dense-wgrad` specializes the fixed-router benchmark case where routed scores are constant `1/top_k` and router gradients are intentionally disabled. It skips per-assignment score gather/save and the score-gradient path while keeping the same grouped expert-batch `bmm` sequence and dense routed expert wgrad contract. If `top_scores.requires_grad` is true, the harness falls back to the general grouped path instead of pretending this is a learned-router implementation.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_uniform_dense_wgrad_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. The uniform dense-wgrad path matched the loop reference with output max abs error `0.0`, output relative L2 `0.0`, and max watched gradient relative L2 `8.8929235628238e-17`. FP4 packed-sim still matched fake FP4 exactly under `attention_impl=sparse-gather-manual-bwd`.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_uniform_dense_wgrad_sparse_gather_manual_bwd_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_adjacent3_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_uniform_dense_wgrad_sparse_gather_manual_bwd_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_adjacent3_5iter
```

Timing result:

| MoE impl | Iters | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `moe.routed_experts` ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `grouped-expert-batches-dense-wgrad` adjacent baseline | `2` | `21.0944` | `45.4809` | `66.6599` | `30723.13` | `146.057188` | `9.0454` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_adjacent3_2iter/summary.json` |
| `grouped-expert-batches-uniform-dense-wgrad` | `2` | `20.9915` | `45.5454` | `66.6218` | `30740.71` | `146.140750` | `9.0373` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_uniform_dense_wgrad_sparse_gather_manual_bwd_2iter/summary.json` |
| `grouped-expert-batches-dense-wgrad` adjacent baseline | `5` | `20.7186` | `45.6262` | `66.4300` | `30829.46` | `146.562701` | `8.9841` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_dense_wgrad_sparse_gather_manual_bwd_adjacent3_5iter/summary.json` |
| `grouped-expert-batches-uniform-dense-wgrad` | `5` | `20.5557` | `45.5391` | `66.1793` | `30946.25` | `147.117907` | `8.8448` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_uniform_dense_wgrad_sparse_gather_manual_bwd_5iter/summary.json` |

Read: this is a small positive fixed-router harness specialization. The 5-iteration adjacent pair improves total time by `0.2507 ms` (`66.4300 -> 66.1793 ms`) and lifts throughput from `30.83k` to `30.95k tokens/s`, mostly by reducing routed-MoE forward overhead (`8.9841 -> 8.8448 ms`). Treat this as the fastest PyTorch packed-sim fixed-router baseline, not a solution for learned router gradients or the real packed-FP4 grouped GEMM.

## Sparse MLA Manual Backward Softmax Cleanup

The `sparse-gather-manual-bwd` custom autograd path now computes the softmax backward directly from key probabilities plus the sink probability. The old path built a zero-filled `grad_probs_all [B, H, S, ATTN_KEYS+1]` tensor and a full sink-extended `grad_scores_all` tensor before slicing key scores. The new path uses:

```text
softmax_dot = sum_k(dP_k * P_k)
dScores_k = P_k * (dP_k - softmax_dot) * scale
dSink = -P_sink * softmax_dot
```

At S2048 this removes two sink-extended fp32 temporaries of shape `[1, 128, 2048, 657]` from the manual backward math, while keeping the same returned `dQ`, `dKV_all`, and `d_attn_sink` shapes.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match,moe_expert_batches_uniform_dense_wgrad_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. Dense-mask and manual sparse-gather attention matched with output max abs error `0.0`; max watched attention gradient relative L2 was `2.1887043997320598e-16`; attention and layer gradcheck passed; FP4 packed-sim still matched fake FP4 exactly; uniform dense-wgrad MoE matched loop routed MoE with max watched gradient relative L2 `1.1961225334218e-16`.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_repeat_5iter
```

Timing result:

| Run | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prior uniform fixed-router best | `20.5557` | `45.5391` | `66.1793` | `30946.25` | `147.117907` | `6.4376` | `8.8448` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_grouped_expert_batches_uniform_dense_wgrad_sparse_gather_manual_bwd_5iter/summary.json` |
| Softmax cleanup | `20.7861` | `45.1539` | `66.0275` | `31017.40` | `147.456142` | `6.4210` | `8.9971` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_5iter/summary.json` |
| Softmax cleanup repeat | `20.7291` | `45.2073` | `66.0232` | `31019.42` | `147.465755` | `6.4746` | `8.8049` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_repeat_5iter/summary.json` |

Segmented attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --profile-attention-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_attr_1iter
```

Attribution result: normal phase total `66.6549 ms`, `30.73k tokens/s`, `146.068115 TFLOPs`; attention dgrad `19.1936 ms`, attention wgrad `19.5931 ms`, MoE dgrad `24.6177 ms`, MoE wgrad `24.3887 ms`. The sparse MLA subblock improved to `attention.sparse_mla.dgrad_qkv = 13.3210 ms` versus the prior manual-bwd attribution `13.6413 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_attr_1iter/summary.json`.

Read: this is a correctness-safe attention backward memory-traffic cleanup and sets the current fixed-router PyTorch packed-sim best to `66.0232 ms`, `31.02k tokens/s`, `147.465755 TFLOPs` at `B=1 S=2048`. It is still not the final sparse MLA kernel because forward and backward continue to materialize `kv_selected [1, 2048, 656, 512]` and use PyTorch einsums/index-add.

Whole-process rocprofv3 trace for this retained path:

```bash
/opt/rocm/bin/rocprofv3 --runtime-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_2iter/summary.txt --output-directory profiles/rocprof_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_2iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/rocprof_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_2iter/torch_summary
```

Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_2iter/`.

The timed-loop summary under rocprof measured `69.0302 ms`, `29.67k tokens/s`, and `141.042016 TFLOPs`; the profiler overhead mostly slowed forward/setup-adjacent blocks versus the no-trace timing. Kernel stats are whole-process, not timed-loop-only: the top entry was the FP4 packed-sim setup quantization argmin/reduce kernel at `58.95%` of kernel time (`215` calls, `1764.215 ms` total), followed by BF16 elementwise add at `6.49%`, BF16 abs at `5.47%`, direct-copy kernels at `3.16%` and `2.80%`, and index elementwise at `2.87%`. Read: this trace is useful for raw kernel/API evidence, but the timed-loop optimization decisions should continue to use the synchronized no-trace and segmented-attribution summaries until a clean profiler-gated timed region is available.

Selected-region rocprof is now available through `--roctx-profile-region`, which wraps only the measured iteration loop in `roctxProfilerResume(0)` / `roctxProfilerPause(0)`. The softmax-cleanup selected-region artifact is `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_2iter/`. Its timed-loop summary under rocprof was `67.1864 ms`, `30.48k tokens/s`, and `144.912497 TFLOPs`; selected-region kernel stats reported `129.658 ms` total kernel time across two measured iterations, with GEMM `93.572 ms`, indexed reduce-add `4.953 ms`, and elementwise/vectorized kernels `25.410 ms`. This trace removes the earlier setup/prepack flood from the top kernel rows.

## Sparse MLA Compressed-Key Grad Reduction

The manual sparse MLA backward now splits `d_kv_all` accumulation by key kind. Vanilla sliding-window keys still use `index_add_`, but CSA/HCA compressed-key columns are aligned with `kv_all[:, S:, :]`, so their gradients are reduced with `grad_selected[:, :, window:, :].sum(dim=1)` and assigned directly. This reduces indexed accumulation from all `656` sparse keys per query to only the `128` vanilla-window keys per query at `B=1 S=2048`.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer,fp4_layer_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. Attention and layer gradcheck passed; FP4 layer match remained finite with output max abs error `0.002455961424857378` and max watched gradient relative L2 `0.3540374210961061`; packed-sim FP4 still matched fake FP4 exactly.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_repeat_5iter
```

Timing result:

| Run | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prior softmax cleanup repeat | `20.7291` | `45.2073` | `66.0232` | `31019.42` | `147.465755` | `6.4746` | `8.8049` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_softmax_lite_repeat_5iter/summary.json` |
| Compressed-key grad reduction | `20.8051` | `44.1902` | `65.0800` | `31468.94` | `149.602757` | `6.4852` | `8.8903` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_5iter/summary.json` |
| Compressed-key grad reduction repeat | `20.8378` | `44.1977` | `65.1216` | `31448.88` | `149.507392` | `6.5763` | `8.9437` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_repeat_5iter/summary.json` |

Segmented attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --profile-attention-blocks --profile-moe-blocks --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_attr_1iter
```

Attribution result: normal phase total `65.8536 ms`, `31.10k tokens/s`, `147.845408 TFLOPs`; attention dgrad `18.2740 ms`, attention wgrad `18.6470 ms`, MoE dgrad `24.7891 ms`, MoE wgrad `24.5612 ms`. The targeted sparse MLA subblock improved to `attention.sparse_mla.dgrad_qkv = 11.9716 ms` from the prior softmax-cleanup `13.3210 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_attr_1iter/summary.json`.

Selected-region rocprof command:

```bash
/opt/rocm/bin/rocprofv3 --selected-regions --marker-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_region_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_2iter/summary.txt --output-directory profiles/rocprof_region_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_2iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --roctx-profile-region --trace-dir profiles/rocprof_region_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_2iter/torch_summary
```

Selected-region rocprof result: timed-loop summary under rocprof `66.4483 ms`, `30.82k tokens/s`, `146.522271 TFLOPs`; selected-region kernel stats reported `127.534 ms` total kernel time across two measured iterations, with GEMM `93.583 ms`, indexed reduce-add `1.390 ms`, new sum/reduction kernels `3.829 ms`, and elementwise/vectorized kernels `26.021 ms`. Compared with the prior selected-region trace, indexed reduce-add fell from `4.953 ms` to `1.390 ms` while total kernel time fell from `129.658 ms` to `127.534 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_2iter/`.

Read: this is a correctness-safe PyTorch sparse MLA backward cleanup and sets the current fixed-router PyTorch packed-sim best to `65.0800 ms`, `31.47k tokens/s`, `149.602757 TFLOPs` at `B=1 S=2048`. It is still not the final sparse MLA kernel because forward and backward continue to materialize `kv_selected [1, 2048, 656, 512]` and use PyTorch einsums; the remaining MLA target is a fused sparse gather/score/softmax/value/dKV kernel that avoids the gathered KV tensor and the residual vanilla-window scatter.

## Grouped Expert-Batch Cyclic Uniform Dense WGrad Candidate

`--moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad` specializes the current fixed-router layout one step further. With the harness router pattern `top_indices[t, k] = (t * top_k + k) % experts`, every expert gets `32` assignments at `B=1 S=2048`. The retained implementation recognizes that cyclic stripe layout and builds the grouped expert input as `[384, 32, 7168]` without the generic sorted `inverse_order` path, while keeping the proven `torch.bmm` shapes for W1/W3/W2 forward, dgrad, and dense routed wgrad. Output and input-gradient reductions are then simple stripe-layout reductions instead of generic assignment-order inversion.

Focused correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_cyclic_uniform_dense_wgrad_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json
```

Focused correctness result: pass on `do-sonle-kernel`. The cyclic uniform dense-wgrad path matched the loop reference with output max abs error `0.0`, output relative L2 `0.0`, and max watched gradient relative L2 `6.449652832387695e-17`.

S2048 no-trace timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_repeat_5iter
```

Timing result:

| Run | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prior compressed-key grad reduction | `20.8051` | `44.1902` | `65.0800` | `31468.94` | `149.602757` | `6.4852` | `8.8903` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_5iter/summary.json` |
| Cyclic broadcast-matmul negative | `20.9054` | `46.7843` | `67.7724` | `30218.79` | `143.659547` | `6.4930` | `9.1073` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_5iter/summary.json` |
| Cyclic bmm-preserving | `20.7982` | `44.0774` | `64.9585` | `31527.82` | `149.882661` | `6.4964` | `8.9960` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_5iter/summary.json` |
| Cyclic bmm-preserving repeat | `20.6910` | `44.0163` | `64.8010` | `31604.46` | `150.247005` | `6.4759` | `8.9655` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_repeat_5iter/summary.json` |

Selected-region rocprof command:

```bash
/opt/rocm/bin/rocprofv3 --selected-regions --marker-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_region_full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_2iter/summary.txt --output-directory profiles/rocprof_region_full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_2iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --roctx-profile-region --trace-dir profiles/rocprof_region_full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_2iter/torch_summary
```

Selected-region rocprof result: timed-loop summary under rocprof `66.4882 ms`, `30.80k tokens/s`, `146.434401 TFLOPs`; selected-region kernel stats reported `127.856 ms` total kernel time across two measured iterations, with GEMM `93.975 ms`, indexed ops `1.410 ms`, sum/reduction kernels `3.846 ms`, and elementwise/vectorized kernels `25.989 ms`. This is nearly identical to the prior compressed-key grad-reduction selected-region profile (`127.534 ms` total kernel time, GEMM `93.583 ms`, indexed ops `1.390 ms`, reductions `3.829 ms`, elementwise/vectorized `26.021 ms`). Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_2iter/`.

Read: the broadcasted `torch.matmul` version is a negative control. The bmm-preserving cyclic path is a small positive fixed-router specialization and sets the current PyTorch packed-sim best to `64.8010 ms`, `31.60k tokens/s`, `150.247005 TFLOPs` at `B=1 S=2048`. The rocprof bucket similarity says this did not create a new kernel class; it mainly removes fixed-layout Python/PyTorch assignment-order overhead while preserving the same GEMM-heavy shape. It is still not learned-router general and not a real packed-FP4 grouped GEMM.

Segmented attribution profiler fix: `profile_microbench.py` now dispatches the attribution-only cached forward through `_routed_experts_grouped_expert_batches_cyclic_uniform_dense_wgrad` and includes the cyclic path in the MoE sub-block splitter. Before this fix, `--profile-moe-blocks` accepted the cyclic CLI flag but the attribution helper fell through to the generic routed path, producing a bogus `moe.routed.wgrad = 12735.8704 ms`.

Fixed attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --profile-attention-blocks --profile-moe-blocks --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_attr_fixed_1iter
```

Fixed attribution result: normal phase total `65.4183 ms`, `31.31k tokens/s`, `148.829291 TFLOPs`; attention dgrad `18.3140 ms`, attention wgrad `18.1959 ms`, MoE dgrad `24.5763 ms`, MoE wgrad `24.4904 ms`. Routed MoE sub-blocks: output scale/reduce dgrad `0.3258 ms`, W2 dgrad `3.1531 ms`, W2 wgrad `5.1661 ms`, SwiGLU dgrad `0.2902 ms`, W1/W3 dgrad `6.2507 ms`, W1 wgrad `4.8103 ms`, W3 wgrad `4.6380 ms`, input gather/reduce dgrad `0.1533 ms`. The routed MoE recomposed input gradient matched the top-level routed dgrad with max abs diff `1.4551915228366852e-11` and relative L2 `3.3430508210585685e-06`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_attr_fixed_1iter/summary.json`.

S4096 primary-proxy and baddbmm dgrad negative-control commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_dgrad_sparse_gather_manual_bwd_comp_sum_2iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 2 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_2iter
```

The baddbmm candidate replaced the cyclic routed input dgrad's two `bmm` calls plus add with one `bmm` followed by `torch.baddbmm`. Focused CUDA correctness still passed for `moe_expert_batches_cyclic_uniform_dense_wgrad_match` with output max abs error `0.0`, output relative L2 `0.0`, and max watched gradient relative L2 `6.449652832387695e-17`.

| Run | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Forward `moe.routed_experts` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Retained cyclic bmm | `B=1 S=2048` | `20.6910` | `44.0163` | `64.8010` | `31604.46` | `150.247005` | `6.4759` | `8.9655` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_repeat_5iter/summary.json` |
| Baddbmm dgrad candidate | `B=1 S=2048` | `20.6151` | `44.1594` | `64.8601` | `31575.64` | `150.110017` | `6.4902` | `8.9508` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_dgrad_sparse_gather_manual_bwd_comp_sum_5iter/summary.json` |
| Baddbmm dgrad candidate | `B=1 S=4096` | `41.7304` | `83.2607` | `125.1236` | `32735.62` | `169.217561` | `22.9633` | `9.8005` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_dgrad_sparse_gather_manual_bwd_comp_sum_2iter/summary.json` |
| Retained cyclic bmm | `B=1 S=4096` | `41.9770` | `83.0132` | `125.1304` | `32733.86` | `169.208441` | `23.3026` | `9.5958` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_2iter/summary.json` |

Read: `baddbmm` is neutral/no-op at S4096 and slightly worse at S2048, so the retained implementation stays with the prior two-`bmm` input dgrad. The S4096 primary proxy now measures `125.1304 ms` against the `86.6 ms` target, or about a `1.45x` time gap. The scaling penalty is mostly attention-side: forward sparse MLA grows from `6.4759 ms` at S2048 to `23.3026 ms` at S4096 because the physical sparse-key axis grows from `656` to `1184`, and the current PyTorch path still materializes gathered KV/probability/score tensors instead of using a fused sparse MLA forward/backward kernel.

S4096 segmented attribution command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --profile-backward-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --profile-attention-blocks --profile-moe-blocks --no-torch-trace --trace-dir profiles/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_attr_1iter
```

S4096 attribution result: normal phase total `125.7094 ms`, `32.58k tokens/s`, `168.428990 TFLOPs`; forward `42.1629 ms`, loss `0.1366 ms`, backward `83.4100 ms`. Forward aggregate split was attention `29.4609 ms`, MoE `10.5505 ms`, boundary `0.6942 ms`. Backward segmented split was attention dgrad `51.3490 ms`, attention wgrad `51.9212 ms`, MoE dgrad `29.2672 ms`, MoE wgrad `29.2512 ms`, boundary dgrad `1.7811 ms`, and boundary wgrad `1.3404 ms`. Attention sub-blocks show the 4K limiter clearly: sparse MLA dgrad_qkv `40.2282 ms`, sink wgrad/softmax path `39.9314 ms`, q-path dgrad/wgrad about `7.0 ms` each, output projection dgrad/wgrad `2.1532/1.5128 ms`, and kv/CSA/HCA paths all sub-`0.8 ms` each. Static traffic estimate for the attention forward path includes gathered sparse KV `4.966 GB`, scores `2.485 GB`, and probabilities `2.483 GB`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_attr_1iter/summary.json`.

Read: at S4096 the next highest-leverage work is a real sparse MLA forward/backward kernel that avoids materializing `kv_selected [1, 4096, 1184, 512]`, scores, and probabilities. MoE remains substantial, but another PyTorch routed-MoE scheduling tweak cannot close the `38.5 ms` S4096 target gap while sparse MLA alone accounts for roughly `23.3 ms` forward plus `40 ms`-class backward substeps.

In-place sparse MLA temporary cleanup:

- Candidate: inside `_SparseGatherManualBackwardMLAFunction`, apply score scaling and invalid-mask fill in place on the freshly allocated score tensor, and accumulate the second selected-KV gradient contribution with `grad_selected.add_(...)`.
- Correctness command: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer,fp4_layer_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json`
- Correctness result: pass. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002455961424857378`, max watched gradient relative L2 `0.3540374210961061`; packed-sim matched fake FP4 exactly.

Timing result:

| Run | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Retained cyclic bmm baseline | `B=1 S=2048` | `20.6910` | `44.0163` | `64.8010` | `31604.46` | `150.247005` | `6.4759` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_bmm_sparse_gather_manual_bwd_comp_sum_repeat_5iter/summary.json` |
| In-place sparse MLA temps | `B=1 S=2048` | `20.6479` | `44.2805` | `65.0139` | `31500.93` | `149.754853` | `6.2684` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_inplace_sparse_mla_temps_5iter/summary.json` |
| Retained cyclic bmm baseline | `B=1 S=4096` | `41.9770` | `83.0132` | `125.1304` | `32733.86` | `169.208441` | `23.3026` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_comp_sum_2iter/summary.json` |
| In-place sparse MLA temps | `B=1 S=4096` | `40.7409` | `83.4034` | `124.2800` | `32957.84` | `170.366251` | `22.0941` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_inplace_sparse_mla_temps_repeat_5iter/summary.json` |

Read: keep this cleanup for the primary S4096 path. It is not a S2048 win (`+0.2129 ms` total), but it gives a repeatable S4096 improvement of `0.8504 ms` total and `1.2084 ms` forward sparse-MLA time. The gap to the `86.6 ms` primary target is now `37.68 ms`, or about `1.43x`; this still requires a real fused sparse MLA kernel rather than more PyTorch temp shaving.

Explicit sparse MLA `bmm` contraction candidate:

- Candidate: replace the custom sparse MLA autograd function's key contractions with explicit per-token batched `torch.bmm` helpers: score contraction `[B*S, H, D] x [B*S, D, K]`, value mix `[B*S, H, K] x [B*S, K, D]`, and selected-KV gradient `[B*S, K, H] x [B*S, H, D]`. This keeps the same materialized `kv_selected`, scores, and probabilities, but makes the contraction layout explicit for ROCm instead of relying on `einsum` lowering.
- Correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer,fp4_layer_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json`. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002455961424857378`, max watched gradient relative L2 `0.3540374210961061`; packed-sim matched fake FP4 exactly.

Timing result:

| Run | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prior retained in-place temps | `B=1 S=2048` | `20.6479` | `44.2805` | `65.0139` | `31500.93` | `149.754853` | `6.2684` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_inplace_sparse_mla_temps_5iter/summary.json` |
| Explicit `bmm` contractions | `B=1 S=2048` | `20.4772` | `44.1962` | `64.7602` | `31624.34` | `150.341539` | `6.2590` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| Prior retained in-place temps | `B=1 S=4096` | `40.7409` | `83.4034` | `124.2800` | `32957.84` | `170.366251` | `22.0941` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_inplace_sparse_mla_temps_repeat_5iter/summary.json` |
| Explicit `bmm` contractions | `B=1 S=4096` | `40.4738` | `83.1777` | `123.7829` | `33090.19` | `171.050416` | `21.9863` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |

S4096 segmented attribution for the `bmm` contraction path measured normal phase total `124.8902 ms`, `32.80k tokens/s`, `171.701348 TFLOPs`; forward `41.4149 ms`, loss `0.1281 ms`, backward `83.3472 ms`. Segmented backward split was attention dgrad `51.7411 ms`, attention wgrad `52.1995 ms`, MoE dgrad `29.5297 ms`, MoE wgrad `29.4413 ms`. Sparse MLA sub-blocks remained the limiter: `attention.sparse_mla.dgrad_qkv = 40.2874 ms`, `attention.sparse_mla.sink_wgrad = 40.3790 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_attr_1iter/summary.json`.

Read: keep the explicit `bmm` contractions. The improvement is small but positive on both active shapes, setting the new PyTorch packed-sim ceiling to `64.7602 ms` at S2048 and `123.7829 ms` at S4096. This still does not change the fundamental sparse MLA contract problem: the path materializes `kv_selected`, scores, and probabilities, and the 4K gap to the `86.6 ms` target remains `37.18 ms` (`1.43x`).

Selectable BSHK sparse MLA probability layout:

- Candidate: add `--attention-impl sparse-gather-manual-bwd-bshk`, which keeps the manual sparse MLA function's score/probability tensors internally as `[B, S, H, K]` instead of `[B, H, S, K]`. This avoids the large head/sequence permutation before the value mix and selected-KV gradient helpers while preserving the original `sparse-gather-manual-bwd` path as the default.
- Correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer,fp4_layer_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-bshk --rmsnorm-save-policy memory-light --json`. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002455961424857378`, max watched gradient relative L2 `0.3540374210961061`; packed-sim matched fake FP4 exactly.
- Baseline dispatch check: `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json` also passed after adding the variant.

| Run | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default explicit `bmm` contractions | `B=1 S=2048` | `20.4772` | `44.1962` | `64.7602` | `31624.34` | `150.341539` | `6.2590` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| BSHK selectable variant | `B=1 S=2048` | `20.5187` | `44.1797` | `64.7801` | `31614.65` | `150.295451` | `6.1908` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bshk_variant_5iter/summary.json` |
| Default explicit `bmm` contractions | `B=1 S=4096` | `40.4738` | `83.1777` | `123.7829` | `33090.19` | `171.050416` | `21.9863` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| BSHK selectable variant | `B=1 S=4096` | `40.7858` | `82.8389` | `123.7602` | `33096.27` | `171.081803` | `22.0650` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bshk_variant_5iter/summary.json` |

Read: keep this as a selectable variant, not as the default. It is correctness-safe and gives the best recorded S4096 total by only `0.0227 ms`, while S2048 is `0.0199 ms` slower; both deltas are in run-noise territory. A prior same-layout run before adding the explicit variant name measured S4096 `123.3391 ms` and S2048 `65.0459 ms`, so the stable conclusion is that BSHK layout is near-neutral in PyTorch and worth preserving only as a swappable probe for the fused sparse MLA kernel contract. The primary target gap is still roughly `37.16 ms` at S4096.

Split-compressed sparse MLA path:

- Candidate: add `--attention-impl sparse-gather-manual-bwd-split-comp`. The path gathers only the vanilla sliding-window keys per query, keeps CSA/HCA compressed KV in their compact `[B, CSA+HCA, D]` form, computes dense compressed score/value/gradient contractions, and writes compressed `dKV` directly while using `index_add_` only for the vanilla window.
- Retained cleanup: split the backward softmax dot and score-gradient computation into window and compressed partitions instead of concatenating `grad_probs` and materializing one full `grad_scores [B, H, S, ATTN_KEYS]` tensor before slicing it again. This keeps the same math and public impl name while reducing PyTorch temporary traffic in the split-compressed custom backward.
- Retained cleanup: replace compressed score/value/gradient einsums with explicit batched `torch.bmm` contractions over `[B*H, S, D] x [B*H, D, C]`, `[B*H, S, C] x [B*H, C, D]`, and `[B*H, C, S] x [B*H, S, D]` followed by head reduction for compressed `dKV`.
- Retained cleanup: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat`, which keeps vanilla KV, CSA, and HCA as separate autograd inputs and avoids materializing `kv_all = cat(kv, csa, hca)` before split-compressed MLA. The sparse MLA math still consumes window KV plus compact CSA/HCA and still materializes score/probability slabs.
- Retained primary-shape cleanup: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-scores`, which keeps the no-KV-cat contract but builds the sink-extended score tensor directly instead of concatenating window/compressed scores and then concatenating the sink. This improves S4096 and is the current primary-shape PyTorch ceiling, but it is not the S2048 fastest path.
- Retained cleanup: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out`, which writes window and compressed score `bmm` outputs directly into the preallocated score slab for `B=1` target shapes and falls back to storage-safe `matmul(..., out=...)` for larger batches. Its no-KV-cat backward also reuses `grad_probs_*` buffers in-place as `grad_scores_*` after the softmax dot is complete. This is now the fastest measured PyTorch packed-sim path on both active shapes.
- Retained cleanup: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-baddbmm`, which keeps the direct score-output path and accumulates compressed CSA/HCA value contributions into the existing `[B, S, H, D]` output/`dQ` buffers via `torch.baddbmm(..., out=...)`. The target view shares storage with the outer BSHD tensor on the MI350 ROCm runtime, so this removes the separate compressed-value output tensor in forward and sparse-MLA backward.
- Retained cleanup: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm`, which keeps the compressed-value `baddbmm` path and accumulates the second selected-window KV-gradient product into the existing `grad_selected [B, S, window, D]` buffer via `torch.baddbmm(..., out=...)`. At the S4096 primary shape this avoids materializing another roughly 1 GiB FP32 selected-gradient tensor before `index_add_`.
- Correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer,fp4_layer_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp --rmsnorm-save-policy memory-light --json`. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002455961424857378`, max watched gradient relative L2 `0.3540374210961061`; packed-sim matched fake FP4 exactly.
- Manual split-softmax negative-control correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-manual-softmax --rmsnorm-save-policy memory-light --json`. Dense-mask and manual split-softmax sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.8402102809193836e-17`, and max watched gradient relative L2 `3.233172278513295e-16`; attention and layer gradcheck passed; FP4 packed-sim matched fake FP4 exactly.
- No-KV-cat correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat --rmsnorm-save-policy memory-light --json`. Dense-mask and no-KV-cat sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; attention and layer gradcheck passed; FP4 packed-sim matched fake FP4 exactly.
- Preallocated-score no-KV-cat correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-scores --rmsnorm-save-policy memory-light --json`. Dense-mask and preallocated-score sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; attention and layer gradcheck passed; FP4 packed-sim matched fake FP4 exactly.
- Preallocated direct-`bmm` no-KV-cat correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out --rmsnorm-save-policy memory-light --json`. Dense-mask and direct-output sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; attention and layer gradcheck passed; FP4 packed-sim matched fake FP4 exactly.
- Compressed-value direct-`baddbmm` no-KV-cat correctness result: pass across focused CUDA checks with `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-baddbmm --rmsnorm-save-policy memory-light`. The `attention_sparse_gather_match` check matched dense-mask attention with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; the `attention,layer,fp4_packed_sim_match` check passed attention/layer gradcheck and packed-sim matched fake FP4 exactly with output max abs error `0.0` and max watched gradient relative L2 `0.0`.
- Selected-gradient direct-`baddbmm` no-KV-cat correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm --rmsnorm-save-policy memory-light --json`. Dense-mask and selected-gradient `baddbmm` sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; attention and layer gradcheck passed; packed-sim matched fake FP4 exactly with output max abs error `0.0` and max watched gradient relative L2 `0.0`.
- Selected-gradient direct-`baddbmm` with RMSNorm `saved-accum` correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm --rmsnorm-save-policy saved-accum --json`. Dense-mask and selected-gradient `baddbmm` sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; attention and layer gradcheck passed; packed-sim matched fake FP4 exactly with output max abs error `0.0` and max watched gradient relative L2 `0.0`.
- Selected-gradient direct-`baddbmm` plus window-sum `dKV` reduction correctness result: pass under the memory-light gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-window-sum --rmsnorm-save-policy memory-light --json` and the saved-accum gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-window-sum --rmsnorm-save-policy saved-accum --json`. Dense-mask and window-sum sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; layer gradcheck passed; packed-sim matched fake FP4 exactly.
- Window-tail-sum `dKV` reduction correctness result: pass under the memory-light gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-window-tail-sum --rmsnorm-save-policy memory-light --json` and the saved-accum gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-window-tail-sum --rmsnorm-save-policy saved-accum --json`. Dense-mask and window-tail-sum sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; layer gradcheck passed; packed-sim matched fake FP4 exactly. A focused CUDA helper comparison against the padded window-sum reducer over `S in {1,2,4,7,128,129}` had max abs difference `7.105427357601002e-15`, consistent with floating reduction-order roundoff.
- Vecdot softmax-dot negative-control correctness result: pass under the memory-light gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-vecdot-window-tail-sum --rmsnorm-save-policy memory-light --json` and the saved-accum gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-vecdot-window-tail-sum --rmsnorm-save-policy saved-accum --json`. Dense-mask and vecdot sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; layer gradcheck passed; packed-sim matched fake FP4 exactly.
- Triton softmax-score backward correctness result: pass under the memory-light gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,attention,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-window-tail-sum --rmsnorm-save-policy memory-light --json` and the saved-accum gate `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-window-tail-sum --rmsnorm-save-policy saved-accum --json`. Dense-mask and sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; layer gradcheck passed; packed-sim matched fake FP4 exactly. The float64 gate validates the mathematical fallback/oracle path; the BF16 timing path below exercises the Triton fp32-accum score-gradient kernel.
- Triton window-gradient reducer negative-control correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-triton-window-sum --rmsnorm-save-policy memory-light --json`. Dense-mask and sparse attention matched with output max abs error `5.551115123125783e-17`, output relative L2 `1.3012251907168185e-17`, and max watched gradient relative L2 `2.1392777142553606e-16`; layer gradcheck passed; packed-sim matched fake FP4 exactly. Standalone BF16-path reducer parity against the PyTorch body/tail diagonal reducer over representative float32 CUDA shapes had max abs error at most `7.62939e-06` and relative L2 at most `1.1289e-07`.
- Shared-x MoE correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_cyclic_uniform_shared_x_dense_wgrad_match --attention-impl sparse-gather-manual-bwd-split-comp --rmsnorm-save-policy memory-light --json`. Output max abs error `0.0`; max watched gradient relative L2 `6.420649535745417e-17`.

Timing result:

| Run | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default explicit `bmm` contractions | `B=1 S=2048` | `20.4772` | `44.1962` | `64.7602` | `31624.34` | `150.341539` | `6.2590` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| Split-compressed sparse MLA | `B=1 S=2048` | `20.8298` | `43.5299` | `64.4440` | `31779.52` | `151.079225` | `6.5443` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_5iter/summary.json` |
| Split-compressed + partitioned softmax bwd | `B=1 S=2048` | `20.8710` | `42.8018` | `63.7588` | `32121.04` | `152.702846` | `6.5200` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_partitioned_softmax_bwd_5iter/summary.json` |
| Split-compressed + compressed bmm | `B=1 S=2048` | `20.0872` | `41.7337` | `61.9075` | `33081.60` | `157.269317` | `5.9599` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_compressed_bmm_5iter/summary.json` |
| Split-compressed + no KV cat | `B=1 S=2048` | `19.9816` | `41.6316` | `61.6962` | `33194.93` | `157.808105` | `5.9718` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_5iter/summary.json` |
| Split-compressed + no KV cat + preallocated scores | `B=1 S=2048` | `19.7024` | `41.9620` | `61.7465` | `33167.87` | `157.679424` | `5.4385` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_scores_5iter/summary.json` |
| Split-compressed + no KV cat + preallocated direct `bmm` scores | `B=1 S=2048` | `19.2461` | `41.9115` | `61.2420` | `33441.08` | `158.978289` | `5.0813` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_inplace_grad_scores_5iter/summary.json` |
| Split-compressed + no KV cat + direct score/value `bmm`/`baddbmm` | `B=1 S=2048` | `18.9955` | `41.6371` | `60.7178` | `33729.82` | `160.350944` | `4.8998` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_baddbmm_5iter/summary.json` |
| Split-compressed + no KV cat + direct score/value/selected-grad `baddbmm` | `B=1 S=2048` | `19.0546` | `41.5163` | `60.6570` | `33763.64` | `160.511710` | `4.8970` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_5iter/summary.json` |
| Same attention + RMSNorm `saved-accum` | `B=1 S=2048` | `19.0467` | `41.3623` | `60.4931` | `33855.12` | `160.946633` | `4.8633` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_rmsnorm_saved_accum_5iter/summary.json` |
| Same attention + window-sum `dKV` + RMSNorm `saved-accum` | `B=1 S=2048` | `18.7830` | `41.1232` | `59.9897` | `34139.18` | `162.297031` | `4.8378` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_window_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Same attention + window-tail-sum `dKV` + RMSNorm `saved-accum` | `B=1 S=2048` | `18.9411` | `40.8110` | `59.8378` | `34225.88` | `162.709175` | `4.8834` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_window_tail_pad_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Same attention + Triton softmax-score backward + RMSNorm `saved-accum` | `B=1 S=2048` | `18.8164` | `39.8155` | `58.7174` | `34878.95` | `165.813880` | n/a | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_triton_softmax_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Same attention + Triton softmax + Triton window reducer negative control | `B=1 S=2048` | `19.0234` | `39.9103` | `59.0190` | `34700.66` | `164.966306` | `4.8534` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_triton_softmax_triton_window_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Same attention + vecdot softmax-dot negative control | `B=1 S=2048` | `22.8403` | `41.1098` | `64.0455` | `31977.27` | `152.019365` | `5.0386` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_vecdot_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Same attention + MoE `baddbmm-xgrad` | `B=1 S=2048` | `19.2131` | `41.5701` | `60.8667` | `33647.31` | `159.958696` | `4.9030` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_5iter/summary.json` |
| Same attention split CSA/HCA branch negative control | `B=1 S=2048` | `19.4984` | `41.9319` | `61.5162` | `33292.04` | `158.269721` | `5.2218` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_ch_5iter/summary.json` |
| Adjacent direct-score baseline for compressed-value A/B | `B=1 S=2048` | `19.1971` | `41.9286` | `61.2141` | `33456.35` | `159.050863` | `5.0813` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_adjacent_comp_baddbmm_baseline_5iter/summary.json` |
| Split-compressed BSHK negative control | `B=1 S=2048` | `20.1477` | `41.8287` | `62.0602` | `33000.22` | `156.882428` | `6.0186` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_bshk_5iter/summary.json` |
| Split-compressed shared-x MoE negative control | `B=1 S=2048` | `20.4661` | `42.2748` | `62.8252` | `32598.40` | `154.972188` | `6.0214` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_shared_x_dense_wgrad_sparse_gather_manual_bwd_split_comp_5iter/summary.json` |
| Split-compressed + in-place score-grad negative control | `B=1 S=2048` | `20.4486` | `42.0213` | `62.5560` | `32738.67` | `155.639013` | `6.0801` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_inplace_scoregrad_5iter/summary.json` |
| Split-compressed manual split-softmax negative control | `B=1 S=2048` | `20.6382` | `41.6772` | `62.3991` | `32821.01` | `156.030467` | `6.3917` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_manual_softmax_5iter/summary.json` |
| BSHK selectable variant | `B=1 S=4096` | `40.7858` | `82.8389` | `123.7602` | `33096.27` | `171.081803` | `22.0650` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bshk_variant_5iter/summary.json` |
| Split-compressed sparse MLA | `B=1 S=4096` | `39.6749` | `77.8022` | `117.6086` | `34827.38` | `180.030319` | `21.0467` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_5iter/summary.json` |
| Split-compressed + partitioned softmax bwd | `B=1 S=4096` | `39.6266` | `76.0133` | `115.7653` | `35381.94` | `182.896958` | `21.0590` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_partitioned_softmax_bwd_5iter/summary.json` |
| Split-compressed + compressed bmm | `B=1 S=4096` | `38.2417` | `73.5797` | `111.9544` | `36586.33` | `189.122720` | `19.7688` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_compressed_bmm_5iter/summary.json` |
| Split-compressed + no KV cat | `B=1 S=4096` | `38.2431` | `73.4260` | `111.8053` | `36635.12` | `189.374919` | `19.8476` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_5iter/summary.json` |
| Split-compressed + no KV cat + preallocated scores | `B=1 S=4096` | `36.9129` | `73.8587` | `110.9018` | `36933.58` | `190.917730` | `17.7258` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_scores_5iter/summary.json` |
| Split-compressed + no KV cat + preallocated direct `bmm` scores | `B=1 S=4096` | `35.0844` | `73.3872` | `108.6055` | `37714.48` | `194.954338` | `16.7742` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_inplace_grad_scores_5iter/summary.json` |
| Split-compressed + no KV cat + direct score/value `bmm`/`baddbmm` | `B=1 S=4096` | `34.5331` | `73.1617` | `107.8308` | `37985.42` | `196.354904` | `16.2223` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_baddbmm_5iter/summary.json` |
| Split-compressed + no KV cat + direct score/value/selected-grad `baddbmm` | `B=1 S=4096` | `34.6563` | `72.5386` | `107.3329` | `38161.63` | `197.265787` | `16.3057` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_5iter/summary.json` |
| Same attention + MoE `baddbmm-xgrad` | `B=1 S=4096` | `34.6534` | `72.2570` | `107.0457` | `38264.04` | `197.795162` | `16.3059` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_5iter/summary.json` |
| Same attention + window-sum `dKV` + MoE `baddbmm-xgrad` | `B=1 S=4096` | `34.7591` | `71.9446` | `106.8413` | `38337.22` | `198.173445` | `16.3105` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_window_sum_5iter/summary.json` |
| Same attention/MoE + window-tail-sum `dKV` | `B=1 S=4096` | `34.5457` | `71.3541` | `106.0356` | `38628.53` | `199.679272` | `16.1984` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_window_tail_pad_sum_5iter/summary.json` |
| Same attention/MoE + Triton softmax-score backward | `B=1 S=4096` | `34.4352` | `67.7588` | `102.3310` | `40026.96` | `206.908063` | n/a | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_window_tail_sum_5iter/summary.json` |
| Same attention/MoE + Triton softmax + Triton window reducer negative control | `B=1 S=4096` | `34.6925` | `67.6138` | `102.4431` | `39983.19` | `206.681808` | `16.2550` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_triton_window_sum_5iter/summary.json` |
| Same attention/MoE + vecdot softmax-dot negative control | `B=1 S=4096` | `37.2896` | `71.9483` | `109.3737` | `37449.60` | `193.585113` | `16.4207` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_vecdot_window_tail_sum_5iter/summary.json` |
| Same attention/MoE + RMSNorm `saved-accum` probe | `B=1 S=4096` | `34.9485` | `72.5535` | `107.6329` | `38055.27` | `196.715978` | `16.3533` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_rmsnorm_saved_accum_5iter/summary.json` |
| Same attention/MoE split CSA/HCA branch negative control | `B=1 S=4096` | `34.7506` | `72.7401` | `107.6244` | `38058.29` | `196.731566` | `16.3378` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_split_ch_5iter/summary.json` |
| Adjacent standard-MoE baseline for MoE `baddbmm-xgrad` A/B | `B=1 S=4096` | `34.6887` | `72.8280` | `107.6494` | `38049.43` | `196.685790` | `16.2178` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_adjacent_moe_baddbmm_xgrad_baseline_5iter/summary.json` |
| Adjacent direct-score baseline for compressed-value A/B | `B=1 S=4096` | `35.1823` | `73.4928` | `108.8068` | `37644.72` | `194.593731` | `16.7651` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_adjacent_comp_baddbmm_baseline_5iter/summary.json` |
| Adjacent compressed-value `baddbmm` baseline for selected-grad A/B | `B=1 S=4096` | `34.6287` | `72.9874` | `107.7525` | `38013.03` | `196.497614` | `16.2614` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_baddbmm_adjacent_sel_baddbmm_baseline_5iter/summary.json` |
| Split-compressed BSHK negative control | `B=1 S=4096` | `38.4758` | `74.0028` | `112.6093` | `36373.55` | `188.022812` | `19.9901` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_bshk_5iter/summary.json` |
| Split-compressed shared-x MoE negative control | `B=1 S=4096` | `38.6312` | `76.9957` | `115.7599` | `35383.58` | `182.905404` | `19.7413` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_shared_x_dense_wgrad_sparse_gather_manual_bwd_split_comp_5iter/summary.json` |
| Split-compressed manual split-softmax negative control | `B=1 S=4096` | `39.1694` | `73.1006` | `112.4029` | `36440.34` | `188.368049` | `20.6605` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_manual_softmax_5iter/summary.json` |

S4096 attribution on the retained compressed-bmm path measured normal phase total `111.3281 ms`, `36.79k tokens/s`, `190.186693 TFLOPs`; forward `38.4789 ms`, loss `0.1359 ms`, backward `72.7133 ms`. Forward aggregate split was attention `25.8151 ms`, MoE `10.5836 ms`, boundary `0.6843 ms`. Segmented backward split was attention dgrad `42.1169 ms`, attention wgrad `42.8506 ms`, MoE dgrad `29.4720 ms`, MoE wgrad `29.1249 ms`, boundary dgrad `1.7747 ms`, and boundary wgrad `1.3101 ms`. Sparse MLA sub-blocks are still the limiter but improved versus the retained partitioned-softmax attribution: `attention.sparse_mla.dgrad_qkv = 30.2582 ms`, `attention.sparse_mla.sink_wgrad = 30.6103 ms`, down from `33.3832/33.1262 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_compressed_bmm_attr_1iter/summary.json`.

S4096 attribution on the no-KV-cat variant measured normal phase total `112.5288 ms`, `36.40k tokens/s`, `188.157327 TFLOPs`; forward `38.6232 ms`, loss `0.1341 ms`, backward `73.7715 ms`. Forward aggregate split was attention `25.8321 ms`, MoE `10.7566 ms`, boundary `0.6725 ms`. Segmented backward split was attention dgrad `42.0807 ms`, attention wgrad `42.8836 ms`, MoE dgrad `29.5483 ms`, MoE wgrad `29.4141 ms`, boundary dgrad `1.7486 ms`, and boundary wgrad `1.3125 ms`. The profiler contract now marks `kv_all_materialized=false` and reports separate sparse MLA gradients `d_kv [1, 4096, 512]`, `d_csa [1, 1024, 512]`, and `d_hca [1, 32, 512]`. Sparse MLA remains the limiter: `attention.sparse_mla.dgrad_qkv = 30.7994 ms`, `attention.sparse_mla.sink_wgrad = 30.6506 ms`, and forward `attn.sparse_mla = 19.8379 ms`. Static attention activation estimate drops the materialized `kv_all` tensor from the no-KV-cat path, but score/probability slabs still dominate at about `2.485 GB` and `2.483 GB`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_attr_1iter/summary.json`.

S4096 attribution on the direct-output preallocated-score no-KV-cat variant measured normal phase total `109.3729 ms`, `37.45k tokens/s`, `193.590069 TFLOPs`; forward `35.5968 ms`, loss `0.1340 ms`, backward `73.6420 ms`. Forward aggregate split was attention `22.9054 ms`, MoE `10.6268 ms`, boundary `0.6649 ms`. Segmented backward split was attention dgrad `41.8030 ms`, attention wgrad `42.4067 ms`, MoE dgrad `29.8327 ms`, MoE wgrad `29.4177 ms`, boundary dgrad `1.7189 ms`, and boundary wgrad `1.3386 ms`. Sparse MLA forward is the main improvement versus preallocated-score copy-in: `attn.sparse_mla = 16.7906 ms`; sparse MLA backward remains essentially the next attention target with `attention.sparse_mla.dgrad_qkv = 30.7328 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_inplace_grad_scores_attr_1iter/summary.json`.

Selected-region rocprof for the retained S4096 compressed-bmm path:

```bash
/opt/rocm/bin/rocprofv3 --selected-regions --marker-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_comp_compressed_bmm_1iter/summary.txt --output-directory profiles/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_comp_compressed_bmm_1iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --roctx-profile-region --trace-dir profiles/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_comp_compressed_bmm_1iter/torch_summary
```

Result: timed loop under rocprof measured `113.2988 ms`, `36152.20 tokens/s`, and `186.878573 TFLOPs`; selected-region kernel dispatch time was `110.7236 ms` over `530` kernel dispatches. Heuristic kernel-name buckets from `trace_kernel_stats.csv`: GEMM/rocBLASLt `77.2645 ms` (`69.78%`), elementwise/mask `16.9094 ms` (`15.27%`), copy/fill/cast `6.8759 ms` (`6.21%`), cat/copy concat `5.6922 ms` (`5.14%`), reductions `1.8236 ms` (`1.65%`), index/scatter `1.2413 ms` (`1.12%`), softmax `0.9168 ms` (`0.83%`). Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_comp_compressed_bmm_1iter/`.

Selected-region rocprof for the no-KV-cat variant measured `112.4812 ms`, `36414.97 tokens/s`, and `188.236907 TFLOPs`; selected-region kernel dispatch time was `110.5896 ms`. Kernel buckets were close to the retained compressed-bmm path: GEMM/rocBLASLt `76.9748 ms` (`69.60%`), elementwise/mask `16.9457 ms` (`15.32%`), copy/fill/cast `7.0250 ms` (`6.35%`), cat/copy concat `5.6270 ms` (`5.09%`), reductions `1.8703 ms` (`1.69%`), index/scatter `1.2087 ms` (`1.09%`), softmax `0.9382 ms` (`0.85%`). Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_comp_no_kv_cat_1iter/`.

Selected-region rocprof for the preallocated-score no-KV-cat variant measured `111.9028 ms`, `36603.21 tokens/s`, and `189.209945 TFLOPs`; selected-region kernel dispatch time was `109.4188 ms`. Kernel buckets show the intended score/sink concat movement: GEMM/rocBLASLt `78.3372 ms` (`71.59%`), copy/fill/cast `17.4822 ms` (`15.98%`), elementwise/mask `7.6031 ms` (`6.95%`), reductions `1.8003 ms` (`1.65%`), cat/copy concat `1.7054 ms` (`1.56%`), index/scatter `1.4426 ms` (`1.32%`), softmax `1.0480 ms` (`0.96%`). Compared with no-KV-cat, cat/copy concat drops `5.6270 -> 1.7054 ms`; some work moves into direct copy/fill/cast, but total selected-region kernel dispatch still drops `110.5896 -> 109.4188 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_comp_no_kv_cat_prealloc_scores_1iter/`.

Selected-region rocprof for the direct-output preallocated-score no-KV-cat variant measured `110.1115 ms`, `37198.67 tokens/s`, and `192.287994 TFLOPs`; selected-region kernel dispatch time was `107.7668 ms`. Kernel buckets: GEMM/rocBLASLt `77.7542 ms` (`72.15%`), copy/fill/cast `16.5919 ms` (`15.40%`), elementwise/mask `7.5511 ms` (`7.01%`), reductions `1.8168 ms` (`1.69%`), cat/copy concat `1.6953 ms` (`1.57%`), index/scatter `1.3655 ms` (`1.27%`), softmax `0.9920 ms` (`0.92%`). Compared with preallocated-score copy-in, total selected-region dispatch drops `109.4188 -> 107.7668 ms`, with copy/fill/cast dropping `17.4822 -> 16.5919 ms` while concat stays near `1.70 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_split_comp_no_kv_cat_prealloc_bmm_out_1iter/`.

Selected-region rocprof for the pre-window-sum S4096 best combo (selected-gradient attention plus MoE `baddbmm-xgrad`) measured `108.2304 ms`, `37845.19 tokens/s`, and `195.630032 TFLOPs`; selected-region kernel dispatch time was `105.9585 ms`. Kernel buckets: GEMM/rocBLASLt `77.7769 ms` (`73.40%`), elementwise/mask `15.3787 ms` (`14.51%`), copy/fill/cast `6.9405 ms` (`6.55%`), reductions `1.7957 ms` (`1.69%`), cat/copy concat `1.6674 ms` (`1.57%`), index/scatter `1.4187 ms` (`1.34%`), softmax `0.9807 ms` (`0.93%`). Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_1iter_v2/`.

Read: add the window-sum `dKV` reduction to the direct-output preallocated no-KV-cat path with compressed-value and selected-gradient `baddbmm` accumulation. This replaces the vanilla sliding-window `index_add_` scatter with a diagonal band sum over `grad_selected [B, S, 128, 512]`. With standard cyclic dense-wgrad MoE and RMSNorm `saved-accum`, the padded window-sum path lifted S2048 to `59.9897 ms`. The follow-up body/tail version keeps the full-window body as a pad-free anti-diagonal view and handles only the final `window-1` tail rows with a small padded diagonal sum; this is now the fastest PyTorch packed-sim attention path on both active shapes. It sets the S2048 ceiling at `59.8378 ms`, improving the prior padded window-sum best by `0.1519 ms` (`59.9897 -> 59.8378`). Combined with MoE `baddbmm-xgrad`, it sets the S4096 primary-shape ceiling at `106.0356 ms`, improving the prior padded window-sum best by `0.8057 ms` (`106.8413 -> 106.0356`). The selected-gradient attention variant itself improved S2048 by `0.0608 ms` (`60.7178 -> 60.6570`) and S4096 by `0.4979 ms` (`107.8308 -> 107.3329`) versus the compressed-value `baddbmm` attention path. The RMSNorm `saved-accum` policy is correctness-safe and helps S2048, but it regresses the S4096 primary combo, so keep it as a per-shape probe. The split-CSA/HCA branch variant, BSHK layout variant, shared-x cyclic MoE variant, manual split-softmax variant, and fused residual/RMSNorm boundary remain correctness-safe negative controls. The new S4096 primary target gap is `19.44 ms` (`1.22x`). This is still a PyTorch custom-autograd surface, not the final fused sparse MLA or packed FP4 MoE kernel; scores/probabilities remain materialized, and the vanilla sliding-window gather is still a PyTorch `index_select` plus diagonal-reduction pair.

S4096 attribution on the selected-gradient `baddbmm` variant measured normal phase total `108.4254 ms`, `37.78k tokens/s`, `195.278190 TFLOPs`; forward `35.1102 ms`, loss `0.1265 ms`, backward `73.1888 ms`. Forward aggregate split was attention `22.3244 ms` and MoE `10.7557 ms`. Segmented backward split was attention dgrad `40.6986 ms`, attention wgrad `41.5736 ms`, MoE dgrad `29.4770 ms`, MoE wgrad `29.4012 ms`, boundary dgrad `1.7651 ms`, and boundary wgrad `1.2953 ms`. The sparse MLA custom backward remains the attention limiter, but selected-gradient accumulation improves the focused sparse-MLA dgrad probe versus the previous compressed-value `baddbmm` attribution (`30.4158 -> 29.5865 ms`). Note that `attention.sparse_mla.sink_wgrad` also invokes the whole custom sparse-MLA backward in this segmented autograd probe, so it should be read as a second full custom-backward invocation rather than a literal sink-only cost. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_attr_1iter/summary.json`.

S4096 attribution on the selected-gradient attention plus MoE `baddbmm-xgrad` variant measured normal phase total `107.5441 ms`, `38.09k tokens/s`, `196.878378 TFLOPs`; forward `34.6926 ms`, loss `0.1322 ms`, backward `72.7193 ms`. Forward aggregate split was attention `22.2028 ms` and MoE `10.5201 ms`. Segmented backward split was attention dgrad `40.5670 ms`, attention wgrad `41.1834 ms`, MoE dgrad `29.4293 ms`, MoE wgrad `29.2842 ms`, boundary dgrad `1.6899 ms`, and boundary wgrad `1.2765 ms`. Sparse MLA remains the limiter: `attention.sparse_mla.dgrad_qkv = 29.7790 ms`; routed MoE dgrad/wgrad remain roughly `27.5861/27.8374 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_attr_1iter/summary.json`.

S4096 attribution on the window-sum selected-gradient attention plus MoE `baddbmm-xgrad` variant measured normal phase total `107.2581 ms`, `38.19k tokens/s`, `197.403445 TFLOPs`; forward `35.1433 ms`, loss `0.1458 ms`, backward `71.9691 ms`. Forward aggregate split was attention `22.4887 ms` and MoE `10.5727 ms`. Segmented backward split was attention dgrad `40.3248 ms`, attention wgrad `41.3200 ms`, MoE dgrad `28.9180 ms`, MoE wgrad `29.1293 ms`, boundary dgrad `1.7431 ms`, and boundary wgrad `1.2882 ms`. The targeted sparse MLA subblock improved to `attention.sparse_mla.dgrad_qkv = 29.2071 ms` and `attention.sparse_mla.sink_wgrad = 29.2470 ms`, down from the pre-window-sum selected-gradient attribution's `29.7790 ms` sparse MLA dgrad signal. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_window_sum_attr_1iter/summary.json`.

S4096 attribution on the retained window-tail-sum selected-gradient attention plus MoE `baddbmm-xgrad` variant measured normal phase total `106.1824 ms`, `38.58k tokens/s`, `199.403182 TFLOPs`; forward `34.8782 ms`, loss `0.1370 ms`, backward `71.1672 ms`. Forward sparse MLA was `16.3907 ms`. Segmented backward split was attention dgrad `40.0763 ms`, attention wgrad `40.9635 ms`, MoE dgrad `27.3083 ms`, MoE wgrad `27.6247 ms`, boundary dgrad `1.6971 ms`, and boundary wgrad `1.2927 ms`. Sparse MLA remains the first attention target: `attention.sparse_mla.dgrad_qkv = 28.8154 ms` and `attention.sparse_mla.sink_wgrad = 28.6777 ms`; routed MoE remains second with `moe.routed.w1_w3.dgrad = 6.9775 ms`, `moe.routed.w2.dgrad = 3.2058 ms`, and routed wgrads around `5.16-5.94 ms` each. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_window_tail_pad_sum_attr_1iter/summary.json`.

S4096 attribution on the Triton softmax-score backward variant measured normal phase total `102.5415 ms`, `39.94k tokens/s`, `206.483402 TFLOPs`; forward `34.9306 ms`, loss `0.1430 ms`, backward `67.4679 ms`. Forward sparse MLA was `16.2731 ms`. Segmented backward split was attention dgrad `35.8929 ms`, attention wgrad `36.9378 ms`, MoE dgrad `29.2458 ms`, MoE wgrad `29.0659 ms`, boundary dgrad `1.7079 ms`, and boundary wgrad `1.3095 ms`. The fused softmax-dot/score-gradient Triton row kernel reduces the sparse MLA custom-backward signals to `attention.sparse_mla.dgrad_qkv = 24.4269 ms` and `attention.sparse_mla.sink_wgrad = 24.3669 ms`; this confirms the previous softmax backward reduction path was a real limiter, although the remaining sparse MLA path still materializes score/probability slabs and uses PyTorch bmm/gather around the fused point. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_window_tail_sum_attr_1iter/summary.json`.

Vecdot softmax-dot negative-control read: the variant replaces the backward softmax-dot elementwise product/reduction with `torch.linalg.vecdot(..., dim=-1)` while keeping the retained window-tail `dKV` reducer and selected-gradient `baddbmm` path. It is mathematically sound and correctness-safe, but it is slower on both active shapes: S2048 regressed `59.8378 -> 64.0455 ms` and S4096 regressed `106.0356 -> 109.3737 ms`. The S4096 forward phase in particular rose `34.5457 -> 37.2896 ms`, likely from less favorable ROCm/PyTorch scheduling around vecdot and adjacent score/probability tensors. Keep it selectable only as a negative control; the real sparse MLA fix still needs a fused softmax-backward/score-gradient kernel rather than replacing one PyTorch reduction primitive with another.

Triton softmax-score backward read: the variant fuses the backward softmax dot, window score-gradient update, compressed score-gradient update, and sink-dot output into one Triton row kernel over `[B, H, S]`. It keeps the retained window-tail `dKV` reducer and selected-gradient/compressed-value `baddbmm` path. This is now the fastest measured PyTorch packed-sim ceiling on both active shapes: S2048 improves `59.8378 -> 58.7174 ms`, and S4096 improves `106.0356 -> 102.3310 ms`. The S4096 primary target gap is now `15.73 ms` (`1.18x`). Keep the variant as retained current best, but treat it as a narrow fused-kernel prototype rather than the final sparse MLA kernel.

Triton masked-softmax forward plus softmax-score backward read: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-window-tail-sum`, which fuses forward scale, invalid-mask application, sink handling, and row softmax into one Triton in-place kernel over the preallocated score slab. Correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` and on CUDA BF16 for `attention_sparse_gather_match,fp4_packed_sim_match`; BF16 dense-vs-sparse attention output max abs error was `0.0` and max watched grad relative L2 was `4.877021980758919e-08`.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward sparse MLA ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Triton fwd+bwd softmax, saved-accum | `B=1 S=2048` | `18.7387` | `39.8602` | `58.6825` | `34899.68` | `165.912416` | `4.6037` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Triton fwd+bwd softmax, memory-light | `B=1 S=4096` | `32.9795` | `67.8430` | `100.9594` | `40570.78` | `209.719177` | `14.5375` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_window_tail_sum_5iter/summary.json` |

Selected-region rocprof for the Triton fwd+bwd softmax S4096 best measured `102.0207 ms`, `40148.71 tokens/s`, and `207.537417 TFLOPs` under profiler overhead; selected-region kernel dispatch time was `99.7934 ms`. Kernel buckets: GEMM/rocBLASLt `77.083 ms` (`77.2%`), copy/fill/cast/mask `11.683 ms` (`11.7%`), elementwise/vectorized `6.069 ms` (`6.1%`), Triton/custom softmax `3.328 ms` (`3.3%`, `_masked_softmax_inplace_kernel` `1.820 ms` plus `_softmax_bwd_scores_kernel` `1.508 ms`), reductions `1.452 ms` (`1.5%`), index/gather/scatter `0.159 ms` (`0.2%`), and native softmax `0.019 ms`. Compared with the prior selected-region Triton-backward-only trace, total kernel time drops `101.104 -> 99.793 ms`, copy/fill/cast/mask drops `13.584 -> 11.683 ms`, native softmax is effectively removed, and Triton/custom softmax rises `1.497 -> 3.328 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_window_tail_sum_1iter/`.

Prior read: keep the fwd+bwd Triton softmax variant as the no-fast-RoPE baseline. It improves S2048 by `0.0349 ms` (`58.7174 -> 58.6825`) and S4096 by `1.3717 ms` (`102.3310 -> 100.9594`). The primary S4096 gap at this point was `14.36 ms` (`1.17x`). The remaining selected-region time was dominated by GEMM/bmm dispatches plus PyTorch copy/cast/elementwise work around the materialized score/probability slabs; this still points toward a real fused sparse MLA forward/backward kernel, not another standalone softmax micro-kernel.

Fast q/kv RoPE read: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-rope-window-tail-sum`, which keeps the retained Triton fwd+bwd softmax path but applies q/kv RoPE tails in-place with a custom autograd wrapper. The inverse output RoPE remains functional because in-place mutation of the custom sparse-MLA autograd output is forbidden by PyTorch; the fully in-place inverse attempt failed with the expected custom-Function view/in-place error and was not retained. Correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` and on CUDA BF16 for `attention_sparse_gather_match,fp4_packed_sim_match`; BF16 dense-vs-sparse attention output max abs error was `0.0` and max watched grad relative L2 was `4.877021980758919e-08`.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward sparse MLA ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Triton fwd+bwd softmax + fast q/kv RoPE, saved-accum | `B=1 S=2048` | `18.1587` | `39.4751` | `57.7190` | `35482.25` | `168.681948` | `4.5581` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_rope_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Triton fwd+bwd softmax + fast q/kv RoPE, memory-light | `B=1 S=4096` | `31.7782` | `67.0643` | `98.9750` | `41384.18` | `213.923838` | `14.5058` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_rope_window_tail_sum_5iter/summary.json` |

Read: keep the fast q/kv RoPE variant as the current PyTorch packed-sim ceiling. It improves S2048 by `0.9635 ms` (`58.6825 -> 57.7190`) and S4096 by `1.9844 ms` (`100.9594 -> 98.9750`). The primary S4096 gap is now `12.38 ms` (`1.14x`). This is still a tensor-movement cleanup inside a PyTorch custom-autograd surface; the remaining work still points to a fused sparse MLA forward/backward kernel and real packed-FP4 grouped expert GEMM rather than more standalone RoPE/softmax shaving.

Triton q/kv RoPE read: add `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-window-tail-sum`, which keeps the retained Triton fwd+bwd softmax path and replaces the q/kv RoPE custom autograd tail rotation with a shape-specialized Triton in-place kernel for contiguous `[B, S, D]` and `[B, S, H, D]` tensors. Correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` and on CUDA BF16 for `attention_sparse_gather_match,fp4_packed_sim_match`; BF16 dense-vs-sparse attention output max abs error was `0.0` and max watched grad relative L2 was `4.877021980758919e-08`. Float64 falls back to the PyTorch in-place RoPE path because the Triton kernel is only used for fp16/bf16/fp32 tensors; the BF16 run covers the actual Triton kernel path.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward sparse MLA ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Triton fwd+bwd softmax + Triton q/kv RoPE, saved-accum | `B=1 S=2048` | `17.9987` | `39.4105` | `57.4940` | `35621.12` | `169.342167` | `4.5674` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Triton fwd+bwd softmax + Triton q/kv RoPE, memory-light | `B=1 S=4096` | `31.7663` | `66.8305` | `98.7259` | `41488.61` | `214.463676` | `14.5875` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_window_tail_sum_5iter/summary.json` |

Read: the Triton q/kv RoPE variant improved S2048 by `0.2250 ms` (`57.7190 -> 57.4940`) and S4096 by `0.2491 ms` (`98.9750 -> 98.7259`). This confirmed there was still a small standalone RoPE dispatch/tensor-motion tax, but it has been superseded by the output inverse RoPE variant below.

S4096 attribution on the previous Triton q/kv RoPE best measured normal phase total `99.3204 ms`, `41.24k tokens/s`, `213.912125 TFLOPs`; forward `32.1799 ms`, loss `0.1354 ms`, backward `67.0051 ms`. Forward aggregate split was attention `19.5734 ms`, MoE `10.5512 ms`, and boundary `0.6842 ms`. Segmented backward split was attention dgrad `35.3549 ms`, attention wgrad `36.0361 ms`, MoE dgrad `29.1164 ms`, MoE wgrad `28.8043 ms`, boundary dgrad `1.7106 ms`, and boundary wgrad `1.2814 ms`. Sparse MLA remained the first attention target (`attention.sparse_mla.dgrad_qkv = 24.8923 ms`, `attention.sparse_mla.sink_wgrad = 24.2897 ms`), and routed MoE remained the main non-attention target (`moe.routed.dgrad = 27.2867 ms`, `moe.routed.wgrad = 27.3967 ms`). Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_window_tail_sum_attr_1iter/summary.json`.

Triton output inverse RoPE read: keep the same `--attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-window-tail-sum`, but route the output inverse RoPE through an out-of-place custom autograd wrapper that clones the sparse-MLA output and applies the same Triton tail-rotation kernel to the clone. This keeps the PyTorch custom-Function in-place guard satisfied while avoiding the functional PyTorch tail cat/stack path. Correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` and on CUDA BF16 for `attention_sparse_gather_match,fp4_packed_sim_match`; BF16 dense-vs-sparse attention output max abs error was `0.0` and max watched grad relative L2 was `4.877021980758919e-08`.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward sparse MLA ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Triton q/kv + output inverse RoPE, saved-accum | `B=1 S=2048` | `17.6822` | `38.9732` | `56.7410` | `36093.85` | `171.589516` | `4.5550` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_out_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Triton q/kv + output inverse RoPE, memory-light | `B=1 S=4096` | `30.8362` | `65.8047` | `96.7746` | `42325.16` | `218.787934` | `14.5602` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_out_window_tail_sum_5iter/summary.json` |

Read: promote the output inverse RoPE variant as the current PyTorch packed-sim ceiling. It improves S2048 by `0.7530 ms` (`57.4940 -> 56.7410`) and S4096 by `1.9513 ms` (`98.7259 -> 96.7746`). The primary S4096 gap is now `10.17 ms` (`1.12x`) against the `86.6 ms` target. This is still a narrow RoPE/tensor-motion cleanup; sparse MLA forward/backward fusion and real packed-FP4 grouped expert GEMM remain the high-leverage work.

S4096 attribution on the current output inverse RoPE best measured normal phase total `96.9791 ms`, `42.24k tokens/s`; forward `31.1299 ms`, loss `0.1400 ms`, backward `65.7092 ms`. Forward aggregate split was attention `19.3111 ms`, MoE `10.6371 ms`, and boundary `0.6730 ms`. Segmented backward split was attention dgrad `34.4051 ms`, attention wgrad `35.3146 ms`, MoE dgrad `29.3955 ms`, MoE wgrad `28.9241 ms`, boundary dgrad `1.7131 ms`, and boundary wgrad `1.2773 ms`. Sparse MLA remains the first attention target (`attention.sparse_mla.dgrad_qkv = 24.8163 ms`, `attention.sparse_mla.sink_wgrad = 24.5807 ms`), output inverse RoPE dgrad is still a visible pointwise/memory path (`attention.output_rope_inv.dgrad = 1.2576 ms`), and routed MoE remains the main non-attention target (`moe.routed.dgrad = 27.5813 ms`, `moe.routed.wgrad = 27.5086 ms`). Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_out_window_tail_sum_attr_1iter/summary.json`.

Single-kernel output inverse RoPE read: replace the out-of-place `clone()` plus in-place Triton tail rotation with one Triton out-of-place copy+rotate kernel. The wrapper uses `block_rows=4` for `S<=2048` and `block_rows=8` for the S4096 primary shape, based on measured shape-specific launch behavior. Correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` and on CUDA BF16 for `attention_sparse_gather_match,fp4_packed_sim_match`; BF16 dense-vs-sparse attention output max abs error was `0.0` and max watched grad relative L2 was `4.877021980758919e-08`.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward sparse MLA ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Single-kernel output inverse RoPE, `block_rows=4` | `B=1 S=2048` | `17.6446` | `38.8813` | `56.6107` | `36176.94` | `171.984479` | `4.5797` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Single-kernel output inverse RoPE, `block_rows=8` | `B=1 S=4096` | `30.6314` | `65.8273` | `96.5923` | `42405.03` | `219.200848` | `14.5265` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_br8_window_tail_sum_5iter/summary.json` |

Read: promote the single-kernel output inverse RoPE variant as the current PyTorch packed-sim ceiling. It improves S2048 by `0.1303 ms` (`56.7410 -> 56.6107`) and S4096 by `0.1823 ms` (`96.7746 -> 96.5923`). The primary S4096 gap is now `9.99 ms` (`1.12x`) against the `86.6 ms` target. A `block_rows=8` S2048 control measured `56.7334 ms`, so keep the shape heuristic rather than one global tile. This is a narrow memory-kernel cleanup; sparse MLA forward/backward fusion and real packed-FP4 grouped expert GEMM remain the high-leverage work.

Triton SwiGLU MoE read: add a Triton fused forward/backward helper for `silu(gate) * up` in the packed-sim routed and shared MoE custom autograd paths. The fallback keeps the prior PyTorch math for CPU, float64, non-contiguous tensors, and non-HIP/CUDA paths. Correctness passed the BF16 gate for `moe_expert_batches_cyclic_uniform_dense_wgrad_baddbmm_xgrad_match,attention_sparse_gather_match,fp4_packed_sim_match` on the retained attention path; the MoE baddbmm-xgrad max watched gradient relative L2 was `0.005704804425938679`, within the BF16 tolerance. The float64 fallback gate also passed for `moe_expert_batches_cyclic_uniform_dense_wgrad_baddbmm_xgrad_match,fp4_packed_sim_match` with max watched gradient relative L2 `6.420649535745417e-17`.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward routed MoE ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Triton SwiGLU + single-kernel output inverse RoPE, saved-accum | `B=1 S=2048` | `17.6788` | `38.3163` | `56.0807` | `36518.82` | `173.609774` | `8.7313` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_swiglu_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Triton SwiGLU + single-kernel output inverse RoPE, memory-light | `B=1 S=4096` | `30.6545` | `64.4109` | `95.2026` | `43024.04` | `222.400619` | `9.3910` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_swiglu_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_window_tail_sum_5iter/summary.json` |

Read: promote the Triton SwiGLU variant as the current PyTorch packed-sim ceiling. It improves S2048 by `0.5300 ms` (`56.6107 -> 56.0807`) and S4096 by `1.3897 ms` (`96.5923 -> 95.2026`). The primary S4096 target gap is now `8.60 ms` (`1.10x`) against the `86.6 ms` target. This is a pointwise-fusion cleanup inside the PyTorch packed-sim MoE path; it does not replace the required real packed-FP4 grouped training GEMM.

Triton q-head RMSNorm read: add selectable attention impl `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum`, which replaces the post-`wq_b` query-head RMSNorm reduction with a Triton custom autograd forward/backward while preserving fallback behavior for CPU, float64, and non-contiguous paths. Correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` with max watched gradient relative L2 `2.1392777142553606e-16`; CUDA BF16 passed for `attention_sparse_gather_match,fp4_packed_sim_match` with attention output max abs error `0.0` and max watched gradient relative L2 `4.877021980758919e-08`.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.q_proj` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Triton q-head RMSNorm + Triton SwiGLU, memory-light | `B=1 S=2048` | `16.9319` | `36.8775` | `53.8940` | `38000.49` | `180.653606` | `0.9570` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_memory_light_5iter/summary.json` |
| Triton q-head RMSNorm + Triton SwiGLU, saved-accum | `B=1 S=4096` | `29.8566` | `61.4322` | `91.4248` | `44801.86` | `231.590559` | `1.7308` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_5iter/summary.json` |

Read: promote the Triton q-head RMSNorm variant plus the retested RMSNorm policy split as the current PyTorch packed-sim ceiling. It improves S2048 by `2.1866 ms` (`56.0807 -> 53.8940`) and S4096 by `3.7778 ms` (`95.2026 -> 91.4248`). Forward `attn.q_proj` drops from `1.3887 -> 0.9570 ms` at S2048 and `2.5710 -> 1.7308 ms` at S4096. The primary S4096 target gap is now `4.82 ms` (`1.06x`) against the `86.6 ms` target.

RMSNorm policy retest after q-head RMSNorm: the old S4096 memory-light preference flipped. On the qnorm stack, S2048 memory-light improved over saved-accum by `0.2998 ms` (`54.1938 -> 53.8940`), while S4096 saved-accum improved over memory-light by `0.4488 ms` (`91.8736 -> 91.4248`). Saved-accum qnorm correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` with max watched gradient relative L2 `2.1392777142553606e-16`; the prior qnorm CUDA BF16 gate passed for `attention_sparse_gather_match,fp4_packed_sim_match` with attention output max abs error `0.0` and max watched gradient relative L2 `4.877021980758919e-08`.

Current S4096 attribution on the Triton q-head RMSNorm saved-accum ceiling measured normal phase total `90.9756 ms`, `45023.08 tokens/s`, and `232.734102 TFLOPs`; forward `29.8215 ms`, loss `0.1272 ms`, backward `61.0269 ms`. Forward aggregate split was attention `18.3769 ms`, MoE `10.3258 ms`, and boundary `0.6113 ms`. Segmented backward split was attention dgrad `31.3030 ms`, attention wgrad `31.8035 ms`, MoE dgrad `28.0050 ms` including shared, MoE wgrad `28.1100 ms` including shared, boundary dgrad `1.5430 ms`, and boundary wgrad `1.1810 ms`. The next high-leverage targets remain sparse MLA (`attention.sparse_mla.dgrad_qkv = 24.3453 ms`, `attention.sparse_mla.sink_wgrad = 24.2384 ms`) and routed MoE (`moe.routed.dgrad = 26.1001 ms`, `moe.routed.wgrad = 26.6504 ms`); the q path is now smaller at `attention.q_path.dgrad = 3.3056 ms` and `attention.q_path.wgrad = 3.1247 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_attr_1iter/summary.json`.

Triton qnorm sink-gradient negative-control read: add selectable attention impl `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-sinkgrad-window-tail-sum`, which folds `d_attn_sink` accumulation into the Triton softmax-backward row kernel with one fp32 atomic add per `[B,H,S]` row. Correctness passed on CUDA float64 for `attention_sparse_gather_match,layer,fp4_packed_sim_match` with max watched gradient relative L2 `2.1392777142553606e-16`, and on CUDA BF16 for `attention_sparse_gather_match,fp4_packed_sim_match` with attention output max abs error `0.0` and max watched gradient relative L2 `0.0`. Timing regressed both active shapes: S2048 memory-light measured `55.3516 ms`, `36.9998k tokens/s`, `175.896596 TFLOPs`; S4096 saved-accum measured `93.6419 ms`, `43.7411k tokens/s`, `226.107383 TFLOPs`. Keep the atomics-based sink-gradient path as a correctness-safe kernel-contract probe only; the retained qnorm path's separate reduction is faster in this PyTorch custom-autograd surface. Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_sinkgrad_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_memory_light_5iter/summary.json` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_sinkgrad_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_5iter/summary.json`.

Previous S4096 attribution on the Triton SwiGLU ceiling measured normal phase total `95.1673 ms`, `43040.00 tokens/s`, and `222.483117 TFLOPs`; forward `30.9875 ms`, loss `0.1346 ms`, backward `64.0452 ms`. Forward aggregate split was attention `19.4757 ms`, MoE `10.2968 ms`, and boundary `0.6750 ms`. Segmented backward split was attention dgrad `34.4424 ms`, attention wgrad `34.4597 ms`, MoE dgrad `27.8782 ms`, MoE wgrad `27.8156 ms`, boundary dgrad `1.7386 ms`, and boundary wgrad `1.2884 ms`. The focused sparse MLA signals remain the first target: `attention.sparse_mla.dgrad_qkv = 25.2868 ms` and `attention.sparse_mla.sink_wgrad = 24.9312 ms`; routed MoE remains second with `moe.routed.dgrad = 25.9789 ms`, `moe.routed.wgrad = 26.4287 ms`, and the fused routed SwiGLU dgrad itself down to `0.5228 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_swiglu_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_window_tail_sum_attr_1iter/summary.json`.

Selected-region rocprof on the Triton SwiGLU ceiling measured timed-loop total `99.1370 ms`, `41316.55 tokens/s`, and `213.574236 TFLOPs` under profiler overhead; selected-region kernel dispatch time was `94.5266 ms` over `409` dispatches, and HIP API time was `86.9563 ms`. Heuristic kernel buckets from `trace_kernel_stats.csv`: GEMM/rocBLASLt `77.3279 ms` (`81.81%`), copy/fill/cast `7.4715 ms` (`7.90%`), elementwise `3.8452 ms` (`4.07%`), Triton/custom softmax `3.3927 ms` (`3.59%`), reductions `1.4719 ms` (`1.56%`), RoPE `0.6146 ms` (`0.65%`), Triton SwiGLU `0.2434 ms` (`0.26%`), and index/gather/scatter `0.1594 ms` (`0.17%`). The first six GEMM rows alone summed to about `64.19 ms`. Read: RoPE, softmax, and SwiGLU are now small enough that the next meaningful target is still fused sparse MLA plus a real packed-FP4 grouped MoE training kernel, not more standalone pointwise cleanup. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_s4096_triton_swiglu_current/`.

Shared-X cyclic MoE retest: `--moe-impl grouped-expert-batches-cyclic-uniform-shared-x-dense-wgrad` remains correctness-safe but is not promotable under the current attention stack. It measured S2048 `57.3460 ms`, `35.71k tokens/s`, `169.778987 TFLOPs`, and S4096 `100.6940 ms`, `40.68k tokens/s`, `210.271905 TFLOPs`. It saves one conceptual repeated `x_group` expansion, but the PyTorch scheduling and backward surface lose more than they save. Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_shared_x_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_shared_x_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_window_tail_sum_5iter/summary.json`.

Shared-X cyclic MoE qnorm-stack retest: the same shared-activation MoE variant also regresses after Triton q-head RMSNorm. A BF16 CUDA smoke passed for `moe_expert_batches_cyclic_uniform_shared_x_dense_wgrad_match,attention_sparse_gather_match,fp4_packed_sim_match`; the MoE max watched gradient relative L2 was `0.005704804425938679`, sparse-attention max watched gradient relative L2 was `4.3429689001578356e-09`, and packed-sim matched fake FP4 exactly. Timing measured S2048 memory-light `54.6247 ms`, `37.49k tokens/s`, `178.237164 TFLOPs`, regressing the retained qnorm S2048 ceiling by `0.7307 ms`; S4096 saved-accum measured `95.1182 ms`, `43.06k tokens/s`, `222.597906 TFLOPs`, regressing the retained qnorm S4096 ceiling by `3.6934 ms`. Keep shared-X as a correctness-safe MoE contract probe, not a promoted path. Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_cyclic_uniform_shared_x_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_memory_light_5iter/summary.json` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_shared_x_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_5iter/summary.json`.

Loop-k cyclic MoE negative-control read: add `--moe-impl grouped-expert-batches-cyclic-uniform-loop-k-dense-wgrad-baddbmm-xgrad`, which avoids materializing the repeated fixed-router `x_group` and routed `y_unscaled` tensors by launching per-top-k grouped `bmm` slices and accumulating the routed output/input-gradient by group. Correctness passed on CUDA float64 for `moe_expert_batches_cyclic_uniform_loop_k_dense_wgrad_baddbmm_xgrad_match,fp4_packed_sim_match` with max watched gradient relative L2 `6.420649535745417e-17`, and on CUDA BF16 for `moe_expert_batches_cyclic_uniform_loop_k_dense_wgrad_baddbmm_xgrad_match,attention_sparse_gather_match,fp4_packed_sim_match` with MoE max watched gradient relative L2 `0.005704804425938679`.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward routed MoE ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Loop-k cyclic dense-wgrad baddbmm-xgrad MoE, saved-accum | `B=1 S=2048` | `17.8607` | `60.1344` | `78.0783` | `26230.07` | `124.697258` | `8.9828` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_loop_k_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |

Read: do not promote the loop-k path and do not spend a primary S4096 run on it unless the surrounding MoE implementation changes substantially. It removes two large conceptual repeated tensors, but splitting each routed expert GEMM into six per-top-k grouped calls fragments the backward path enough to regress S2048 by `21.9977 ms` versus the Triton SwiGLU ceiling (`56.0807 -> 78.0783 ms`). The useful conclusion is that this PyTorch/hipBLASLt surface prefers the larger batched expert GEMMs even with the repeated fixed-router activation materialization.

S4096 RMSNorm `saved-accum` retest on the pre-SwiGLU retained stack measured `96.7606 ms`, `42331.26 tokens/s`, and `218.819496 TFLOPs`, regressing the memory-light primary-shape result (`96.5923 ms`). Keep `saved-accum` as the S2048 default and a correctness-safe probe, but do not promote it for S4096 until a lower-level RMSNorm backward/statistic-passing kernel changes the tradeoff. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_5iter/summary.json`.

Cyclic fused-W13 MoE negative-control read: add `--moe-impl grouped-expert-batches-cyclic-uniform-fused-w13-dense-wgrad`, which preserves the fixed cyclic router contract but concatenates W1/W3 into one packed-sim `W13` projection for routed forward, W1/W3 wgrad, and routed input dgrad. Correctness passed on CUDA float64 and BF16 for `moe_expert_batches_cyclic_uniform_fused_w13_dense_wgrad_match`; outputs matched the loop routed MoE exactly, and watched gradients matched with max relative L2 `6.420649535745417e-17` in float64 and `0.0` in BF16. It regresses timing badly, so do not promote it.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward routed MoE ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cyclic fused-W13 dense-wgrad MoE, saved-accum | `B=1 S=2048` | `18.0405` | `53.6740` | `71.7978` | `28524.53` | `135.605099` | `8.8341` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_fused_w13_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_window_tail_sum_rmsnorm_saved_accum_5iter/summary.json` |
| Cyclic fused-W13 dense-wgrad MoE, memory-light | `B=1 S=4096` | `31.5226` | `81.7008` | `113.3559` | `36133.97` | `186.784332` | `9.6067` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_fused_w13_dense_wgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_window_tail_sum_5iter/summary.json` |

Read: keep the cyclic fused-W13 path only as a correctness-safe negative control. The forward routed-MoE block is neutral (`8.8341 ms` at S2048 and `9.6067 ms` at S4096), but backward regresses sharply versus the retained separate W1/W3 + `baddbmm` xgrad path (`39.4105 -> 53.6740 ms` at S2048 and `66.8305 -> 81.7008 ms` at S4096). Larger fused W13 batched GEMMs are less favorable in this ROCm/PyTorch packed-sim surface; the next MoE work should move toward a real packed-FP4 grouped training kernel rather than combining PyTorch `bmm` calls at this level.

Triton window-gradient reducer negative-control read: a separate variant replaces the PyTorch body/tail diagonal `grad_selected -> d_kv` reducer with a Triton reduction over `[B, S, window, D]`. The standalone reducer is numerically sound, but full-layer timing loses on both active shapes: S2048 regresses `58.7174 -> 59.0190 ms`, and S4096 regresses `102.3310 -> 102.4431 ms`. Keep it selectable for kernel-contract comparison only; the current PyTorch body/tail diagonal reducer is still better inside this PyTorch custom-autograd surface.

Attention-to-MoE residual/RMSNorm fusion negative control:

- Candidate: add `--boundary-impl fused-attn-residual-moe-norm`, which returns both the attention residual output and MoE RMSNorm input from one custom autograd boundary. This preserves the same forward contract as `hidden = hidden + attn_out; moe_in = ffn_norm(hidden)` and lets backward sum the residual and RMSNorm derivatives before returning identical gradients to the residual parent and attention branch.
- Correctness result, memory-light policy: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention_sparse_gather_match,layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm --rmsnorm-save-policy memory-light --boundary-impl fused-attn-residual-moe-norm --json`. Dense-mask vs selected-gradient sparse attention output max abs error was `5.551115123125783e-17`, max watched gradient relative L2 `2.1392777142553606e-16`; layer gradcheck passed; packed-sim matched fake FP4 exactly.
- Correctness result, saved-accum policy: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks layer,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm --rmsnorm-save-policy saved-accum --boundary-impl fused-attn-residual-moe-norm --json`. Layer gradcheck passed and packed-sim matched fake FP4 exactly.

| Variant | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward boundary ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Separate-boundary A/B baseline | `B=1 S=2048` | `19.0467` | `41.3623` | `60.4931` | `33855.12` | `160.946633` | `0.3917` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_rmsnorm_saved_accum_5iter/summary.json` |
| Fused residual + MoE RMSNorm | `B=1 S=2048` | `19.0447` | `41.3779` | `60.5067` | `33847.50` | `160.910396` | `0.3789` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_comp_sel_baddbmm_rmsnorm_saved_accum_fused_boundary_5iter/summary.json` |
| Separate-boundary A/B baseline | `B=1 S=4096` | `34.6534` | `72.2570` | `107.0457` | `38264.04` | `197.795162` | `0.6038` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_comp_sel_baddbmm_5iter/summary.json` |
| Fused residual + MoE RMSNorm | `B=1 S=4096` | `34.6890` | `72.5074` | `107.3291` | `38163.01` | `197.272916` | `0.5822` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_fused_boundary_5iter/summary.json` |

Read: keep the fused boundary selectable as a correctness-gated contract probe, but do not promote it. At S2048 it regresses by `0.0136 ms` versus the retained saved-accum baseline, and at S4096 it regresses by `0.2834 ms` versus the retained MoE `baddbmm-xgrad` baseline. The static boundary activation estimate is unchanged for these two retained configs (`340,924,480` bytes at S2048 and `546,445,440` bytes at S4096), so this PyTorch custom-autograd wrapping does not buy enough traffic reduction to matter. The next useful boundary work should be a lower-level RMSNorm statistic-passing kernel or fusing into the real sparse MLA/MoE kernels, not more Python-level residual wrappers.

MoE routed-input-gradient `baddbmm` negative control:

- Candidate: add `--moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad`, which keeps the cyclic uniform fixed-router contract but accumulates the W1/W3 routed input-gradient contributions with `torch.baddbmm(..., out=grad_x_group)` instead of materializing a second full `grad_x_group` bmm output followed by an add.
- Correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_cyclic_uniform_dense_wgrad_baddbmm_xgrad_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out --rmsnorm-save-policy memory-light --json`. Candidate vs loop MoE output max abs error `0.0`; max watched gradient relative L2 `6.420649535745417e-17`; packed-sim still matched fake FP4 exactly.
- S2048 candidate result: `61.2091 ms`, `33.46k tokens/s`, `159.063738 TFLOPs`; forward `19.2452 ms`, backward `41.8794 ms`, sparse MLA forward `5.1137 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_5iter/summary.json`.
- S4096 candidate result: `108.6763 ms`, `37.69k tokens/s`, `194.827319 TFLOPs`; forward `35.0122 ms`, backward `73.5297 ms`, sparse MLA forward `16.6602 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_baddbmm_xgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_5iter/summary.json`.
- Adjacent retained-path resample: S2048 `61.5885 ms`, `33.25k tokens/s`, `158.083963 TFLOPs`; S4096 `108.5289 ms`, `37.74k tokens/s`, `195.091944 TFLOPs`. Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_adjacent_baddbmm_baseline_5iter/summary.json` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_adjacent_baddbmm_baseline_5iter/summary.json`.
- Read: do not make the `baddbmm` path the primary default. It is a correctness-safe selectable variant and gives the best observed S2048 timing this turn, but it regresses the primary S4096 shape by `0.1474 ms` versus the adjacent retained-path resample. The retained primary path's adjacent S4096 resample is now the best recorded primary-shape PyTorch packed-sim result.

In-place compressed value/gradient accumulation negative control:

- Candidate: replace `out = out + compressed_value`, `grad_q = grad_q + compressed_value`, `grad_compressed = grad_compressed + compressed_grad`, and the compressed contribution to `softmax_dot` with in-place `add_` accumulation inside the split-compressed sparse MLA paths.
- Correctness result: pass under the direct-output no-KV-cat CUDA gate (`attention_sparse_gather_match`, `attention`, `layer`, `fp4_packed_sim_match`) before and after reverting the candidate.
- S2048 result: `61.1596 ms`, `33.49k tokens/s`, `159.192494 TFLOPs`, forward sparse MLA `5.1189 ms`; artifact `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_inplace_value_accum_5iter/summary.json`.
- S4096 results: `108.8252 ms` then `109.1799 ms` on resample versus retained best `108.6055 ms`; artifacts `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_inplace_value_accum_5iter/summary.json` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_split_comp_no_kv_cat_prealloc_bmm_out_inplace_value_accum_resample_5iter/summary.json`.
- Read: do not keep this candidate. It is slightly faster at S2048 but regresses the primary S4096 target on two samples, so the harness was reverted to the retained direct-output no-KV-cat path.

Sparse MLA backward score-gradient reuse negative control:

- Candidate: reuse the freshly allocated `grad_probs [B, H, S, ATTN_KEYS]` tensor as `grad_scores` via in-place `sub_`, `mul_`, and `mul_` after computing the softmax dot product, avoiding a separate score-gradient allocation in the custom backward.
- Correctness result: pass under the same CUDA gate as the retained in-place sparse MLA temp cleanup.
- S4096 timing result: forward `41.2776 ms`, backward `86.3754 ms`, total `127.7854 ms`, `32053.73 tokens/s`, `165.692704 TFLOPs`; forward sparse MLA `22.2867 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_reuse_grad_probs_scores_5iter/summary.json`.
- Read: do not keep this variant. It removes one conceptual temporary but serializes the PyTorch pointwise chain enough to regress backward by roughly `3 ms` versus the retained in-place-temp S4096 run. The real fix needs a fused softmax-backward/score-gradient kernel, not manual in-place mutation of PyTorch tensors.

Sparse MLA q-scaled score negative control:

- Candidate: move attention scale from the materialized score and score-gradient tensors to the query side: forward computes scores from `q * scale`, backward computes unscaled `grad_scores`, scales only `grad_q`, and uses the saved scaled query for selected-KV gradients. This is algebraically equivalent and reduces conceptual pointwise work on `[B, H, S, ATTN_KEYS]` tensors, but adds a q-sized multiply and changes PyTorch scheduling.
- Correctness result: pass under `python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer,fp4_layer_match,fp4_packed_sim_match --attention-impl sparse-gather-manual-bwd --rmsnorm-save-policy memory-light --json`. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002455961424857378`, max watched gradient relative L2 `0.3540374210961061`; packed-sim matched fake FP4 exactly.

| Run | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Retained explicit `bmm` contractions | `B=1 S=2048` | `20.4772` | `44.1962` | `64.7602` | `31624.34` | `150.341539` | `6.2590` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| Q-scaled scores candidate | `B=1 S=2048` | `20.5731` | `44.1967` | `64.8549` | `31578.17` | `150.122057` | `6.3063` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_q_scaled_scores_5iter/summary.json` |
| Retained explicit `bmm` contractions | `B=1 S=4096` | `40.4738` | `83.1777` | `123.7829` | `33090.19` | `171.050416` | `21.9863` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| Q-scaled scores candidate | `B=1 S=4096` | `40.6717` | `83.1903` | `124.0014` | `33031.89` | `170.749034` | `21.9811` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_q_scaled_scores_5iter/summary.json` |

Read: do not keep this variant in the PyTorch ceiling path. It is mathematically sound and correctness-safe, but timing is neutral-to-slower on both active shapes: `+0.0947 ms` at S2048 and `+0.2185 ms` at S4096. The retained explicit-`bmm` path still sets the current best. A fused sparse MLA kernel can place the scale in the best load/accumulation location without exposing extra PyTorch pointwise scheduling.

Sparse MLA flatten-first score-bmm negative control:

- Candidate: change `_sparse_scores_bmm` from transposing `selected [B, S, K, D]` before flattening to flattening first as `[B*S, K, D]` and passing `selected_flat.transpose(1, 2)` directly to `torch.bmm`. The goal was to let the bmm consume a per-batch transposed operand instead of forcing a larger pre-flatten transpose/copy.
- Correctness result: pass under the same CUDA gate as the q-scaled-score candidate. Attention and layer gradcheck passed; FP4 layer match output max abs error `0.002455961424857378`, max watched gradient relative L2 `0.3540374210961061`; packed-sim matched fake FP4 exactly.

| Run | Shape | Forward ms | Backward ms | Total ms | Tokens/s | Est. TFLOPs | Forward `attn.sparse_mla` ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Retained explicit `bmm` contractions | `B=1 S=2048` | `20.4772` | `44.1962` | `64.7602` | `31624.34` | `150.341539` | `6.2590` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| Flatten-first score bmm candidate | `B=1 S=2048` | `20.4431` | `44.3554` | `64.8841` | `31563.99` | `150.054632` | `6.2470` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_flatten_first_score_bmm_5iter/summary.json` |
| Retained explicit `bmm` contractions | `B=1 S=4096` | `40.4738` | `83.1777` | `123.7829` | `33090.19` | `171.050416` | `21.9863` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_bmm_contract_5iter/summary.json` |
| Flatten-first score bmm candidate | `B=1 S=4096` | `40.5371` | `83.3320` | `124.0021` | `33031.69` | `170.747970` | `21.9307` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_cyclic_uniform_dense_wgrad_sparse_gather_manual_bwd_flatten_first_score_bmm_5iter/summary.json` |

Read: do not keep this variant. It shaves a tiny amount from the forward sparse MLA block but regresses backward enough to lose the full iteration on both active shapes: `+0.1239 ms` at S2048 and `+0.2193 ms` at S4096. The retained score helper still materializes too much, but this PyTorch-level operand-stride tweak is not the fix.

Contiguous key-probability negative control:

- Candidate: after softmax over `[ATTN_KEYS+1]`, save `probs_all[..., :-1].contiguous()` plus a separate contiguous sink-prob tensor before the value einsum/backward. Correctness passed under the same CUDA gate as above.
- S2048 command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_contig_probs_5iter`
- Result: `66.4162 ms`, `30.84k tokens/s`, `146.593014 TFLOPs`; forward sparse MLA worsened to `6.7365 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_contig_probs_5iter/summary.json`.
- Read: do not keep this variant. The explicit contiguous probability copy costs more than it saves in PyTorch einsum/backward at S2048, so the retained code keeps the sink-extended `probs_all` save and slices key probabilities as views.

BF16-save Q/KV negative control:

- Candidate: save original BF16 `q` and gathered `kv_selected` tensors in the manual sparse MLA context instead of saving their FP32 accumulation casts, then recast them to accumulation dtype inside backward. Correctness passed under the same CUDA gate as above.
- S2048 command: `python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-uniform-dense-wgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_save_bf16_5iter`
- Result: `66.7501 ms`, `30.68k tokens/s`, `145.859719 TFLOPs`; backward worsened to `45.9636 ms`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_uniform_dense_wgrad_sparse_gather_manual_bwd_save_bf16_5iter/summary.json`.
- Read: do not keep this variant in the PyTorch ceiling path. It reduces saved activation footprint on paper, but the large backward recasts dominate. A lower-level fused sparse MLA kernel can choose a better load/cast schedule without paying PyTorch tensor materialization overhead.

## S4096 ROCTx/Rocprof Selected-Region Profile

`profile_microbench.py` now has `--roctx-block-ranges` and `--roctx-profile-attribution`. The first flag emits ROCTx ranges around measured phases, forward blocks, and segmented backward attribution calls. The second flag brackets the measured segmented backward attribution loop with `roctxProfilerResume/Pause`, so `rocprofv3 --selected-regions` can collect the attribution pass without setup/warmup noise.

Forward selected-region command:

```bash
rocprofv3 --kernel-trace --marker-trace --kernel-rename --stats --summary --summary-output-file profiles/rocprof_s4096_qnorm_best_region_1iter_block_ranges/rocprof_summary.txt --summary-units msec --selected-regions -d profiles/rocprof_s4096_qnorm_best_region_1iter_block_ranges -o rocprof -f csv -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --roctx-profile-region --roctx-block-ranges --trace-dir profiles/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_rocprof_block_ranges_1iter
```

Result: total `92.7382 ms`, forward `31.4152 ms`, backward `61.1751 ms`, `44.17k tokens/s`, `228.310612 TFLOPs`. The renamed forward ROCTx ranges align with the normal block timers: `attn.sparse_mla` `15.0740 ms`, `moe.routed_experts` `9.6304 ms`, `attn.q_proj` `1.9302 ms`, `attn.out_proj` `1.0705 ms`. Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_s4096_qnorm_best_region_1iter_block_ranges/` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_rocprof_block_ranges_1iter/summary.json`.

Backward attribution selected-region command:

```bash
rocprofv3 --kernel-trace --marker-trace --kernel-rename --stats --summary --summary-output-file profiles/rocprof_s4096_qnorm_best_backward_attr_block_ranges_sync_1iter/rocprof_summary.txt --summary-units msec --selected-regions -d profiles/rocprof_s4096_qnorm_best_backward_attr_block_ranges_sync_1iter -o rocprof -f csv -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --profile-backward-blocks --profile-attention-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --roctx-profile-attribution --roctx-block-ranges --trace-dir profiles/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_rocprof_backward_attr_block_ranges_sync_1iter
```

Result: normal timing in the same run was total `91.9030 ms`, forward `30.7249 ms`, backward `61.0385 ms`, `44.57k tokens/s`, `230.385541 TFLOPs`. Segmented attribution again ranks sparse MLA and routed MoE as the first targets: `attention.sparse_mla.dgrad_qkv` `25.3717 ms`, `attention.sparse_mla.sink_wgrad` `24.6248 ms`, `moe.routed.dgrad` `26.1691 ms`, and `moe.routed.wgrad` `26.3659 ms`. The ROCTx marker trace now records block-duration ranges that match these timers, but `rocprofv3 --kernel-rename` still collapses many backward kernels to the outer `backward_attribution` range; use `rocprof_marker_api_trace.csv` for block boundaries and `rocprof_kernel_stats.csv` for raw kernel mix. Top raw kernel groups in the attribution region were hipBLASLt GEMM families: `47.6332 ms` across 9 calls, `35.2168 ms` across 8 calls, `29.0220 ms` across 9 calls, then `_softmax_bwd_scores_kernel` `6.0292 ms` across 4 calls and `_head_rmsnorm_backward_kernel` `3.2478 ms` across 4 calls. Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_s4096_qnorm_best_backward_attr_block_ranges_sync_1iter/` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_rocprof_backward_attr_block_ranges_sync_1iter/summary.json`.

Raw-kernel backward attribution rerun command, executed inside container `b9d33e6e8227` from `/local/data/sonle5/dsv4_amd_sonle5_port_pr23882_20260510/tmp/dsv4_layer_microbench_20260530`:

```bash
rocprofv3 --kernel-trace --marker-trace --stats --summary --summary-output-file profiles/rocprof_s4096_qnorm_best_backward_attr_blocks_raw_docker_1iter/rocprof_summary.txt --summary-units msec --selected-regions -d profiles/rocprof_s4096_qnorm_best_backward_attr_blocks_raw_docker_1iter -o rocprof -f csv -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --profile-backward-blocks --profile-attention-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --roctx-profile-attribution --roctx-block-ranges --trace-dir profiles/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_rocprof_backward_attr_blocks_raw_docker_1iter
```

Result: normal timing in the same raw-kernel run was total `91.9555 ms`, forward `30.7474 ms`, loss `0.1410 ms`, backward `61.0671 ms`, `44.54k tokens/s`, `230.253903 TFLOPs`.

Added `dsv4_layer_microbench/summarize_rocprof_blocks.py` to map `rocprof_kernel_trace.csv` rows onto ROCTx block ranges by timestamp. Summary command:

```bash
python3 dsv4_layer_microbench/summarize_rocprof_blocks.py --kernel-trace profiles/rocprof_s4096_qnorm_best_backward_attr_blocks_raw_docker_1iter/rocprof_kernel_trace.csv --marker-trace profiles/rocprof_s4096_qnorm_best_backward_attr_blocks_raw_docker_1iter/rocprof_marker_api_trace.csv --top-blocks 12 --top-kernels 6 --summary-path profiles/rocprof_s4096_qnorm_best_backward_attr_blocks_raw_docker_1iter/block_kernel_summary.json
```

Block/kernel summary: total selected-region kernel duration was `326.0237 ms`; `212.0759 ms` mapped to explicit ROCTx blocks and `113.9478 ms` remained outside the current inner block labels. The total exceeds normal backward phase time because this is the segmented attribution loop with repeated `autograd.grad` calls; use the per-block rows to rank kernel families, not as a phase-time sum.

| Block | Marker ms | Kernel ms | Main kernel read |
| --- | ---: | ---: | --- |
| `backward.attention.wgrad` | `32.2535` | `32.0477` | Three large hipBLASLt GEMM families dominate (`8.9031`, `5.4093`, `5.1217 ms`); `_softmax_bwd_scores_kernel` is `1.5393 ms`. |
| `backward.attention.dgrad` | `31.7076` | `31.5537` | Same sparse-MLA/GEMM family mix; `_softmax_bwd_scores_kernel` is `1.5145 ms`. |
| `backward.moe.routed.wgrad` | `26.4283` | `26.2646` | Two grouped-BMM hipBLASLt families account for `15.6968 + 9.7879 ms`; pointwise/reduce kernels are secondary. |
| `backward.moe.routed.dgrad` | `26.0039` | `25.8758` | Two grouped-BMM hipBLASLt families account for `15.3940 + 9.7784 ms`; SwiGLU backward is only `0.1195 ms`. |
| `backward.attention.sparse_mla.dgrad_qkv` | `25.0047` | `24.9117` | Dominated by three large GEMM families (`8.7622`, `5.3334`, `5.1938 ms`) plus `_softmax_bwd_scores_kernel` `1.4899 ms`. |
| `backward.attention.sparse_mla.sink_wgrad` | `24.8963` | `24.7866` | Same sparse-MLA GEMM family mix; sink-gradient arithmetic itself is not the bottleneck. |
| `backward.moe.routed.w1_w3.dgrad` | `6.6551` | `6.4848` | Two W1/W3 input-gradient GEMMs account for `6.3188 ms`. |
| `backward.moe.routed.w1.wgrad` | `5.7434` | `5.6515` | Single grouped-BMM wgrad kernel. |
| `backward.moe.routed.w2.wgrad` | `5.3340` | `5.2335` | Single grouped-BMM wgrad kernel. |
| `backward.moe.routed.w3.wgrad` | `5.1211` | `5.0456` | Single grouped-BMM wgrad kernel. |
| `backward.moe.routed.w2.dgrad` | `3.2634` | `3.1412` | Single grouped-BMM dgrad kernel. |

Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_s4096_qnorm_best_backward_attr_blocks_raw_docker_1iter/block_kernel_summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_s4096_qnorm_best_backward_attr_blocks_raw_docker_1iter/`, and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_rocprof_backward_attr_blocks_raw_docker_1iter/summary.json`.

Read: the next high-leverage work is still real packed-FP4 grouped MoE training GEMM and a fused sparse-MLA kernel. Python/Triton pointwise tweaks around SwiGLU, sink-gradient accumulation, or softmax backward alone cannot plausibly close the remaining `4.82 ms` S4096 gap because the dominant rows are hipBLASLt GEMM families inside routed MoE and sparse MLA.

## Cyclic MoE W13-Wgrad Negative Control

Added `grouped-expert-batches-cyclic-uniform-dense-w13-wgrad-baddbmm-xgrad` as a selectable MoE variant. It keeps the retained cyclic-uniform forward and retained `baddbmm` routed xgrad path, but computes `d_w1` and `d_w3` through one larger grouped `bmm` over concatenated `grad_gate/grad_up`, then splits the result back into separate W1/W3 gradients. This isolates the W1/W3 wgrad fusion question from the older full fused-W13 variant, which also changed forward and xgrad.

Correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_cyclic_uniform_dense_w13_wgrad_baddbmm_xgrad_match --json
```

Result: pass. Output max abs error `0.0`, output relative L2 `0.0`, max watched gradient relative L2 `6.449652832387695e-17`.

S2048 timing command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-w13-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_w13_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_memory_light_5iter
```

Result: `68.8743 ms`, forward `17.1809 ms`, loss `0.0914 ms`, backward `51.6020 ms`, `29.74k tokens/s`, `141.361132 TFLOPs`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_w13_wgrad_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_memory_light_5iter/summary.json`.

Read: do not retain this variant for the performance ceiling. It is correctness-safe, but it regresses S2048 by `14.98 ms` versus the retained `53.8940 ms` path. The single larger `M=6144, N=7168, K=32` grouped wgrad shape loses badly enough that a S4096 run is not worth spending.

## Cyclic MoE BMM-Out Negative Control

Added `grouped-expert-batches-cyclic-uniform-dense-wgrad-bmm-out-baddbmm-xgrad` as a selectable MoE variant. It keeps the retained cyclic-uniform math and grouped-BMM shapes, but uses explicit `torch.bmm(..., out=...)` buffers for gate/up/y forward products and for the backward `grad_hidden`, `grad_w1`, `grad_w2`, `grad_w3`, and first routed-xgrad product.

Correctness command:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_cyclic_uniform_dense_wgrad_bmm_out_baddbmm_xgrad_match --json
```

Result: pass. Output max abs error `0.0`, output relative L2 `0.0`, max watched gradient relative L2 `6.449652832387695e-17`.

S2048 timing command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-bmm-out-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_bmm_out_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_memory_light_5iter
```

S2048 result: `54.1384 ms`, forward `17.2781 ms`, loss `0.0923 ms`, backward `36.7680 ms`, `37.83k tokens/s`, `179.838268 TFLOPs`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_bmm_out_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_memory_light_5iter/summary.json`.

S4096 timing command:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-bmm-out-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_bmm_out_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_5iter
```

S4096 result: `91.8826 ms`, forward `30.1025 ms`, loss `0.1384 ms`, backward `61.6417 ms`, `44.58k tokens/s`, `230.436510 TFLOPs`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_cyclic_uniform_dense_wgrad_bmm_out_baddbmm_xgrad_comp_sel_baddbmm_triton_softmax_fwd_bwd_fast_triton_rope_outplace_kernel_rmsnorm_saved_accum_5iter/summary.json`.

Read: do not retain this variant. The explicit `out=` buffers are correctness-safe, but they are neutral-to-slower: S2048 regresses by `0.2443 ms` versus retained `53.8940 ms`, and S4096 regresses by `0.4579 ms` versus retained `91.4248 ms`. Allocation removal is not enough to beat the library-selected grouped-BMM path here; keep the simpler retained cyclic path.

## AITER FP4 MoE Forward Probe

Added `dsv4_layer_microbench/probe_aiter_moe_a4w4.py` as an AITER FlyDSL FP4x2/MXFP4 routed-MoE forward probe. This is not the benchmark training path: AITER uses per-1x32 MXFP4/FP4x2 scales here, while the layer harness target remains E2M1/UE8M0 128x128 packed-sim with backward and wgrad coverage. The probe is useful for checking whether an available ROCm kernel family can cover the dominant DSv4 routed MoE forward shape.

Correctness smoke, using AITER's own top-k router:

```bash
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 64 --model-dim 512 --inter-dim 256 --experts 16 --topk 4 --block-m 32 --router aiter-topk --warmup 1 --iters 2 --json --summary-path profiles/aiter_moe_a4w4_smoke_topk_t64_d512_i256_e16_k4/summary.json
```

Result: pass. Stage1 max abs `0.0`, stage2 max abs `0.0009765625`, e2e max abs `0.0009765625`, all `100.0%` close at `atol=1.0`, `rtol=0.08`. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_smoke_topk_t64_d512_i256_e16_k4/summary.json`.

Balanced-router smoke command:

```bash
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 64 --model-dim 512 --inter-dim 256 --experts 16 --topk 4 --block-m 32 --router balanced --warmup 1 --iters 2 --json --summary-path profiles/aiter_moe_a4w4_smoke_t64_d512_i256_e16_k4/summary.json
```

Result: fail. Stage1 max abs was `5.233835233403106e+36`, `0.0%` close. Stage2/e2e pass under the broad smoke tolerance but both reported `rel_l2=1.0`, so the balanced-router probe is invalid for promotion. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_smoke_t64_d512_i256_e16_k4/summary.json`.

Real-shape reference-quantized command:

```bash
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 2048 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 32 --router aiter-topk --warmup 1 --iters 2 --json --summary-path profiles/aiter_moe_a4w4_topk_t2048_d7168_i3072_e384_k6_bm32/summary.json
```

Result: OOM during BF16-to-MXFP4 quantization of W13 before timed kernels. The failing allocation was `15.75 GiB` with PyTorch already holding `272.21 GiB`, so full BF16 expert weights plus reference quantization is too memory-heavy for the real 384-expert shape in this probe.

To keep the real-shape timing path memory-bounded, the probe now has `--data-mode synthetic-packed`, which directly generates FP4x2 packed tensors and E8M0 scale bytes, skips BF16 references, and reports `errors=null`. This mode proves kernel launch/perf shape viability only; it does not prove numerical correctness.

Real-shape synthetic timing commands:

```bash
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 2048 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 32 --router aiter-topk --data-mode synthetic-packed --warmup 1 --iters 2 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm32/summary.json
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 2048 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 64 --router aiter-topk --data-mode synthetic-packed --warmup 1 --iters 3 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm64/summary.json
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 2048 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 128 --router aiter-topk --data-mode synthetic-packed --warmup 1 --iters 3 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm128/summary.json
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 2048 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 64 --router aiter-topk --data-mode synthetic-packed --mode atomic --warmup 2 --iters 10 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm64_10iter/summary.json
```

Timing result:

| Probe | Stage1 ms | Stage2 ms | E2E ms | Stage1 TFLOPs | Stage2 TFLOPs | E2E TFLOPs | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `block_m=32`, atomic | `2.4856` | `1.4499` | `5.2022` | `435.44` | `373.23` | `312.08` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm32/summary.json` |
| `block_m=64`, atomic, 3 iters | `1.9628` | `1.1281` | `4.6507` | `551.42` | `479.71` | `349.09` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm64/summary.json` |
| `block_m=128`, atomic | `4.3329` | `1.5913` | `7.2017` | `249.80` | `340.07` | `225.43` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm128/summary.json` |
| `block_m=64`, atomic, 10 iters | `1.9973` | `1.1177` | `4.3779` | `541.89` | `484.17` | `370.84` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t2048_d7168_i3072_e384_k6_bm64_10iter/summary.json` |

Primary-shape S4096 synthetic timing commands:

```bash
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 4096 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 32 --router aiter-topk --data-mode synthetic-packed --mode atomic --warmup 2 --iters 10 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm32_10iter/summary.json
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 4096 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 64 --router aiter-topk --data-mode synthetic-packed --mode atomic --warmup 2 --iters 10 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm64_10iter/summary.json
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 4096 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 128 --router aiter-topk --data-mode synthetic-packed --mode atomic --warmup 2 --iters 10 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm128_10iter/summary.json
python3 dsv4_layer_microbench/probe_aiter_moe_a4w4.py --tokens 4096 --model-dim 7168 --inter-dim 3072 --experts 384 --topk 6 --block-m 64 --router aiter-topk --data-mode synthetic-packed --mode reduce --warmup 2 --iters 10 --json --summary-path profiles/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm64_reduce_10iter/summary.json
```

S4096 timing result:

| Probe | Stage1 ms | Stage2 ms | E2E ms | Stage1 TFLOPs | Stage2 TFLOPs | E2E TFLOPs | Artifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `block_m=32`, atomic | `2.8851` | `2.0393` | `7.7016` | `750.28` | `530.73` | `421.60` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm32_10iter/summary.json` |
| `block_m=64`, atomic | `2.5215` | `1.5608` | `6.6730` | `858.47` | `693.44` | `486.59` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm64_10iter/summary.json` |
| `block_m=128`, atomic | `3.9868` | `1.6213` | `8.3731` | `542.96` | `667.56` | `387.79` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm128_10iter/summary.json` |
| `block_m=64`, reduce | `2.5729` | `1.5749` | `6.6811` | `841.33` | `687.24` | `486.00` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_moe_a4w4_synthetic_topk_t4096_d7168_i3072_e384_k6_bm64_reduce_10iter/summary.json` |

Additional sweep notes: `block_m=16` is rejected by AITER scale sorting (`assert block_size % BLOCK_SIZE_M == 0`), and `block_m=64 --mode reduce` measured e2e `4.6741 ms` at S2048 and `6.6811 ms` at S4096, slightly slower than atomic. Read: AITER's forward-only FP4x2 routed MoE kernels are shape-viable and fast at DSv4 dimensions, with `block_m=64` the current best setting at both S2048 and S4096. They still need a real integration decision because they do not cover the current 128x128 FP4 training contract or backward/wgrad path.

## AITER BF16 GMM Cyclic MoE Negative Control

Added `dsv4_layer_microbench/probe_aiter_gmm_cyclic_moe.py` to compare AITER Triton BF16 grouped-GEMM wrappers (`gmm`, `ptgmm`, `nptgmm`) against the exact cyclic-uniform routed MoE BMM contractions used by the current packed-sim ceiling path. This probes a lower-risk integration idea than AITER FP4 forward-only MoE: replace the existing grouped BF16 `torch.bmm` contractions in forward, dgrad, and wgrad without changing the 128x128 FP4 packed-sim training contract.

Smoke command:

```bash
python3 dsv4_layer_microbench/probe_aiter_gmm_cyclic_moe.py --tokens 64 --hidden 128 --intermediate 64 --experts 8 --topk 2 --warmup 1 --iters 1 --json --summary-path profiles/aiter_gmm_cyclic_moe_smoke_t64_h128_i64_e8_k2/summary.json
```

Smoke result: pass. All AITER GMM/TGMM outputs were `100.0%` close to the `torch.bmm` references. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_gmm_cyclic_moe_smoke_t64_h128_i64_e8_k2/summary.json`.

Real-shape commands:

```bash
python3 dsv4_layer_microbench/probe_aiter_gmm_cyclic_moe.py --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 1 --iters 3 --json --summary-path profiles/aiter_gmm_cyclic_moe_t2048_h7168_i3072_e384_k6_3iter/summary.json
python3 dsv4_layer_microbench/probe_aiter_gmm_cyclic_moe.py --tokens 4096 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 1 --iters 3 --json --summary-path profiles/aiter_gmm_cyclic_moe_t4096_h7168_i3072_e384_k6_3iter/summary.json
```

Correctness result: pass for both active shapes. Forward and W2 dgrad were exact vs `torch.bmm`; combined W1/W3 xgrad had max abs `0.25`, relative L2 about `0.00247`, and `100.0%` close; S4096 TGMM wgrad had max abs `0.015625`, relative L2 about `9.1e-6`, and `100.0%` close.

Timing result:

| Shape | Contraction | `torch.bmm` ms | Best AITER ms | `torch.bmm` TFLOPs | Best AITER TFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| `B=1 S=2048` | forward W1-like | `3.6730` | `7.4460` | `147.34` | `72.68` |
| `B=1 S=2048` | W2 dgrad | `3.9304` | `7.2821` | `137.69` | `74.31` |
| `B=1 S=2048` | W1+W3 xgrad | `6.4388` | `14.5103` | `168.09` | `74.59` |
| `B=1 S=2048` | W2 wgrad | `4.9525` | `6.2792` | `109.27` | `86.18` |
| `B=1 S=2048` | W1+W3 wgrad | `8.9324` | `11.5627` | `121.17` | `93.61` |
| `B=1 S=4096` | forward W1-like | `4.1860` | `7.5416` | `258.56` | `143.52` |
| `B=1 S=4096` | W2 dgrad | `4.2639` | `7.3152` | `253.83` | `147.96` |
| `B=1 S=4096` | W1+W3 xgrad | `7.0349` | `14.6573` | `307.70` | `147.68` |
| `B=1 S=4096` | W2 wgrad | `5.5212` | `6.7703` | `196.03` | `159.87` |
| `B=1 S=4096` | W1+W3 wgrad | `10.0673` | `13.0971` | `215.02` | `165.28` |

Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_gmm_cyclic_moe_t2048_h7168_i3072_e384_k6_3iter/summary.json` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_gmm_cyclic_moe_t4096_h7168_i3072_e384_k6_3iter/summary.json`.

Read: do not integrate AITER BF16 GMM/TGMM into the current cyclic packed-sim MoE path. It is correctness-clean, but slower than the existing hipBLASLt-backed `torch.bmm` contractions on every measured S2048 and S4096 forward/dgrad/wgrad subcase. The promising AITER surface remains the forward-only FP4x2/MXFP4 MoE probe, but training progress still needs either a real packed-FP4 backward/wgrad path or a fused sparse MLA kernel rather than swapping in AITER BF16 GMM wrappers.

## BF16-Contract Sparse MLA Attention

Added a selectable sparse MLA variant:

`sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum`

This keeps the existing FP32 score/probability softmax path, but when the layer input is BF16/FP16 it runs the sparse score/value and compressed score/value BMM contractions in the input dtype. Float64 gradcheck falls back to the original accumulation dtype.

Correctness/smoke commands:

```bash
python3 -m py_compile dsv4_layer_microbench/reference_layer.py dsv4_layer_microbench/profile_microbench.py dsv4_layer_microbench/run_microbench.py dsv4_layer_microbench/check_correctness.py
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks attention,layer --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --rmsnorm-save-policy memory-light --json
```

Correctness result: pass. Float64 attention and layer gradcheck both passed. A BF16 tiny A/B against the retained qnorm baseline with identical weights and fixed router was finite with output max abs `0.00390625`, output rel L2 `0.000155`, input-grad max abs `1.5259e-05`, input-grad rel L2 `0.000397`, and all-parameter-grad rel L2 `0.00528` with worst max-abs parameter `attention.wo_a`.

Timing commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy memory-light --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s4096_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 4096 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --profile-backward-blocks --profile-attention-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --trace-dir profiles/full_b1s4096_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_attr_1iter
```

Timing result:

| Shape | RMSNorm policy | Total ms | Target ms | Tokens/s | TFLOPs | Forward ms | Backward ms | Sparse MLA fwd ms | Routed MoE fwd ms | Artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B=1 S=2048` | `memory-light` | `47.2691` | `43.3` | `43,326.39` | `205.9728` | `15.1985` | `31.9785` | `2.3923` | `8.9055` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_5iter/summary.json` |
| `B=1 S=2048` | `saved-accum` | `47.0589` | `43.3` | `43,519.91` | `206.8929` | `14.9830` | `31.9795` | `2.3713` | `8.7475` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_5iter/summary.json` |
| `B=1 S=4096` | `saved-accum` | `68.2089` | `86.6` | `60,050.82` | `310.4158` | `22.5770` | `45.4889` | `6.9870` | `9.6633` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_5iter/summary.json` |
| `B=1 S=4096` attribution | `saved-accum` | `68.5276` | `86.6` | `59,771.53` | `308.9720` | `23.2344` | `45.1513` | `6.9909` | `9.6852` | `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s4096_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_attr_1iter/summary.json` |

S4096 segmented backward attribution from the 1-iter run:

| Block | ms |
| --- | ---: |
| `moe.routed.wgrad` | `26.4905` |
| `moe.routed.dgrad` | `26.3351` |
| `attention.wgrad` | `15.7438` |
| `attention.dgrad` | `15.3536` |
| `attention.sparse_mla.sink_wgrad` | `8.8608` |
| `attention.sparse_mla.dgrad_qkv` | `8.8394` |
| `moe.routed.w1_w3.dgrad` | `6.5695` |
| `moe.routed.w2.wgrad` | `5.9223` |
| `moe.routed.w1.wgrad` | `5.7928` |
| `moe.routed.w3.wgrad` | `5.1172` |
| `attention.q_path.dgrad` | `3.2536` |
| `moe.routed.w2.dgrad` | `3.2013` |
| `attention.q_path.wgrad` | `3.1399` |

S2048 retained-path MoE sub-block attribution from `full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_moe_attr_1iter`:

| Block | ms |
| --- | ---: |
| `moe.routed.wgrad` | `23.3269` |
| `moe.routed.dgrad` | `23.0063` |
| `attention.wgrad` | `7.0456` |
| `attention.dgrad` | `6.9001` |
| `moe.routed.w1_w3.dgrad` | `6.0621` |
| `moe.routed.w2.wgrad` | `5.0061` |
| `moe.routed.w1.wgrad` | `4.8256` |
| `moe.routed.w3.wgrad` | `4.4456` |
| `moe.routed.w2.dgrad` | `3.0707` |

Read: promote this BF16-contract sparse MLA variant as the current packed-sim ceiling path. It beats the prior retained S4096 path by `23.216 ms` and clears the `86.6 ms` target by `18.391 ms`; the best S2048 policy is now `saved-accum` at `47.0589 ms`, `3.759 ms` above the `43.3 ms` development target. The next bottleneck is routed MoE backward/wgrad rather than attention forward.

S2048 grouped-top-k routed wgrad negative-control commands:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-topk-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_topk_wgrad_baddbmm_xgrad_saved_accum_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-topk-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy saved-accum --profile-backward-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_topk_wgrad_baddbmm_xgrad_saved_accum_attr_1iter
```

Result: correctness-safe but slower. The 5-iter timing regressed to `60.4491 ms`, `33,879.76 tokens/s`, `161.0637 TFLOPs`; forward stayed flat at `14.9681 ms`, but backward rose to `45.3833 ms`. The 1-iter attribution measured routed MoE dgrad `33.1273 ms` and routed MoE wgrad `33.0679 ms`, versus the retained adjacent S2048 baseline around `23.1 ms` each. Artifact paths: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_topk_wgrad_baddbmm_xgrad_saved_accum_5iter/summary.json` and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_topk_wgrad_baddbmm_xgrad_saved_accum_attr_1iter/summary.json`.

## S2048 BF16-Contract Current-Best rocprof

Selected-region rocprof command shape:

```bash
/opt/rocm/bin/rocprofv3 --selected-regions --marker-trace --kernel-trace --memory-copy-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_region_s2048_bf16_contract_current_1iter/summary.txt --output-directory profiles/rocprof_region_s2048_bf16_contract_current_1iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --roctx-profile-region --roctx-block-ranges --trace-dir profiles/rocprof_region_s2048_bf16_contract_current_1iter/torch_summary
```

Timed-loop result under rocprof: total `49.0831 ms`, forward `16.3784 ms`, loss `0.1132 ms`, backward `32.5916 ms`, `41,725.12 tokens/s`, `198.3604 TFLOPs`. rocprof attributed `46.5275 ms` of `46.5474 ms` kernel time to ROCTX blocks. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_region_s2048_bf16_contract_current_1iter/`.

Top retained-path kernel evidence:

| Block | Marker ms | Kernel ms | Dispatches | Dominant kernel families |
| --- | ---: | ---: | ---: | --- |
| `phase.backward` | `32.5862` | `32.4072` | `242` | routed-MoE GEMM families: `MT256x256x32` `13.5905 ms` over 3 launches, `MT256x32x64` `9.1007 ms` over 3 launches |
| `forward.moe.routed_experts` | `9.0876` | `8.9823` | `8` | forward routed-MoE GEMM family `MT256x32x128` `8.6627 ms` over 3 launches |
| `forward.attn.sparse_mla` | `2.4180` | `2.2970` | `14` | `_masked_softmax_inplace_kernel` `0.6911 ms`; remaining GEMMs each below `0.30 ms` |

Backward attribution rocprof command shape:

```bash
/opt/rocm/bin/rocprofv3 --selected-regions --marker-trace --kernel-trace --stats --summary --summary-units msec --summary-output-file profiles/rocprof_attr_s2048_bf16_contract_current_moe_blocks_1iter/summary.txt --output-directory profiles/rocprof_attr_s2048_bf16_contract_current_moe_blocks_1iter --output-file trace --output-format csv json -- python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 1 --iters 1 --fixed-router --rmsnorm-save-policy saved-accum --profile-backward-blocks --profile-moe-blocks --backward-attribution-warmup 1 --backward-attribution-iters 1 --no-torch-trace --roctx-profile-attribution --roctx-block-ranges --trace-dir profiles/rocprof_attr_s2048_bf16_contract_current_moe_blocks_1iter/torch_summary
```

Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/rocprof_attr_s2048_bf16_contract_current_moe_blocks_1iter/`. The attribution profile includes warmup/normal-loop kernels outside the attribution ROCTX region, so use attributed block rows for interpretation rather than the full trace total.

| Block | Marker ms | Kernel ms | Dominant kernel evidence |
| --- | ---: | ---: | --- |
| `backward.moe.routed.dgrad` | `23.3882` | `23.2749` | `MT256x256x32` `13.7930 ms` over 3 launches, `MT256x32x64` `9.1188 ms` over 3 launches |
| `backward.moe.routed.wgrad` | `23.1202` | `22.9811` | `MT256x256x32` `13.4905 ms` over 3 launches, `MT256x32x64` `9.1349 ms` over 3 launches |
| `backward.moe.routed.w1_w3.dgrad` | `6.0694` | `5.9803` | `MT256x32x64` `5.9028 ms` over 2 launches |
| `backward.moe.routed.w2.wgrad` | `5.0291` | `4.9484` | one `MT256x256x32` launch |
| `backward.moe.routed.w1.wgrad` | `4.7967` | `4.7206` | one `MT256x256x32` launch |
| `backward.moe.routed.w3.wgrad` | `4.3745` | `4.2880` | one `MT256x256x32` launch |
| `backward.moe.routed.w2.dgrad` | `3.0558` | `2.9526` | one `MT256x32x64` launch |

Read: the remaining S2048 gap is GEMM-bound inside the routed-MoE training contractions. Pointwise work, routing gather/reduce, SwigLU, copies, and sparse-MLA forward are too small to close the `3.76 ms` gap. The next useful optimization should target a real packed-FP4 grouped MoE training GEMM or a ROCm-library/CK/TileLang replacement for the specific routed MoE backward/wgrad GEMM shapes; more PyTorch-level routing/layout tweaks are now low expected value unless they change the GEMM shapes substantially.

## TunableOp Routed-MoE BMM Probe

Added `dsv4_layer_microbench/probe_tunableop_bmm.py` to isolate the exact cyclic routed-MoE BF16 BMM shapes without the full layer, AITER comparison path, or correctness tensor reductions. It can run the same shape family with `torch.cuda.tunable` enabled and optionally with tuning.

S2048 baseline command:

```bash
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --json --summary-path profiles/tunableop_bmm_all_s2048_baseline/summary.json
```

S2048 TunableOp enabled/no-tuning command:

```bash
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --tunable --json --summary-path profiles/tunableop_bmm_all_s2048_enabled_notuning/summary.json
```

| Op | A shape | B shape | Baseline median ms | Baseline TFLOPs | Tunable enabled/no-tune median ms | Tunable enabled/no-tune TFLOPs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `fwd_w1` | `[384, 32, 7168]` | `[384, 7168, 3072]` | `3.0544` | `177.18` | `3.0091` | `179.84` |
| `w2_dgrad` | `[384, 32, 7168]` | `[384, 7168, 3072]` | `2.9977` | `180.52` | `2.9349` | `184.39` |
| `w1w3_xgrad` | `[384, 32, 3072]` | `[384, 3072, 7168]` twice | `6.2161` | `174.12` | `6.2346` | `173.60` |
| `w2_wgrad` | `[384, 7168, 32]` | `[384, 32, 3072]` | `3.5913` | `150.69` | `3.5285` | `153.37` |
| `w1_wgrad` | `[384, 3072, 32]` | `[384, 32, 7168]` | `3.5636` | `151.86` | `3.5664` | `151.74` |
| `w3_wgrad` | `[384, 3072, 32]` | `[384, 32, 7168]` | `3.6354` | `148.86` | `3.5405` | `152.85` |

Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/tunableop_bmm_all_s2048_baseline/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/tunableop_bmm_all_s2048_enabled_notuning/summary.json`, and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/tunableop_bmm_w2_wgrad_s2048_baseline/summary.json`.

Inline tuning negative controls:

```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=10 PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=50 PYTORCH_TUNABLEOP_FILENAME=profiles/tunableop_s2048_bf16_contract_current.csv python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_tunableop_10iter_50ms
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op w2_wgrad --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --tunable --tune --max-tuning-iters 1 --max-tuning-duration-ms 1 --tunable-file profiles/tunableop_bmm_w2_wgrad_s2048_max1/tunable.csv --json --summary-path profiles/tunableop_bmm_w2_wgrad_s2048_max1/summary.json
```

Both inline tuning runs were stopped manually after remaining active on the GPU without producing a measured result: the full layer after about six minutes, and the single `w2_wgrad` BMM after about two minutes. Read: `torch.cuda.tunable` is available in this ROCm PyTorch (`2.9.1+rocm7.2.0`), but inline tuning is not currently a practical iteration path for the DSv4 routed-MoE shapes. Enabling TunableOp without tuning is near-neutral, not a material optimization. If this direction is revisited, use an offline/precomputed tuning-file workflow and benchmark only after the tuning file exists; otherwise prioritize a real packed-FP4 grouped MoE training GEMM or CK/TileLang kernels for these shapes.

## BLAS Backend and BMM Layout Probe

Extended `dsv4_layer_microbench/probe_tunableop_bmm.py` with `--blas-backend`, `--layout`, and `--measure-copy-cost` controls. The probe now distinguishes the ideal contiguous operand shape from the actual cyclic routed-MoE path, where forward W1/W3/W2 use strided RHS weight transposes and wgrad uses strided LHS activation transposes.

S2048 BLAS backend commands:

```bash
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --blas-backend current --json --summary-path profiles/blas_backend_bmm_all_s2048_current/summary.json
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --blas-backend hipblas --json --summary-path profiles/blas_backend_bmm_all_s2048_hipblas/summary.json
```

| Op | Current backend `Cublaslt` ms | Forced `hipblas` ms | Current TFLOPs | Forced `hipblas` TFLOPs |
| --- | ---: | ---: | ---: | ---: |
| `fwd_w1` | `3.0134` | `3.0517` | `179.59` | `177.33` |
| `w2_dgrad` | `2.9405` | `2.9141` | `184.04` | `185.71` |
| `w1w3_xgrad` | `5.9072` | `6.2020` | `183.22` | `174.51` |
| `w2_wgrad` | `3.5491` | `3.5538` | `152.48` | `152.28` |
| `w1_wgrad` | `3.5259` | `3.5271` | `153.48` | `153.43` |
| `w3_wgrad` | `3.5452` | `3.5385` | `152.65` | `152.94` |

Forced CK backend failed for both the full all-op sweep and a single `w2_wgrad` probe with `RuntimeError: wrong! device_gemm with the specified compilation parameters does not support this GEMM problem`. Read: keep the ROCm PyTorch default `Cublaslt`/hipBLASLt backend for these grouped BMM shapes; `hipblas` is mixed and not a useful override, and CK is not a drop-in backend for this contract.

S2048 layout commands:

```bash
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --blas-backend current --layout contiguous --json --summary-path profiles/bmm_layout_s2048_contiguous/summary.json
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --blas-backend current --layout actual --json --summary-path profiles/bmm_layout_s2048_actual/summary.json
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --blas-backend current --layout materialized-transpose --json --summary-path profiles/bmm_layout_s2048_materialized_transpose/summary.json
python3 dsv4_layer_microbench/probe_tunableop_bmm.py --op w2_wgrad --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 2 --iters 5 --blas-backend current --layout actual --measure-copy-cost --json --summary-path profiles/bmm_layout_s2048_wgrad_copy_cost/summary.json
```

| Op | Contiguous ms | Actual-layout ms | Materialized-transpose BMM-only ms |
| --- | ---: | ---: | ---: |
| `fwd_w1` | `2.9715` | `2.8317` | `3.0290` |
| `w2_dgrad` | `2.9223` | `3.0249` | `2.9937` |
| `w1w3_xgrad` | `6.1982` | `5.9247` | `6.0095` |
| `w2_wgrad` | `3.5584` | `3.9369` | `3.5652` |
| `w1_wgrad` | `3.5216` | `3.7123` | `3.5481` |
| `w3_wgrad` | `3.5229` | `3.6370` | `3.5207` |

Copy-cost timing for `grad.transpose(1, 2).contiguous()` on the three actual wgrad LHS operands: W2 `0.2034 ms` for `176,160,768` bytes, W1 `0.0966 ms` for `75,497,472` bytes, W3 `0.0934 ms` for `75,497,472` bytes, all three `0.3943 ms` for `327,155,712` bytes.

Added selectable MoE variant `grouped-expert-batches-cyclic-uniform-contig-wgrad-lhs-baddbmm-xgrad` to test this in the full layer. Correctness gate passed:

```bash
python3 dsv4_layer_microbench/check_correctness.py --device cuda --dtype float64 --batch 1 --seqlen 4 --checks moe_expert_batches_cyclic_uniform_contig_wgrad_lhs_baddbmm_xgrad_match --json
```

Full-layer adjacent S2048 A/B:

```bash
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-contig-wgrad-lhs-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_contig_wgrad_lhs_baddbmm_xgrad_saved_accum_5iter
python3 dsv4_layer_microbench/profile_microbench.py --preset full --impl fp4-packed-sim --moe-impl grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad --shared-expert-impl linear-pair --attention-impl sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum --batch 1 --seqlen 2048 --dtype bfloat16 --device cuda --warmup 2 --iters 5 --fixed-router --rmsnorm-save-policy saved-accum --no-torch-trace --trace-dir profiles/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_adjacent_layout_probe_5iter
```

| MoE impl | Total ms | Tokens/s | TFLOPs | Forward ms | Backward ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| retained `dense-wgrad-baddbmm-xgrad` adjacent baseline | `47.4972` | `43118.34` | `204.9838` | `15.1106` | `32.2915` |
| `contig-wgrad-lhs-baddbmm-xgrad` | `47.5242` | `43093.80` | `204.8671` | `15.0927` | `32.3388` |

Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/blas_backend_bmm_all_s2048_current/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/blas_backend_bmm_all_s2048_hipblas/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/bmm_layout_s2048_contiguous/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/bmm_layout_s2048_actual/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/bmm_layout_s2048_materialized_transpose/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/bmm_layout_s2048_wgrad_copy_cost/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_contig_wgrad_lhs_baddbmm_xgrad_saved_accum_5iter/summary.json`, and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/full_b1s2048_fp4_packed_sim_triton_qnorm_bf16_contract_cyclic_uniform_dense_wgrad_baddbmm_xgrad_saved_accum_adjacent_layout_probe_5iter/summary.json`.

Read: materializing routed-wgrad LHS transposes is correctness-safe but not a full-layer win. It saves BMM time in isolation, but the copies and allocator/scheduling effects make the adjacent full-layer result neutral/slightly slower. Keep the retained `dense-wgrad-baddbmm-xgrad` path as the current ceiling and treat `contig-wgrad-lhs-baddbmm-xgrad` as another negative-control layout probe.

## AITER Batched A16WFP4 Routed-MoE Probe

Added `dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py` to isolate AITER's batched GEMM API for the cyclic routed-MoE forward/dgrad-like shapes. Contract caveat: the API accepts BF16/FP16 `x`, packed MXFP4 weights, and K-group-32 E8M0 scales, but the kernel quantizes activations to MXFP4 on the fly. This is not the current 128x128 UE8M0 packed-sim training contract and does not cover routed expert wgrad.

Smoke/correctness command:

```bash
python3 dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py --op all --tokens 64 --hidden 256 --intermediate 128 --experts 16 --topk 4 --warmup 1 --iters 2 --check --check-expert-chunk 4 --json --summary-path profiles/aiter_batched_a16wfp4_smoke_t64_h256_i128_e16_k4/summary.json
```

Smoke result: pass. Both `w1_like` and `w2_like` were exact against the chunked PyTorch dequantized reference: max abs `0.0`, rel L2 `0.0`, `100.0%` close. Artifact: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_batched_a16wfp4_smoke_t64_h256_i128_e16_k4/summary.json`.

Real-shape S2048 commands:

```bash
python3 dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 5 --iters 20 --json --summary-path profiles/aiter_batched_a16wfp4_t2048_h7168_i3072_e384_k6_20iter/summary.json
python3 dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py --op all --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 3 --iters 5 --config-label bm32_split1 --config-json '{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"kpack":1,"cache_modifier":".cg","NUM_KSPLIT":1}' --json --summary-path profiles/aiter_batched_a16wfp4_t2048_all_bm32_split1/summary.json
python3 dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py --op w1_like --tokens 2048 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 1 --iters 1 --check --check-expert-chunk 4 --config-label bm32_split1 --config-json '{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"kpack":1,"cache_modifier":".cg","NUM_KSPLIT":1}' --json --summary-path profiles/aiter_batched_a16wfp4_t2048_w1_like_bm32_split1_check/summary.json
```

S2048 result:

| Config | Op | Shape | Median ms | TFLOPs | Notes |
| --- | --- | --- | ---: | ---: | --- |
| default | `w1_like` | `[384,32,7168] x [384,3072,7168]` | `2.6686` | `202.79` | W1/W3 forward or W2 dgrad |
| default | `w2_like` | `[384,32,3072] x [384,7168,3072]` | `3.0498` | `177.44` | W2 forward |
| `bm32_split1` | `w1_like` | `[384,32,7168] x [384,3072,7168]` | `2.5196` | `214.78` | best S2048 W1-like |
| `bm32_split1` | `w2_like` | `[384,32,3072] x [384,7168,3072]` | `2.4452` | `221.32` | best S2048 W2-like |

The real-shape S2048 `w1_like` correctness check with `bm32_split1` passed against chunked dequantized PyTorch reference: max abs `0.0`, rel L2 `0.0`, `100.0%` close. Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_batched_a16wfp4_t2048_h7168_i3072_e384_k6_20iter/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_batched_a16wfp4_t2048_all_bm32_split1/summary.json`, and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_batched_a16wfp4_t2048_w1_like_bm32_split1_check/summary.json`.

Real-shape S4096 commands:

```bash
python3 dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py --op all --tokens 4096 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 5 --iters 20 --json --summary-path profiles/aiter_batched_a16wfp4_t4096_h7168_i3072_e384_k6_20iter/summary.json
python3 dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py --op w1_like --tokens 4096 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 3 --iters 5 --config-label bm32_split1 --config-json '{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"kpack":1,"cache_modifier":".cg","NUM_KSPLIT":1}' --json --summary-path profiles/aiter_batched_a16wfp4_t4096_w1_like_bm32_split1/summary.json
python3 dsv4_layer_microbench/probe_aiter_batched_a16wfp4.py --op w2_like --tokens 4096 --hidden 7168 --intermediate 3072 --experts 384 --topk 6 --warmup 3 --iters 5 --config-label bm32_split1 --config-json '{"BLOCK_SIZE_M":32,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":256,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"kpack":1,"cache_modifier":".cg","NUM_KSPLIT":1}' --json --summary-path profiles/aiter_batched_a16wfp4_t4096_w2_like_bm32_split1/summary.json
```

S4096 result:

| Config | Op | Shape | Median ms | TFLOPs | Notes |
| --- | --- | --- | ---: | ---: | --- |
| default | `w1_like` | `[384,64,7168] x [384,3072,7168]` | `12.8086` | `84.50` | bad default config at M=64 |
| default | `w2_like` | `[384,64,3072] x [384,7168,3072]` | `12.1164` | `89.33` | bad default config at M=64 |
| `bm32_split4` | `w1_like` | `[384,64,7168] x [384,3072,7168]` | `5.2763` | `205.13` | smaller M tile recovers most loss |
| `bm32_split1` | `w1_like` | `[384,64,7168] x [384,3072,7168]` | `4.8899` | `221.34` | best checked S4096 W1-like |
| `bm32_split1` | `w2_like` | `[384,64,3072] x [384,7168,3072]` | `4.7319` | `228.73` | best checked S4096 W2-like |

Artifacts: `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_batched_a16wfp4_t4096_h7168_i3072_e384_k6_20iter/summary.json`, `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_batched_a16wfp4_t4096_w1_like_bm32_split1/summary.json`, and `/Users/sonle5/amd_profile/dsv4_layer_microbench_20260530/aiter_batched_a16wfp4_t4096_w2_like_bm32_split1/summary.json`.

Read: this is useful as a kernel-contract probe but not ready for layer integration. At S2048, `bm32_split1` would put the three routed-MoE forward GEMMs around `2 * 2.5196 + 2.4452 = 7.4844 ms`, versus the retained rocprof routed-MoE forward GEMM family at about `8.6627 ms`, so it could save roughly `1.18 ms` in forward only if the activation-quantized MXFP4 contract were acceptable. At S4096, the same estimate is `2 * 4.8899 + 4.7319 = 14.5117 ms`, worse than the retained full routed-MoE forward block around `9.66 ms`. More importantly, this AITER surface still quantizes activations to MXFP4, uses K-group-32 MXFP4 scales rather than the experiment's 128x128 UE8M0 FP4 contract, and does not address routed wgrad/backward, which rocprof shows is the real S2048 limiter. Do not integrate into the retained layer path yet; keep it as evidence that AITER's FP4-family kernels need shape/config care and that forward-only substitutions cannot close the training gap.

## 2026-06-01 LightningIndexer Integrated Attribution

Integrated the standalone LightningIndexer path into the reference sparse-MLA attention mode `sparse-gather-lightning-indexer` and updated `profile_microbench.py` so attention subblock replay uses the same per-batch dynamic selected-KV gather as the real forward. The attribution path now allows unused top-level attention params because LightningIndexer top-k indices are discrete and do not backpropagate through the selected-index path.

MI350 command shape:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size=64g -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/work/tools/dsv4_layer_microbench -v /local/data/sonle5/dsv4_pretrain_lightning_indexer_20260601:/work -w /work/tools/dsv4_layer_microbench rocm/sgl-dev:rocm720-mi35x-f96ac98-20260527-DSv4 python3 profile_microbench.py --device cuda --dtype bfloat16 --preset small --batch 1 --seqlen 2048 --attention-impl sparse-gather-lightning-indexer --moe-impl grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad --shared-expert-impl fused-w13 --profile-backward-blocks --profile-attention-blocks --backward-attribution-warmup 0 --backward-attribution-iters 1 --warmup 0 --iters 1 --no-torch-trace --trace-dir /work/traces/lightning_indexer_integrated_small_s2048_mi350_bf16_backward_attr_20260601
```

Result on `do-sonle-kernel`: passed. Local compact artifact: `run_artifacts/lightning_indexer_integrated_small_s2048_mi350_bf16_backward_attr_20260601.json`; raw summary source remains at `/local/data/sonle5/dsv4_pretrain_lightning_indexer_20260601/traces/lightning_indexer_integrated_small_s2048_mi350_bf16_backward_attr_20260601/summary.json`.

| Probe | Value |
| --- | ---: |
| `phase_ms.forward` | `7119.7731` |
| `phase_ms.backward` | `4586.8781` |
| `attn.q_rank` forward | `3185.1458 ms` |
| `attn.lightning_indexer` forward | `1460.1299 ms` |
| `attn.sparse_mla` forward | `924.7473 ms` |
| `moe.routed_experts` forward | `905.8679 ms` |
| `attention.sparse_mla.dgrad_qkv` segmented | `0.5248 ms` |
| `attention.dgrad` segmented | `2.5255 ms` |
| `attention.wgrad` segmented | `2.3829 ms` |
| `moe.routed.dgrad` segmented | `152.2359 ms` |
| `moe.routed.wgrad` segmented | `2.1687 ms` |
| sparse keys/query | `656` |
| `kv_all` tokens | `2688` |
| static traffic estimate | `221.1 MB` |

The same attribution path passed a tiny CPU container smoke at `preset=gradcheck`, `S=4`, artifact `run_artifacts/lightning_indexer_integrated_cpu_profile_attr_smoke_20260601.json`. Its attention-subblock input gradient matched the top-level attention dgrad with relative L2 `4.86e-8`; the MI350 BF16 run matched to max abs `1.09e-8`, relative L2 `4.01e-2`, which is acceptable for this one-iteration BF16 attribution smoke.

Full preset `S=512` with BF16/reference dynamic MoE OOMed after LightningIndexer had executed because the BF16 MoE fallback materialized selected expert weights. Artifact: `run_artifacts/lightning_indexer_integrated_full_s512_mi350_bf16_oom_20260601.json`. The full-shape path should use `fp4-packed-sim` compact MoE for integrated LI probes until a BF16 compact-native grouped MoE path avoids selected-weight materialization.

Read: this is correctness/profile plumbing progress, not a performance win yet. The current integrated LightningIndexer reference is dominated by PyTorch FP4-packed-sim overheads (`attn.q_rank`) and materialized selected-KV sparse attention; the next aligned work is a fused LightningIndexer/sparse-MLA kernel contract and compact-native MoE backward, then port the path into the direct TorchTitan DSv4 canary.

## 2026-06-01 Direct TorchTitan Sparse MLA/LightningIndexer Smoke

Ported a bounded PyTorch sparse-MLA plus LightningIndexer path into `torchtitan.models.deepseek_v4` for compressed DSv4 layers. Layers with `compress_ratio=0` intentionally stay dense. Layers with `compress_ratio>0` use the sparse path only when `seqlen <= 512`; longer sequences keep the existing dense attention fallback so the current S4096 canary does not materialize `[B, S, sparse_keys, H, D]` tensors before a fused kernel exists.

Config-only check on `do-sonle5-mi350-gpu` with the direct non-Primus TorchTitan launcher:

```bash
CANARY_SEQ_LEN=128 CANARY_STEPS=1 CANARY_CHECKPOINT_ENABLE=false \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh check-config
```

Result: passed. The parsed direct TorchTitan config reports model status `direct_dsv4_pytorch_sparse_mla_lightning_indexer_for_compressed_layers_dense_fallback_over_seq_threshold`, attention statuses `['dense_mla_for_uncompressed_dsv4_layer', 'dense_mla_for_uncompressed_dsv4_layer', 'pytorch_sparse_mla_lightning_indexer_shortseq_dense_fallback_over_seq_threshold', 'pytorch_sparse_mla_lightning_indexer_shortseq_dense_fallback_over_seq_threshold']`, and compress ratios `[0, 0, 4, 128]`.

Short-sequence attention smoke command, inside `onenexus/nexus-titan:rocm722-pytorch-nightly`:

```bash
python3 /local/data/sonle5/dsv4_pretrain_canary_20260527/tools/dsv4_runtime_probe/smoke_torchtitan_dsv4_sparse_mla.py \
  --device cpu --dtype float32 --seq-len 32 \
  --summary-path /local/data/sonle5/dsv4_pretrain_canary_20260527/run_artifacts/torchtitan_dsv4_sparse_mla_li_cpu_smoke_20260601.json \
  --json
```

Result: passed. Compact artifact: `run_artifacts/torchtitan_dsv4_sparse_mla_li_cpu_smoke_20260601.json`.

| Probe | Value |
| --- | --- |
| output shape | `[1, 32, 64]` |
| sparse path used | `true` |
| indexer loss | `1.7687e-09` |
| LI `wq.weight` grad norm | `7.5441e-09` |
| LI `wk.weight` grad norm | `7.1184e-09` |
| LI `weights_proj.weight` grad norm | `6.1304e-09` |

Threshold-guard smoke command:

```bash
python3 /local/data/sonle5/dsv4_pretrain_canary_20260527/tools/dsv4_runtime_probe/smoke_torchtitan_dsv4_sparse_mla.py \
  --device cpu --dtype float32 --seq-len 640 \
  --summary-path /local/data/sonle5/dsv4_pretrain_canary_20260527/run_artifacts/torchtitan_dsv4_sparse_mla_threshold_guard_cpu_smoke_20260601.json \
  --json
```

Result: passed. Compact artifact: `run_artifacts/torchtitan_dsv4_sparse_mla_threshold_guard_cpu_smoke_20260601.json`; it reports `sparse_used=false`, `last_indexer_loss=null`, output shape `[1, 640, 64]`, and null indexer gradients as expected.

Real Flash-dimension MI350 attention-layer probe:

```bash
python3 /local/data/sonle5/dsv4_pretrain_canary_20260527/tools/dsv4_runtime_probe/verify_torchtitan_dsv4_sparse_li_flash_layers.py \
  --device cuda --dtype bfloat16 --seq-len 128 --batch 1 --layers compressed \
  --summary-path /local/data/sonle5/dsv4_pretrain_canary_20260527/run_artifacts/torchtitan_dsv4_flash_layers_sparse_li_mi350_bf16_20260601.json \
  --json
```

Result: passed on one MI350 inside `onenexus/nexus-titan:rocm722-pytorch-nightly`. The probe builds the real `deepseek_v4 flash_4layer` attention configs at model dimension `4096` and verifies the compressed layers explicitly set `last_sparse_used=true`. Compact artifact: `run_artifacts/torchtitan_dsv4_flash_layers_sparse_li_mi350_bf16_20260601.json`.

| Layer | S | Compress ratio | Compressed groups | Selected keys/query | LI loss | LI grad read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | `128` | `4` | `32` | `160` | `4.2649e-05` | nonzero `wq/wk/weights_proj` grads |
| 3 | `128` | `128` | `1` | `129` | `0.0` | zero grads expected because only one compressed group exists |

Layer 3 high-compression backward proof:

```bash
python3 /local/data/sonle5/dsv4_pretrain_canary_20260527/tools/dsv4_runtime_probe/verify_torchtitan_dsv4_sparse_li_flash_layers.py \
  --device cuda --dtype bfloat16 --seq-len 512 --batch 1 --layers 3 \
  --summary-path /local/data/sonle5/dsv4_pretrain_canary_20260527/run_artifacts/torchtitan_dsv4_flash_layer3_sparse_li_s512_mi350_bf16_20260601.json \
  --json
```

Result: passed. Compact artifact: `run_artifacts/torchtitan_dsv4_flash_layer3_sparse_li_s512_mi350_bf16_20260601.json`; layer 3 used sparse MLA with `compress_ratio=128`, `compressed_groups=4`, `selected_keys=132`, `last_indexer_loss=4.7848e-07`, and nonzero LI gradient norms: `wq.weight=3.2370e-06`, `wk.weight=4.6755e-06`, `weights_proj.weight=2.5870e-06`.

Direct 4xMI350 short-sequence training smoke command:

```bash
env GPUS_PER_NODE=4 CANARY_VISIBLE_DEVICES=0,1,2,3 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=128 CANARY_STEPS=2 CANARY_MBS=1 CANARY_GBS=64 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  RUN_ID=torchtitan_direct_dsv4_sparse_li_shortseq_4xmi350_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: passed, direct TorchTitan only, no Primus. The run built `deepseek_v4 flash_4layer`, created active mesh dimensions `['batch', 'loss', 'ep', 'fsdp']` on all four ranks, used `S=128`, `GBS=64`, `FSDP=4`, `EP=4`, `gradient_accumulation_steps=16`, and wrote only `1.1M` of remote output with checkpointing disabled. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_sparse_li_shortseq_4xmi350_20260601.json`.

| Step | Loss | Grad norm | Memory GiB | Tokens/GPU/s | TFLOPs/GPU | MFU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `12.30273` | `9.3750` | `76.41` | `115-116` | `1.46-1.47` | `0.47%` |
| 2 | `11.79978` | `18.2500` | `95.11` | `461` | `5.85` | `1.87%` |

Read: this removes the source-level "missing sparse MLA/indexer" hole for short-sequence TorchTitan correctness and gives compressed layers trainable LI parameters at both toy and real Flash attention dimensions. It is not a production S4096 sparse-MLA implementation. The guard is intentional until the single-layer fused sparse-MLA/LI kernel can replace the materializing PyTorch fallback.

Direct 4xMI350 S512 sparse-LI training isolation:

```bash
env GPUS_PER_NODE=4 CANARY_VISIBLE_DEVICES=0,1,2,3 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=2 CANARY_MBS=1 CANARY_GBS=64 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  RUN_ID=torchtitan_direct_dsv4_sparse_li_s512_4xmi350_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: failed before first step metric during `loss.backward()` with GPU memory access faults and `SIGABRT`. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_sparse_li_s512_4xmi350_fullac_failure_20260601.json`.

Activation-checkpoint isolation command:

```bash
env GPUS_PER_NODE=4 CANARY_VISIBLE_DEVICES=0,1,2,3 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=64 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none \
  RUN_ID=torchtitan_direct_dsv4_sparse_li_s512_4xmi350_noac_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: failed the same way before first step metric during `loss.backward()`. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_sparse_li_s512_4xmi350_noac_failure_20260601.json`.

| Run | Activation checkpoint | Step metrics | Failure | Output | Checkpoint dir |
| --- | --- | --- | --- | ---: | --- |
| `torchtitan_direct_dsv4_sparse_li_s512_4xmi350_20260601T000000` | `full` | none | GPU memory access fault, root rank 1, `SIGABRT` | `652K` | no |
| `torchtitan_direct_dsv4_sparse_li_s512_4xmi350_noac_20260601T000000` | `none` | none | GPU memory access fault, root rank 0, `SIGABRT` | `652K` | no |

Read: this reinforces the execution decision to skip Primus. The bounded sparse-LI implementation itself works in direct TorchTitan at `S=128`, and standalone real Flash layer-3 LI backward works at `S=512`; the new blocker is the direct distributed `S=512` backward integration surface, likely around sparse-LI materialization/autograd plus FSDP/EP/ROCm interaction. Next isolation should compare `S=256/S=512`, 1x versus 4x, dense fallback versus sparse-LI, and FSDP/EP interaction before returning to long-sequence performance.

Direct S512 fault isolation follow-up:

Added `TORCHTITAN_DSV4_PYTORCH_SPARSE_MLA_MAX_SEQ_LEN` support in the direct TorchTitan WIP and launcher env `CANARY_SPARSE_MLA_MAX_SEQ_LEN` so the same DSv4 config can force compressed layers back to dense MLA above a chosen threshold. A remote check-config with `CANARY_SPARSE_MLA_MAX_SEQ_LEN=128` reported `attention_sparse_threshold_by_layer [128, 128, 128, 128]`.

4xMI350 S256 sparse-LI command:

```bash
env GPUS_PER_NODE=4 CANARY_VISIBLE_DEVICES=0,1,2,3 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=256 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=64 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none MASTER_PORT=29541 \
  RUN_ID=torchtitan_direct_dsv4_sparse_li_s256_4xmi350_noac_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: passed one checkpoint-free step. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_sparse_li_s256_4xmi350_noac_20260601.json`.

4xMI350 S512 dense-fallback command:

```bash
env GPUS_PER_NODE=4 CANARY_VISIBLE_DEVICES=0,1,2,3 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=64 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none CANARY_SPARSE_MLA_MAX_SEQ_LEN=128 \
  MASTER_PORT=29542 RUN_ID=torchtitan_direct_dsv4_s512_densefallback_4xmi350_noac_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: passed one checkpoint-free step. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_s512_densefallback_4xmi350_noac_20260601.json`.

1xMI350 S512 sparse-LI command:

```bash
env GPUS_PER_NODE=1 CANARY_VISIBLE_DEVICES=0 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=16 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none MASTER_PORT=29543 \
  RUN_ID=torchtitan_direct_dsv4_sparse_li_s512_1xmi350_noac_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: failed before first step metric during `loss.backward()` with GPU memory access fault and `SIGABRT`. The traceback reached `torch.distributed.fsdp._fully_shard._fsdp_collectives.chunk_cat` and `foreach_reduce_scatter_copy_in` in post-backward. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_sparse_li_s512_1xmi350_noac_failure_20260601.json`.

1xMI350 S512 sparse-LI no-FSDP/no-aux-loss/no-guard control:

```bash
env GPUS_PER_NODE=1 CANARY_VISIBLE_DEVICES=0 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=1 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none CANARY_SKIP_SINGLE_RANK_FSDP=true \
  CANARY_DSA_INDEXER_LOSS_COEFF=0 CANARY_SPARSE_MLA_MAX_SELECTED_STATE_ELEMENTS=0 \
  MASTER_PORT=29553 RUN_ID=torchtitan_direct_dsv4_s512_sparse_li_1xmi350_noac_nofsdp_noaux_noguard_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: failed before first step metric during `loss.backward()` with GPU memory access fault and `SIGABRT`. This run skipped single-rank FSDP, disabled the LI aux loss, and disabled the selected-state guard so the unguarded sparse path was forced. TorchTitan initialized with the corrected MI350X peak FLOPs (`2.310e+15`). Compact artifact: `run_artifacts/torchtitan_direct_dsv4_sparse_li_s512_1xmi350_noac_nofsdp_noaux_noguard_failure_20260601.json`.

1xMI350 S512 sparse-LI layer-3-only no-FSDP/no-aux-loss control:

```bash
env GPUS_PER_NODE=1 CANARY_VISIBLE_DEVICES=0 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=1 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none CANARY_SKIP_SINGLE_RANK_FSDP=true \
  CANARY_DSA_INDEXER_LOSS_COEFF=0 CANARY_SPARSE_MLA_MAX_SELECTED_STATE_ELEMENTS=4500000000 \
  MASTER_PORT=29554 RUN_ID=torchtitan_direct_dsv4_s512_sparse_li_layer3only_1xmi350_noac_nofsdp_noaux_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: failed before first step metric during `loss.backward()` with GPU memory access fault and `SIGABRT`. The companion config parse reported selected-state thresholds of `4,500,000,000` for all layers, LI loss coefficients `0.0`, and compress ratios `[0, 0, 4, 128]`. At `S=512`, layer 2 estimates `8,589,934,592` selected-state elements and should fall back, while layer 3 estimates `4,429,185,024` and should remain sparse. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_sparse_li_s512_layer3only_1xmi350_noac_nofsdp_noaux_failure_20260601.json`.

| Run | Shape | Sparse threshold | Result | Loss | Grad norm | Tokens/GPU/s | Read |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `torchtitan_direct_dsv4_sparse_li_s256_4xmi350_noac_20260601T000000` | `FSDP4_EP4_GBS64_S256` | `512` | pass | `12.40494` | `9.5625` | `233` | sparse-LI works beyond S128 |
| `torchtitan_direct_dsv4_s512_densefallback_4xmi350_noac_20260601T000000` | `FSDP4_EP4_GBS64_S512` | `128` | pass | `12.24314` | `10.3125` | `356-358` | generic S512/FSDP/EP is not the issue; `0.20-0.21%` MI350X-normalized MFU |
| `torchtitan_direct_dsv4_sparse_li_s512_1xmi350_noac_20260601T000000` | `FSDP1_EP1_GBS16_S512` | `512` | fail | n/a | n/a | n/a | failure is not primarily multi-rank EP |
| `torchtitan_direct_dsv4_s512_sparse_li_1xmi350_noac_nofsdp_noaux_noguard_20260601T000000` | `FSDP1_EP1_GBS1_S512`, no FSDP, no aux loss, no guard | `512` | fail | n/a | n/a | n/a | failure is in materialized sparse attention backward itself |
| `torchtitan_direct_dsv4_s512_sparse_li_layer3only_1xmi350_noac_nofsdp_noaux_20260601T000000` | `FSDP1_EP1_GBS1_S512`, no FSDP, no aux loss, layer 3 only sparse | `512`, selected-state `4.5B` | fail | n/a | n/a | n/a | failure is not layer-2-only; layer-3 sparse also fails inside full model |

Read: the S512 full-model failure is now localized to the materialized PyTorch sparse-MLA backward path. It reproduces without Primus, without multi-rank EP, without activation checkpointing, without single-rank FSDP, and without LI auxiliary loss. Dense fallback at S512 passes on 4x and standalone layer-3 LI backward at S512 passes, but layer-3-only sparse inside the full model still faults. The next fix should avoid materializing the selected K/V backward shape for compressed-layer integration generally, not only for the layer-2 `compress_ratio=4` shape.

Direct S512 selected-state guard follow-up:

Added `TORCHTITAN_DSV4_PYTORCH_SPARSE_MLA_MAX_SELECTED_STATE_ELEMENTS` and launcher env `CANARY_SPARSE_MLA_MAX_SELECTED_STATE_ELEMENTS`. The default is `4,000,000,000` selected K/V state elements. This is a safety guard for the temporary materializing PyTorch sparse-MLA fallback: it preserves S256 sparse-LI but falls back before the S512 selected-state backward shapes that fault on ROCm.

Layer probe boundary:

| Probe | Layer 2 estimate | Layer 3 estimate | Sparse used | Result |
| --- | ---: | ---: | --- | --- |
| `S=256`, layers `2,3` | `3,221,225,472` | `2,181,038,080` | both layers | passed, nonzero LI grads |
| `S=512`, layers `2,3` | `8,589,934,592` | `4,429,185,024` | neither layer | passed via selected-state fallback |

1x and 4x guarded training:

```bash
env GPUS_PER_NODE=1 CANARY_VISIBLE_DEVICES=0 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=1 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none MASTER_PORT=29550 \
  RUN_ID=torchtitan_direct_dsv4_s512_selected_state_guard4b_1xmi350_gbs1_noac_fsdp_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

```bash
env GPUS_PER_NODE=4 CANARY_VISIBLE_DEVICES=0,1,2,3 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=512 CANARY_STEPS=1 CANARY_MBS=1 CANARY_GBS=64 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true \
  CANARY_ACTIVATION_CHECKPOINT_MODE=none MASTER_PORT=29551 \
  RUN_ID=torchtitan_direct_dsv4_s512_selected_state_guard4b_4xmi350_noac_retry_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

| Run | Shape | Result | Loss | Grad norm | Tokens/GPU/s | Read |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `torchtitan_direct_dsv4_s512_selected_state_guard4b_1xmi350_gbs1_noac_nofsdp_20260601T000000` | `FSDP1_EP1_GBS1_S512`, diagnostic no-FSDP | pass | `12.15967` | `13.25` | `53` | raw full-model autograd safe after guard |
| `torchtitan_direct_dsv4_s512_selected_state_guard4b_1xmi350_gbs1_noac_fsdp_20260601T000000` | `FSDP1_EP1_GBS1_S512` | pass | `12.26154` | `13.4375` | `45` | prior FSDP-looking stack was downstream of materialized sparse fault |
| `torchtitan_direct_dsv4_s512_selected_state_guard4b_4xmi350_noac_retry_20260601T000000` | `FSDP4_EP4_GBS64_S512` | pass | `12.26200` | `10.3125` | `554-557` | distributed default S512 path is stable; `0.32%` MI350X-normalized MFU |

Compact artifact: `run_artifacts/torchtitan_direct_dsv4_s512_selected_state_guard4b_20260601.json`. The 4x run wrote `828K` under `/local/data/sonle5/dsv4_pretrain_canary_20260527/outputs/torchtitan_direct_dsv4_s512_selected_state_guard4b_4xmi350_noac_retry_20260601T000000`, created no checkpoint directory, and destroyed process groups cleanly.

Normalized 4x target comparison for the guarded S512 run: target remains `2,200-2,400` tokens/GPU/s, or `8,800-9,600` tokens/node/s. The observed `554-557` tokens/GPU/s is `23-25%` of target and needs about `4.0-4.3x` speedup. At `GBS=64`, `S=512`, the observed step time is about `14.7-14.8 s` versus a normalized target of `3.4-3.7 s`. The raw TorchTitan MFU fields from this run used the old A100 fallback denominator; MI350X-normalized MFU is about `0.32%`.

Read: direct TorchTitan remains the right path and Primus is not involved. The old S512 failure is now specifically a materialized selected-state backward hazard in the temporary PyTorch sparse fallback. This guard is not the final answer; the long-sequence path still needs the fused sparse MLA/LightningIndexer kernel so `S=4096` does not fall back to dense.

Direct 8xMI350 S4096 guarded checkpoint-free refresh:

```bash
env GPUS_PER_NODE=8 CANARY_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CANARY_REQUIRE_IDLE_GPUS=true \
  CANARY_SEQ_LEN=4096 CANARY_STEPS=3 CANARY_MBS=1 CANARY_GBS=128 \
  CANARY_CHECKPOINT_ENABLE=false CANARY_DEDICATED_EP_PG=true MASTER_PORT=29552 \
  RUN_ID=torchtitan_direct_dsv4_s4096_guarded_8xmi350_ckptfree_20260601T000000 \
  experiments/2026-05-27-deepseek-v4-amd-pretrain/launchers/torchtitan_direct_deepseek_v4_4x8xmi350_canary.sh run
```

Result: passed three checkpoint-free direct TorchTitan steps after the selected-state guard. Compact artifact: `run_artifacts/torchtitan_direct_dsv4_s4096_guarded_8xmi350_ckptfree_20260601.json`. The run wrote `2.7M` under `/local/data/sonle5/dsv4_pretrain_canary_20260527/outputs/torchtitan_direct_dsv4_s4096_guarded_8xmi350_ckptfree_20260601T000000`, wrote `40K` of logs, created no checkpoint directory, and destroyed all process groups cleanly.

| Step | Loss | Grad norm | Tokens/GPU/s | TFLOPs/GPU | MFU read | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `12.24032` | `11.1875` | `2,436-2,442` | `46.09-46.22` | `2.00%` MI350X-normalized; TorchTitan logged `14.77-14.81%` via A100 fallback | warmup |
| 2 | `9.48662` | `22.7500` | `10,907` | `206.38-206.39` | `8.94%` MI350X-normalized; TorchTitan logged `66.15%` via A100 fallback | clean |
| 3 | `13.43889` | `39.7500` | `10,747` | `203.35` | `8.80%` MI350X-normalized; TorchTitan logged `65.18%` via A100 fallback | clean |

Clean-step average: `10,827` tokens/GPU/s, `86,616` tokens/node/s, implied `6.05 s/step` at `GBS=128`, and `204.865` TFLOPs/GPU. Normalized against MI350X dense BF16 peak `2309.6e12`, this is `8.87%` MFU; the previously recorded `65.665%` was TorchTitan's A100 fallback (`312e12`) because the MI350-class device name was not matched. This is `4.51-4.92x` the current `2,200-2,400` tokens/GPU/s 8x canary target, but it remains the 4-layer guarded dense-fallback systems baseline rather than final 43-layer sparse-MLA/MORI-overlap DeepSeek-V4-Flash performance.

## 2026-06-03 EP4/EP8 Target-Shape Full-Run Profile

Compact artifact: `run_artifacts/torchtitan_ep4_ep8_ownerdense_profile_breakdown_20260603.json`. Raw traces stay under `/scratch/sonle5/dsv4_pretrain_canary_20260527/outputs/`.

Run shape and knobs:

```bash
IMAGE=onenexus/nexus-titan:rocm722-pytorch-nightly-mori
TORCHTITAN_DSV4_PROFILE_DENSE_LINEAR_BACKWARD=1
TORCHTITAN_DSV4_PYTORCH_SPARSE_MLA_BACKWARD_MODE=shared_kv
TORCHTITAN_DSV4_SPARSE_MLA_SHAREDKV_SCATTER_MODE=auto
TORCHTITAN_DSV4_LI_KL_BACKEND=triton_tl_dot
TORCHTITAN_DSV4_MOE_COMM_BACKEND=mori
TORCHTITAN_EXPERIMENTAL_MORI_AITER_MOE=1
TORCHTITAN_MORI_AITER_BACKWARD_MODE=routed
TORCHTITAN_MORI_AITER_LOCAL_VJP_MODE=padded
TORCHTITAN_MORI_AITER_LOCAL_VJP_SWIGLU_BACKEND=triton
TORCHTITAN_MORI_AITER_OP_CACHE_MODE=backward
TORCHTITAN_MORI_AITER_ASSUME_IDENTICAL_DISPATCH_ORDER=1
MORI_SHMEM_HEAP_SIZE=32G
```

Both runs use `TP=1`, `PP=1`, `CP=1`, `MBS=1`, `S=4096`, 4 Flash layers, full activation checkpointing, AdamW, checkpoint disabled, and 16 gradient accumulation microbatches per optimizer step. The useful profiler cycle is `iteration_1`; `iteration_2` trace files in these runs are small profiler-flush traces with about 74 events.

| Run | Shape | Step 2 loss | Step 2 grad norm | Step 2 tok/GPU/s | Step 2 implied step time | Profiled-step wall, rank mean | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `torchtitan_dsv4_full_profile_ep4_s4096_gbs64_mbs1_ownerdense_20260603` | `FSDP4_EP4_GBS64` | `10.73168` | `52.25` | `3,085` | `21.24 s` | `28.45 s` | passes the normalized 4x throughput target even with profiling labels enabled |
| `torchtitan_dsv4_full_profile_ep8_s4096_gbs128_mbs1_ownerdense_20260603` | `FSDP8_EP8_GBS128` | `9.98689` | `51.5` | `2,383` | `27.51 s` | `35.51 s` | lands inside the `2,200-2,400` tok/GPU/s and `27-30 s/step` target band under profiler |

Rank-mean GPU annotation buckets:

| Bucket | EP4 rank-mean ms | EP8 rank-mean ms | Calls/rank | Read |
| --- | ---: | ---: | ---: | --- |
| `loss_lm_head_forward` | `14,690.73` | `19,815.03` | `142` | large loss-side dense head/FSDP interaction; not MoE or sparse attention |
| `fsdp_prefetch` | `18,748.38` | `21,091.57` | `161` | dominant trace range but heavily overlaps other work; use for system scheduling, not kernel attribution |
| `moe_routed_backward_leaf_or_call` | `3,504.68` | `3,952.58` | `1600` | routed-MoE backward leaf ladder is now the main MoE-specific profile target |
| `moe_shared_experts_backward` | `1,968.32` | `1,903.52` | `448` | shared expert dense backward is stable and second-order versus routed VJP plus loss/FSDP |
| `attention_projection_backward` | `223.95` | `216.28` | `640` | owner-tag attention projection backward is small in this run |

Routed MoE backward leaf ladder, rank-mean GPU annotation time:

| Leaf | EP4 ms | EP4 ms/call | EP8 ms | EP8 ms/call | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `mori_aiter.local_padded_vjp.sort_counts` | `978.40` | `15.29` | `1,330.32` | `20.79` | hottest routed-VJP mechanism; rebuilds per-expert padded layout with `nonzero/argsort/bincount/cumsum/max.item` |
| `mori_aiter.backward.combine_dx.call` | `451.66` | `7.06` | `761.18` | `11.89` | second routed-VJP mechanism; MORI combine of local `dX` back to token order |
| `mori_aiter.local_padded_vjp.w13_xgrad_gemm_reduce` | `349.16` | `5.46` | `280.25` | `4.38` | grouped dense math is meaningful but not the top wall at EP8 |
| `mori_aiter.local_padded_vjp.w13_wgrad_gemm` | `340.20` | `5.32` | `255.73` | `4.00` | EP8 smooths the expert GEMM shape relative to EP4 |
| `mori_aiter.local_padded_vjp.w2_dgrad_gemm` | `182.79` | `2.86` | `139.06` | `2.17` | not the leading gap |
| `mori_aiter.local_padded_vjp.w2_wgrad_gemm` | `157.52` | `2.46` | `109.68` | `1.71` | not the leading gap |
| `mori_aiter.local_padded_vjp.swiglu_backward` | `35.14` | `0.55` | `28.43` | `0.44` | Triton SwiGLU has already made this second-order |
| `mori_aiter.local_padded_vjp.swiglu_forward` | `23.03` | `0.36` | `18.70` | `0.29` | already second-order |

Follow-up layout-only probe:

Compact artifacts: `run_artifacts/owned_expert_vjp_layout_ep4_20260603.json` and `run_artifacts/owned_expert_vjp_layout_ep8_20260603.json`.

```bash
docker run --rm --network=host --ipc=host --privileged --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --volume /scratch/sonle5/dsv4_pretrain_canary_20260527:/scratch/sonle5/dsv4_pretrain_canary_20260527 \
  --entrypoint python onenexus/nexus-titan:rocm722-pytorch-nightly-mori \
  /scratch/sonle5/dsv4_pretrain_canary_20260527/tools/dsv4_runtime_probe/probe_owned_expert_vjp_ep_shape.py \
  --layout-only --tokens-per-rank 4096 --world-size 8 --hidden 4096 --intermediate 2048 \
  --experts 256 --topk 6 --warmup 5 --iters 30 --summary-path ... --json
```

| Shape | Packed recv rows | Owner assignments | `route_pack` median | `sort_counts_full` median | `sort_counts_max_hint` median | `sort_counts_skip_argsort` median | `cached_metadata_clone` median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `EP4`, `FSDP4`-equivalent owner rank | `13,514` | `24,508` | `0.0925 ms` | `0.2702 ms` | `0.2595 ms` | `0.1642 ms` | `0.0410 ms` |
| `EP8`, `FSDP8`-equivalent owner rank | `18,215` | `24,695` | `0.0907 ms` | `0.2679 ms` | `0.2501 ms` | `0.1613 ms` | `0.0399 ms` |

Read: the standalone production-shaped layout builder is sub-ms on MI350. The `max_count.item()` read is not the wall, skipping `argsort` would only save about `0.10 ms`, and saving/cloning route metadata is a `~0.04 ms` floor. Therefore the integrated `mori_aiter.local_padded_vjp.sort_counts` range in the full-run profiler is very likely attribution bleed from synchronization, allocator pressure, profiler range nesting, or surrounding MORI work rather than raw `nonzero/argsort/bincount` metadata construction. Do not spend the next optimization rung on a custom layout-sort kernel unless a production-call CUDA-event probe reproduces the 15-21 ms/call cost.

Production-call event-timing follow-up:

Compact artifacts: `run_artifacts/mori_eventtiming_ep4_summary_20260603.json`, `run_artifacts/mori_eventtiming_ep4_comparison_20260603.json`, `run_artifacts/torchtitan_dsv4_mori_swiglu_neltsruntime_4x_s4096_gbs64_20260603.json`, and `run_artifacts/torchtitan_dsv4_mori_swiglu_neltsruntime_8x_s4096_gbs128_20260603.json`.

The event probe adds opt-in `TORCHTITAN_MORI_AITER_TIMING_PATH` timing around the integrated MORI routed backward path. It intentionally synchronizes once per routed backward call, so its step throughput is diagnostic only. The EP4 timing run used `FSDP4_EP4_GBS4`, `MBS=1`, `S=4096`, full activation checkpointing, and one accumulation microbatch to preserve the target owner-rank MoE call shape while keeping the raw JSONL small.

| EP4 production-call stage | Baseline steady median | Runtime `n_elements` steady median | Read |
| --- | ---: | ---: | --- |
| `backward.local_vjp.padded` | `78.96 ms` | `25.27 ms` | local padded VJP wall removed from Triton SwiGLU specialization, not route layout |
| `local_padded_vjp.swiglu_forward` | `25.35 ms` | `0.398 ms` | `n_elements` must be a runtime scalar for dynamic routed-row counts |
| `local_padded_vjp.swiglu_backward` | `27.25 ms` | `0.527 ms` | same specialization issue in backward |
| `local_padded_vjp.sort_counts` | `0.465 ms` | `0.489 ms` | confirms route metadata is sub-ms in steady state |
| `backward.combine_dx.call` | `5.00 ms` | `4.56 ms` | still the largest non-GEMM routed-backward stage |
| `local_padded_vjp.w13_wgrad_gemm` | `5.02 ms` | `4.43 ms` | grouped GEMM family remains the math floor |
| `local_padded_vjp.w13_xgrad_gemm_reduce` | `4.78 ms` | `4.82 ms` | unchanged by the SwiGLU fix |

First routed calls on each rank still carry warmup/synchronization backlog: baseline call-1 `sort_counts` ranges `951-984 ms`, `route_pack` `199-206 ms`, and `swiglu_forward` `391-403 ms`, while steady calls drop `sort_counts` below `0.53 ms`. This explains the earlier full-profiler `sort_counts` attribution as range/sync bleed rather than real layout-sort cost.

Normal no-timing validation after the runtime-`n_elements` patch:

| Run | Shape | Step 2 loss | Step 2 grad norm | Step 2 tok/GPU/s | Step 2 implied step time | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `torchtitan_dsv4_mori_swiglu_neltsruntime_4x_s4096_gbs64_20260603` | `FSDP4_EP4_GBS64` | `8.95948` | `47.25` | `4,088` | `16.03 s` | improves over earlier 4x safe-default `3,226` and Triton-SwiGLU `2,882` runs |
| `torchtitan_dsv4_mori_swiglu_neltsruntime_8x_s4096_gbs128_20260603` | `FSDP8_EP8_GBS128` | `8.89764` | `49.25` | `2,936` | `22.32 s` | clears the normalized `2,200-2,400` tok/GPU/s 8x target band |

The EP8 event-timing attempt failed before training during deterministic seed broadcast with `hipIpcGetMemHandle invalid argument`, so it produced no MoE timing JSONL. A later normal 8x no-timing run cleared the same startup area and passed cleanly, so treat the event-timing failure as a perturbation/runtime-state failure, not evidence that EP8 target-shape training is broken.

Dense/backward classification correction:

- `MmBackward0` is not an attention/MoE projection bucket in this trace. Its rank0 input shape is always `[[512,129280]]`, and it appears 128 times per rank, matching chunked lm-head/loss-side backward.
- `_ProfiledLinearBackward` owner tags are the reliable split for attention projections, router gate, and shared experts.
- `ScaledDotProductFlashAttentionBackward0` is only about `26.5-26.7 ms` rank-mean CPU time over 64 calls, so the present profile does not support chasing dense SDPA backward as the main target.

Read: the next MoE optimization target should not be another pure SwiGLU tweak, a generic `MmBackward0` chase, or a standalone layout-sort rewrite. For the current EP4/EP8 training shape, the actionable MoE ladder is:

1. Keep the runtime-`n_elements` Triton SwiGLU patch; it removes the false `~25 ms` forward and backward SwiGLU stages inside the production routed VJP.
2. Attack MORI `combine_dx.call` for backward token-gradient return, including whether the backward path can reuse forward dispatch/combine metadata or avoid full-size padded combine buffers. Current EP4 steady median is `4.56-5.00 ms`.
3. Revisit grouped training GEMM scheduling for `w13_xgrad`, `w13_wgrad`, `w2_dgrad`, and `w2_wgrad`. After the SwiGLU fix these are again the main local-VJP math floor, with `w13_xgrad` and `w13_wgrad` around `4.4-4.8 ms` each in the EP4 production-call probe.
4. Keep the loss/FSDP buckets separate from MoE kernel work. They matter for end-to-end step time, but they are not evidence that sparse attention or routed expert GEMMs are the immediate MoE bottleneck.

Target-shape EP4/EP8 profiling split:

Compact artifact: `run_artifacts/torchtitan_ep4_ep8_target_profile_breakdown_20260603.json`.
Raw traces are kept outside the repo under `/scratch/sonle5/dsv4_pretrain_canary_20260527/outputs/<run_id>/profiling/traces/iteration_2`; MORI timing JSONL is under `<run_id>/mori_timing`.

Both runs used the TorchTitan direct DeepSeek-V4 Flash 4-layer canary with `MBS=1`, `S=4096`, full activation checkpointing, checkpointing disabled, MORI+AITER forward, routed padded VJP backward, runtime-`n_elements` Triton SwiGLU, `sparse_mla_backward_mode=shared_kv`, and `li_kl_backend=triton_tl_dot`. The profiler captured step 2 after one warmup step, so these timings are diagnostic and not exclusive wall time.

| Bucket | EP4/FSDP4/GBS64 ms/rank/step | EP8/FSDP8/GBS128 ms/rank/step | Read |
| --- | ---: | ---: | --- |
| `mori_aiter.backward.local_vjp.total` | `2451.2` | `2271.1` | EP8 smooths local routed expert VJP slightly |
| `mori_aiter.backward.local_vjp.gemm_family` | `2085.5` | `1900.6` | GEMM math is still most of local VJP |
| `mori_aiter.backward.combine_dx` | `924.3` | `2874.0` | largest EP8 MoE-specific wall; much worse than EP4 |
| Dense shared experts backward | `1973.2` | `1934.2` | large dense bucket, independent of EP |
| Attention kernel family total | `2066.7` | `2070.9` | stable across EP4/EP8 in this guarded/shared-KV path |
| Dense attention projection backward | `208.8` | `216.0` | not the anonymous `MmBackward0` wall |
| MORI forward fused MoE | `294.7` | `261.0` | forward fused MoE itself is not the top wall |
| MORI forward combine | `227.0` | `345.5` | forward combine grows with EP8 but is smaller than backward combine |
| Triton SwiGLU in local VJP | `105.6` | `97.2` | now second-order after runtime-`n_elements` fix |

Top kernel-family read from the same traces:

| Kernel family | EP4 ms/rank/step | EP8 ms/rank/step | Read |
| --- | ---: | ---: | --- |
| MORI `EpCombine` | `1155.2` | `3225.8` | confirms `combine_dx` is the EP8 regression target |
| rocBLAS/Tensile GEMM `Cijk` | `2961.7` | `2756.9` | broad dense and routed GEMM bucket |
| Copy/elementwise | `2639.8` | `1810.0` | still large but less diagnostic than named ranges |
| NCCL | `1492.4` | `985.5` | communication bucket, not only MoE |
| Attention backward `dK/dV` | `919.6` | `930.4` | stable |
| Attention forward | `714.2` | `707.4` | stable |
| Attention backward `dQ` | `432.9` | `433.2` | stable |
| MORI `EpDispatch` | `292.2` | `369.6` | smaller than combine |
| AITER CK MoE forward | `286.5` | `251.3` | forward fused MoE is not the main gap |

Read: for the target EP4/EP8 shape, the next profiling/optimization branch should be `combine_dx` first, then fused/grouped training VJP GEMM. EP8 does reduce the local owner-rank VJP math a bit, but it exposes a much larger MORI backward combine path. The shared expert dense backward and attention kernels are real step-time buckets, but they should stay separate from the routed MoE kernel story. Headline throughput should continue to use the no-timing canaries: `4,088 tok/GPU/s` on 4x and `3,026 tok/GPU/s` on 8x, because profiler/event timing perturbs the run.

Registered MORI backward combine follow-up:

Compact artifacts: `run_artifacts/mori_backward_combine_registered_ab_20260603.json`, `run_artifacts/torchtitan_dsv4_mori_registered_neltsruntime_8x_s4096_gbs128_20260603.json`, `run_artifacts/mori_auto_launch_config_ab_20260603.json`, and `run_artifacts/mori_manual_launch_geometry_probe_20260603.json`.

The first `combine_dx` lever was MORI's registered zero-copy combine input buffer, controlled by `TORCHTITAN_MORI_AITER_BACKWARD_COMBINE_INPUT_MODE`. This is not universally better:

| Shape | Mode | Diagnostic `combine_dx.call` median | Read |
| --- | --- | ---: | --- |
| `EP4/FSDP4`, `GBS4`, step2 steady | external | `4.56 ms/call` | retained for 4x |
| `EP4/FSDP4`, `GBS4`, step2 steady | registered | `12.68 ms/call` | regresses, not a 4x default |
| `EP8/FSDP8`, `GBS8`, step2 steady | external | `40.63 ms/call` | confirms EP8 combine wall |
| `EP8/FSDP8`, `GBS8`, step2 steady | registered | `28.15 ms/call` | improves EP8 combine by about `31%` |

The actual no-timing 8x target-shape check then passes with registered mode:

| Run | Shape | Step 2 loss | Step 2 grad norm | Step 2 tok/GPU/s | Step 2 implied step time | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| external baseline | `FSDP8_EP8_MBS1_GBS128_S4096` | `8.89764` | `49.25` | `2,936` | `22.32 s` | previous retained no-timing run |
| registered combine | `FSDP8_EP8_MBS1_GBS128_S4096` | `7.21987` | `25.125` | `3,026` | `21.65 s` | current retained 8x canary setting |

Read: registered combine input should now be the retained `8xMI350` target-shape canary setting, but not the global default for every shape. It gives a small target-shape throughput uplift and closes the first `combine_dx` optimization branch. The next EP8 MoE branch is MORI combine launch/block-rule tuning or split/overlapped backward combine, plus fused/grouped training VJP GEMM; the final path still needs fused sparse-MLA/LI and fused MoE backward kernels.

MORI AUTO launch-config follow-up:

The next `combine_dx` branch tested MORI's shipped `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` tuning lookup. MORI fell back from `mi350x` to the shipped `gfx950_mi355x` tuning configs, which did select different actual combine launch tuples from the manual TorchTitan bridge defaults. It was not a usable improvement:

| Shape | Mode | Actual `combine_dx` launch | Diagnostic `combine_dx.call` median | Read |
| --- | --- | --- | ---: | --- |
| `EP4/FSDP4`, `GBS4`, step2 excluding call 5 | external, manual | manual bridge default | `4.56 ms/call` | retained 4x baseline |
| `EP4/FSDP4`, `GBS4`, step2 excluding call 5 | external, `AUTO` | `[256, 0, 16]` | `22.55 ms/call` | regresses by about `4.95x` |
| `EP8/FSDP8`, `GBS8`, step2 excluding call 5 | registered, manual | manual bridge default | `28.15 ms/call` | retained 8x setting |
| `EP8/FSDP8`, `GBS8` | registered, `AUTO` | `[128, 0, 4]` | no steady sample | hung before step output; first-call-only median was `42.58 ms/call` |

Read: do not enable blanket MORI `AUTO` for this canary. The current retained path stays manual registered combine for `8xMI350`; the next combine branch should be explicit manual launch-geometry micro-sweeps or a split/overlapped backward combine design.

Manual launch-geometry follow-up:

The first manual geometry isolation kept registered combine and the retained `block=64`, `warp=8` values, but changed only `rdma_block_num` from `32` to `0`. It passed both steps at `S4096`, `GBS8`, with loss movement `12.40855 -> 8.21377`; the timing record confirms actual `combine_dx` launch `[64, 0, 8]`.

| Shape | Mode | Actual `combine_dx` launch | Diagnostic `combine_dx.call` median | Read |
| --- | --- | --- | ---: | --- |
| `EP8/FSDP8`, `GBS8`, step2 excluding call 5 | registered, manual default | `[64, 32, 8]` | `28.15 ms/call` | retained 8x setting |
| `EP8/FSDP8`, `GBS8`, step2 excluding call 5 | registered, manual no-RDMA isolation | `[64, 0, 8]` | `29.57 ms/call` | passes but regresses about `5%` |
| `EP8/FSDP8`, `GBS8`, step2 excluding call 5 | registered, manual larger-block candidate | `[128, 32, 8]` | `34.86 ms/call` | passes but regresses about `24%` |
| `EP8/FSDP8`, `GBS8`, step2 excluding call 5 | registered, manual low-warp candidate | `[64, 32, 4]` | `24.61 ms/call` | event-timed combine improves about `12.5%`, but see promotion check |

No-timing promotion check:

| Shape | Mode | Actual `combine_dx` launch | Step 2 throughput | Read |
| --- | --- | --- | ---: | --- |
| `EP8/FSDP8`, `GBS128` | registered, manual default | `[64, 32, 8]` | `3,026 tok/GPU/s` | retained 8x baseline |
| `EP8/FSDP8`, `GBS128` | registered, manual low-warp candidate | `[64, 32, 4]` | `2,801 tok/GPU/s` | regresses about `7.4%` despite better timed combine slice |

Read: `rdma_block_num=0` by itself is not the EP8 combine fix, and the AUTO failure is not explained only by no-RDMA. Increasing block count to `128` while keeping `rdma=32`, `warp=8` is also worse. Lowering warp count to `4` improves the event-timed combine median, but the real no-timing `GBS128` run loses throughput. Keep `[64, 32, 8]` as the retained manual registered tuple. This branch has now rejected the obvious no-RDMA, larger-block, and low-warp promotion variants, so the next combine branch should pivot to split/overlapped backward combine and a full EP4/EP8 bucket profile rather than many blind tuple sweeps.

Full EP4/EP8 full-shape profiler split:

Artifact: [`run_artifacts/torchtitan_dsv4_full_profile_ep4_ep8_s4096_mbs1_20260603.json`](run_artifacts/torchtitan_dsv4_full_profile_ep4_ep8_s4096_mbs1_20260603.json). Raw Torch traces and MORI timing JSONL stay under `/scratch/sonle5/dsv4_pretrain_canary_20260527/outputs/`.

Both runs use `MBS=1`, `S=4096`, full activation checkpointing, direct TorchTitan, MORI+AITER routed padded VJP, Triton SwiGLU local VJP, registered backward combine input, and retained manual MORI tuple `[64,32,8]`. Torch profiler timings are diagnostic; annotation buckets can overlap and should not be summed as wall time.

| Shape | Run id | Step 2 throughput | Profiler step wall | Loss movement |
| --- | --- | ---: | ---: | --- |
| `EP4/FSDP4`, `GBS64` | `torchtitan_dsv4_full_profile_ep4_s4096_gbs64_mbs1_pf2_20260603` | `3,862 tok/GPU/s` | `16.97 s` | `12.22685 -> 9.34519` |
| `EP8/FSDP8`, `GBS128` | `torchtitan_dsv4_full_profile_ep8_s4096_gbs128_mbs1_pf2_20260603` | `2,889 tok/GPU/s` | `22.68 s` | `12.28368 -> 7.70633` |

MORI routed-backward steady step-2 per-call medians, excluding the first call of the profiled step:

| Shape | Records | Recv rows median | `local_vjp.padded` | `combine_dx.call` | `dispatch_x.call` | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `EP4/FSDP4`, `GBS64` | `252` | `14,175.5` | `45.44 ms` | `14.74 ms` | `1.02 ms` | heavier owner-rank VJP math |
| `EP8/FSDP8`, `GBS128` | `504` | `18,719.5` | `29.63 ms` | `25.12 ms` | `2.80 ms` | combine tail gets worse; p90 `64.53 ms` |

Rank-median kernel-family totals inside the profiled step:

| Bucket | EP4 | EP8 | Read |
| --- | ---: | ---: | --- |
| ROCm GEMM / `Cijk*` kernels | `3.93 s` | `2.83 s` | EP4 owner-rank local VJP is GEMM-heavier |
| MORI combine kernels | `1.21 s` | `2.38 s` | EP8 combine is now a first-class wall |
| Attention fwd/bwd kernels | `2.09 s` | `2.10 s` | similar across EP4/EP8 in this guarded path |
| Copy/cat/fill/gather kernels | `1.98 s` | `1.44 s` | still meaningful movement/bookkeeping |
| RCCL/NCCL kernels | `1.42 s` | `1.03 s` | not the largest EP8 bucket in this trace |

Dense-linear label split for the raw `MmBackward0` family:

| Bucket | EP4 | EP8 | Read |
| --- | ---: | ---: | --- |
| Shared expert backward linear | `1.96 s` | `2.01 s` | real dense shared-expert VJP work |
| Attention projection backward linear | `0.22 s` | `0.21 s` | not the main `MmBackward0` wall |
| Router gate backward linear | `0.01 s` | `0.01 s` | negligible |
| Profiled-linear CPU span | `12.13 s` | `18.02 s` | includes nesting/waits; not GPU GEMM time |

Read: EP setting changes the latency story exactly as suspected. EP8 reduces local owner-rank VJP math versus EP4, but it exposes MORI backward combine tail and imbalance. EP4 is more local-GEMM-heavy; EP8 is more combine-tail-heavy. Attention fwd/bwd kernels are about `2.1 s` rank-median in both profiles, so in the current guarded systems path attention is not the dominant new regression versus MoE/combine/loss-side work. The surprising extra bucket is loss/lm-head chunk forward (`14.63 s` EP4, `20.06 s` EP8 rank-median profiler annotation); keep it separate from MoE/attention until we inspect whether this canary loss path is doing avoidable full-vocab chunk work that final training should shard or fuse.

EP4/EP8 sidecar deep split:

Artifact: [`run_artifacts/full_profile_ep4_ep8_sidecar_deep_split_20260603.json`](run_artifacts/full_profile_ep4_ep8_sidecar_deep_split_20260603.json). This adds opt-in CUDA-event sidecars for profiled DSv4 dense linears and `ChunkedCELoss`, plus the existing MORI sidecars. The sidecar synchronizes profiled regions, so the two runs are attribution evidence, not throughput baselines. Both completed two steps with loss movement: EP8 `12.24233 -> 9.11398`, step 2 `2,874 tok/GPU/s`; EP4 `12.23229 -> 8.85830`, step 2 `3,871 tok/GPU/s`.

MORI routed-backward steady totals use `call_id > 65`, giving `63` calls per rank, approximately one steady profiled step:

| Bucket | EP4 rank-median total | EP8 rank-median total | Read |
| --- | ---: | ---: | --- |
| `combine_dx` | `1.16 s` | `2.72 s` | EP8 combine is the first MoE backward wall |
| local VJP recompute forward | `0.93 s` | `0.71 s` | EP8 smooths owner-rank math |
| local VJP wgrad | `0.76 s` | `0.59 s` | still a real training-GEMM lever |
| local VJP xgrad/reduce | `0.63 s` | `0.48 s` | still needs fused grouped scheduling |
| local VJP W2 dgrad + SwiGLU | `0.38 s` | `0.29 s` | smaller after Triton SwiGLU |
| local pack/scatter | `0.20 s` | `0.20 s` | not the top wall |

Dense/loss sidecar totals cover both steps; the table below normalizes by two for per-step rank-median attribution:

| Bucket | EP4 per step | EP8 per step | Read |
| --- | ---: | ---: | --- |
| chunked CE / lm-head backward region | `~2.40 s` | `~2.35 s` | major loss-side dense bucket |
| loss chunk backward sub-slice | `~2.18 s` | `~2.17 s` | mostly lm-head backward and CE grad |
| lm-head chunk forward | `~0.16 s` | `~0.12 s` | full-vocab forward is smaller than backward |
| attention projection backward linears | `~0.23 s` | `~0.22 s` | not the main `MmBackward0` wall |
| shared expert dense linears | `~0.05 s` | `~0.05 s` | previous broad trace attribution overcounted this |
| router gate linear | `~0.01 s` | `~0.01 s` | negligible |

Read: the next EP8 MoE lever is not another dispatch-only tweak; it is split/overlapped `combine_dx` plus fused/grouped owner-rank VJP math. EP4 is still local-VJP-math-heavy, while EP8 has shifted to combine tail. Separately, the canary's full-vocab chunked loss path is a real multi-second bucket and should be treated as its own optimization branch: loss/vocab parallelism, fused linear+CE, or a sharded loss path. The owner-tagged dense attention projections and shared-expert linears are small after event timing; the earlier `MmBackward0` ambiguity came from async/nested profiler scopes.

Holistic EP8 optimization target ledger:

Artifact: [`run_artifacts/holistic_optimization_targets_ep8_s4096_20260603.json`](run_artifacts/holistic_optimization_targets_ep8_s4096_20260603.json).

The combine probe is not the whole story. Ranking the current `8xMI350`, `S4096`, `MBS=1`, `GBS=128`, `FSDP8/EP8` guarded systems canary by expected end-to-end return gives:

| Rank | Target | Current EP8 evidence | Return read |
| ---: | --- | ---: | --- |
| 1 | MoE backward return skew / overlap | `combine_dx` sidecar `2.72 s/step`; barrier split shows `32.61 ms` pre-barrier vs `0.62 ms` raw combine call | biggest structural wall; start dX return earlier/in chunks and reduce peer-arrival skew |
| 2 | MoE owner-rank VJP math | recompute `0.71 s`, wgrad `0.59 s`, xgrad/reduce `0.48 s`, W2 dgrad+SwiGLU `0.29 s` | feeds the combine wait; needs fused/grouped training GEMM scheduling |
| 3 | Loss/lm-head chunked CE | sidecar `~2.35 s/step`; CE2 no-profile run reaches `3,136-3,137 tok/GPU/s` vs retained CE8 `3,026` | real near-term canary lever; CE2 is the current candidate, while CE4 and CE1 regress |
| 4 | Attention / final sparse MLA+LI | broad trace attention kernels `~2.10 s` | not the current guarded-path top bucket, but still mandatory for the final sparse path |
| 5 | Copy/gather/fill/bookkeeping | broad trace `~1.44 s` | meaningful but below MoE/loss for near-term return |
| 6 | RCCL/NCCL | broad trace `~1.03 s` | not the largest single-node bucket before MoE/loss changes |

Chunked CE A/B, same retained 8x MORI settings and no profiling:

| Loss chunks | Run id | Step 2 loss | Step 2 grad norm | Step 2 tok/GPU/s | Peak logged memory | Read |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8 | retained registered combine baseline | `7.21987` | `25.125` | `3,026` | `248.68 GiB` | current retained baseline |
| 4 | `torchtitan_dsv4_chunked_ce4_ep8_s4096_gbs128_20260603` | `8.47222` | `48.25` | `2,989` | `226.50 GiB` | passes but regresses slightly |
| 2 | `torchtitan_dsv4_chunked_ce2_ep8_s4096_gbs128_20260603` | `8.02047` | `45.00` | `3,136-3,137` | `139.57 GiB` | positive `~3.6%` throughput uplift vs CE8 |
| 1 | `torchtitan_dsv4_chunked_ce1_ep8_s4096_gbs128_20260603` | `9.13966` | `22.375` | `3,002` | `143.68 GiB` | passes, but regresses versus CE8 and CE2 |

CE2 launcher-default validation: [`run_artifacts/torchtitan_dsv4_ce2_default_retained_8x_s4096_gbs128_20260603.json`](run_artifacts/torchtitan_dsv4_ce2_default_retained_8x_s4096_gbs128_20260603.json). The retained 8x target-shape run left `CANARY_CHUNKED_CE_NUM_CHUNKS` unset, the launcher header printed `chunked_ce_num_chunks=2`, and the two-step canary passed with loss movement `12.20713 -> 8.75234`, step 2 grad norm `20.75`, step 2 `3,103 tok/GPU/s`, `21.12 s` implied step time, peak logged memory `166.47 GiB`, and clean process-group teardown. This is slightly below the prior manual CE2 A/B but still improves the old CE8 retained baseline by about `2.5%`.

Read: the biggest optimization target is not simply "MoE or attention." For the current distributed canary, the top ranked returns are (1) MoE backward overlap/skew, (2) owner-rank VJP math that creates the skew, and (3) the full-vocab loss/lm-head path. Attention is still a required final-path blocker because the current systems canary is guarded/prototype sparse MLA, but the current guarded EP8 profile does not rank attention above MoE/loss. CE2 is now the retained loss-chunk setting for the exact 8x target shape; CE1 was rerun once the 8-GPU node was idle and passed, but it regressed to `3,002 tok/GPU/s`.

MORI `dX` combine mechanism probe:

Artifacts: [`run_artifacts/mori_dx_combine_kernel_type_ep8_20260603.json`](run_artifacts/mori_dx_combine_kernel_type_ep8_20260603.json) and [`run_artifacts/mori_dx_combine_barrier_diag_ep8_s4096_gbs8_20260603.json`](run_artifacts/mori_dx_combine_barrier_diag_ep8_s4096_gbs8_20260603.json).

Standalone target-shape EP8 probe, `tokens=4096`, `hidden=4096`, `experts=256`, `topk=6`, random routes, exact MORI unweighted-combine parity:

| Variant | Rank-median total | Rank range | Launch | Read |
| --- | ---: | ---: | --- | --- |
| IntraNode registered | `0.438 ms` | `0.424-0.441 ms` | `[64,32,8]` | retained TorchTitan mode raw floor |
| IntraNode external | `0.242 ms` | `0.240-0.248 ms` | `[64,32,8]` | external-input raw floor is lower in isolation |
| AsyncLL external | `0.164 ms` | `0.160-0.168 ms` | `[64,0,8]` | promising split-send/recv migration probe |

Integrated EP8 barrier diagnostic, `S4096`, `GBS8`, full AC, registered combine, retained `[64,32,8]`, with opt-in `TORCHTITAN_MORI_AITER_BARRIER_BEFORE_COMBINE_DX=1`, completed two steps with loss movement `12.29093 -> 7.53290` and step 2 `2,835 tok/GPU/s`:

| Stage | Rank-median | Rank range | Read |
| --- | ---: | ---: | --- |
| `backward.combine_dx.pre_barrier` | `32.61 ms` | `0.09-36.44 ms` | peer-arrival skew is the wall |
| `backward.combine_dx.call` | `0.62 ms` | `0.58-0.65 ms` | raw registered combine matches the standalone floor class |
| `backward.local_vjp.padded` | `29.41 ms` | `20.30-60.32 ms` | routed owner-rank VJP skew feeds the combine wait |
| `backward.combine_dtopk.call` | `0.048 ms` | `0.046-0.049 ms` | router-weight combine is not material |

Read: the previous integrated `combine_dx.call` tens-of-ms number is mostly synchronization/arrival tail hidden inside MORI combine, not raw dX-combine bandwidth. The next EP8 MoE optimization should therefore target overlap and skew reduction: start dX return earlier or in chunks, reduce owner-rank VJP imbalance, and only then consider AsyncLL split-send/recv as a TorchTitan autograd migration. More blind `[block,rdma,warp]` sweeps are unlikely to close the real wall.

AsyncLL `dX` overlap contract:

Artifact: [`run_artifacts/mori_asyncll_dx_overlap_contract_20260603.json`](run_artifacts/mori_asyncll_dx_overlap_contract_20260603.json).

Source inspection makes the next implementation branch concrete. MORI exposes `combine_send(...)` / `combine_recv(...)`, but `combine_recv` is supported only for `EpDispatchCombineKernelType.AsyncLL`, and the returned output tensor view from `combine_send` must stay live until `combine_recv` completes. The shipped AsyncLL tuning table is not directly promotable for this canary because it is `ep16`, `hidden_dim=7168`, and MI355-labeled, while the current target is EP8 hidden `4096`.

Current TorchTitan routed backward still computes the full padded local VJP and then calls synchronous `grad_op.combine(...)`. Inside the padded VJP, W2/W13 wgrads run before `w13_xgrad_gemm_reduce`, so `dX` is produced late. The useful staged branch is: after `swiglu_backward`, compute `w13_xgrad_gemm_reduce` and scatter `dpacked_x` first, launch AsyncLL `combine_send`, then compute W2/W13 wgrads while the dX transfer is in flight, call `combine_recv`, and only then slice/read `dx_full`. This is a real implementation contract, not a validated speedup yet; correctness gates are small 4x/8x parity smokes, then S4096 `GBS8` timing, then the no-profile `GBS128` canary.

AsyncLL staged `dX` implementation smoke:

Artifact: [`run_artifacts/mori_asyncll_dx_staged_smokes_20260603.json`](run_artifacts/mori_asyncll_dx_staged_smokes_20260603.json).

The opt-in TorchTitan branch now makes x, grad, and top-k dummy dispatch route-consistent under `TORCHTITAN_MORI_AITER_DX_RETURN_MODE=asyncll_staged`, uses AsyncLL `combine_send`/`combine_recv` for both `dX` and `dTopk`, and gives AsyncLL ops a derived receive budget of `max_tokens * top_k * 2` when `TORCHTITAN_MORI_AITER_MAX_TOTAL_RECV_TOKENS` is unset.

| Run | Shape | Guard | Result | Read |
| --- | --- | --- | --- | --- |
| `torchtitan_dsv4_asyncll_dx_4x_s128_smoke_topkroute_20260603` | `4xMI350`, `S128`, `GBS4`, no AC | strict source-position check | pass, loss `12.34595`, grad norm `12.0`, clean teardown | top-k route consistency is fixed at small shape |
| `torchtitan_dsv4_asyncll_dx_4x_s4096_gbs4_smoke_topkroute_20260603` | `4xMI350`, `S4096`, `GBS4`, full AC | strict | fail, AsyncLL `maxTotalRecvTokens` assert | long routes need explicit/derived receive capacity |
| `torchtitan_dsv4_asyncll_dx_4x_s4096_gbs4_smoke_recvbudget_20260603` | `4xMI350`, `S4096`, `GBS4`, full AC | strict | fail, `grad_output` source-position set mismatch | capacity fixed; remaining blocker is AsyncLL source-position metadata/order |
| `torchtitan_dsv4_asyncll_dx_4x_s4096_gbs4_smoke_assumeorder_20260603` | `4xMI350`, `S4096`, `GBS4`, full AC | `assume_identical_dispatch_order=true` | pass, loss `12.29587`, grad norm `8.3125`, rank memory `71.17-93.15 GiB`, clean teardown | data path can complete, but this is diagnostic-only until parity validates order |

Read: this moves the top-ranked holistic lever from "paper contract" to a smokeable implementation branch, but it is not yet a retained optimization. The next gate is deterministic parity for AsyncLL staged `dX`/`dTopk` versus the retained sync/IntraNode backward, or a reliable long-sequence AsyncLL source-position map. Only after that should we run no-profile `S4096` throughput A/Bs.

AsyncLL route-order localization:

Artifacts: [`run_artifacts/mori_asyncll_route_order_s128_4x_20260603.json`](run_artifacts/mori_asyncll_route_order_s128_4x_20260603.json), [`run_artifacts/mori_asyncll_route_order_s4096_4x_20260603.json`](run_artifacts/mori_asyncll_route_order_s4096_4x_20260603.json), [`run_artifacts/mori_asyncll_route_order_s4096_h4096_topkh6_4x_20260603.json`](run_artifacts/mori_asyncll_route_order_s4096_h4096_topkh6_4x_20260603.json), and the retained target-shape check [`run_artifacts/mori_asyncll_route_order_s4096_h4096_topkh6_8x_decoded_20260603.json`](run_artifacts/mori_asyncll_route_order_s4096_h4096_topkh6_8x_decoded_20260603.json). Probe: [`tools/dsv4_runtime_probe/probe_mori_asyncll_route_order.py`](tools/dsv4_runtime_probe/probe_mori_asyncll_route_order.py).

The probe dispatches three identical routes through independent AsyncLL handles for saved `x`, `grad_output`, and top-k dummy payloads, with payload column 0 encoding source token identity. The target-shape `8xMI350`, `S4096`, `hidden=4096`, `topk_hidden=6`, random top-k run reports:

| Check | Result |
| --- | --- |
| ordered route keys equal across x/grad/topk | `false` |
| route-key multisets equal across x/grad/topk | `true` |
| route keys equal after raw `src_pos` sort | `false` |
| route keys equal after decoded `src_pos` sort | `false` |
| raw `src_pos` multisets equal | `true` |
| decoded `src_pos` matches payload source token | `true` |
| recv rows by rank | `[19634, 19421, 19642, 19604, 19495, 19395, 19672, 19523]` |

Read: AsyncLL separate handles conserve the same routed rows, so this is not a dropped-row or receive-capacity problem. However, they do not preserve row order, and source-token-only metadata is not enough to reconstruct the route-key order because duplicate source-token routes exist. Therefore `TORCHTITAN_MORI_AITER_ASSUME_IDENTICAL_DISPATCH_ORDER=true` remains diagnostic-only, and the current source-position-only reorder is not a production-safe long-sequence alignment rule. The next implementation branch should align by a stronger key such as `(decoded source token, packed top-k ids)` or use a shared MORI route/order handle before retrying staged `dX` as a throughput optimization.

AsyncLL route-key alignment gate:

Artifact: [`run_artifacts/mori_asyncll_route_key_alignment_20260603.json`](run_artifacts/mori_asyncll_route_key_alignment_20260603.json).

The TorchTitan bridge now has a diagnostic route-key reorder using `(src_pos, packed_ids)` plus improved first-mismatch reporting. The direct launcher also exposes `CANARY_DEBUG_SEED` through `--debug.seed` so A/B route gates can be pinned. Local `py_compile`, remote in-image `py_compile`, and `bash -n` on the launcher all pass.

| Run | Shape | Seed | `dX` mode | Result | Read |
| --- | --- | ---: | --- | --- | --- |
| `torchtitan_dsv4_asyncll_dx_routekey_debug_4x_s4096_gbs4_20260603` | `4xMI350`, `S4096`, `GBS4`, full AC | unset | `asyncll_staged` | pass, loss `12.31336`, grad norm `11.0625` | useful smoke, not parity |
| `torchtitan_dsv4_sync_dx_routekey_ref_4x_s4096_gbs4_20260603` | same | unset | `sync` | pass, loss `12.25010`, grad norm `10.6875` | retained path healthy; unseeded pair differs |
| `torchtitan_dsv4_asyncll_dx_routekey_seeded_4x_s4096_gbs4_20260603` | same | `20260603` | `asyncll_staged` | fail, `grad_output MORI route-key set mismatch` | strict route-key gate rejects current split-handle AsyncLL path |
| `torchtitan_dsv4_sync_dx_seeded_ref_4x_s4096_gbs4_20260603` | same | `20260603` | `sync` | pass, loss `12.23368`, grad norm `11.0` | retained registered-combine path remains valid under same seed |

The seeded failure is informative. Ranks 2/3 show the source side sorted route key has repeated zero-packed ids such as `[0,0,0,0,0,0,0]`, while the destination side has real packed top-k ids for nearby source positions, e.g. `[1,21,6,155,97,161,219]` and `[0,37,120,199,123,210,219]`. That means the current `(src_pos, packed_ids)` reconstruction is still not a valid integrated learned-router contract for independently dispatched AsyncLL handles. The likely issue is that the saved `x` dispatch metadata and the grad/top-k dispatch metadata are not carrying the same per-route key, even when source-position multisets are conserved.

Read: reject the current split-handle AsyncLL staged `dX` branch for production. The next overlap attempt needs a real route metadata contract: either reuse/share the original MORI route/order handle across saved `x`, `grad_output`, and top-k dummy dispatches, or carry explicit dispatched payload/source metadata that makes their route keys identical. Do not run no-profile S4096 throughput A/Bs on this AsyncLL branch until the seeded route-key gate passes. The retained EP8 canary path remains sync registered combine with routed padded VJP.

AsyncLL alignment-mode retry:

Artifact: [`run_artifacts/mori_asyncll_align_mode_srcpos_retry_20260603.json`](run_artifacts/mori_asyncll_align_mode_srcpos_retry_20260603.json).

I added `TORCHTITAN_MORI_AITER_ASYNCLL_ALIGN_MODE` / `CANARY_MORI_AITER_ASYNCLL_ALIGN_MODE` so the staged path can test `route_key` versus `src_pos` alignment without editing source. Local `py_compile`, launcher `bash -n`, and `print-command` passed. The source-position retry used the same `4xMI350`, `S4096`, `GBS4`, full-AC staged AsyncLL gate and explicitly printed `mori_aiter_asyncll_align_mode=src_pos`.

| Run | Align mode | Shape | Result | Read |
| --- | --- | --- | --- | --- |
| `torchtitan_dsv4_asyncll_staged_dx_4x_s128_smoke_20260603` | implicit `src_pos` default | `4x`, `S128`, `GBS4`, no AC | pass, loss `12.25415`, grad norm `12.25`, clean teardown | staged path remains smokeable at short sequence |
| `torchtitan_dsv4_asyncll_staged_dx_4x_s4096_gbs4_smoke_20260603` | `route_key` | `4x`, `S4096`, `GBS4`, full AC | fail, `grad_output MORI route-key set mismatch` | strict route-key is not valid for independent handles |
| `torchtitan_dsv4_asyncll_srcpos_dx_4x_s4096_gbs4_smoke_20260603` | `src_pos` | `4x`, `S4096`, `GBS4`, full AC | fail, `grad_output MORI source-position set mismatch` | source-only alignment is also not valid at long sequence |

Read: this closes the tempting "just sort by source token" branch. The current independent-handle AsyncLL staged path is rejected for production. The next overlap attempt needs shared MORI route/order metadata or explicit per-dispatched-row route identity, then a seeded S4096 parity gate. Until then, the retained EP8 canary remains sync registered combine with routed padded VJP.

## Full-Step Critical Path

Artifact: [`run_artifacts/holistic_critical_path_ladder_ep8_s4096_20260603.json`](run_artifacts/holistic_critical_path_ladder_ep8_s4096_20260603.json).

The retained no-profiler 8xMI350 target-shape canary is now the headline systems anchor: `S4096`, `GBS128`, `FSDP8`, `EP8`, full activation checkpointing, MORI routed padded VJP, registered backward combine, Triton SwiGLU, and launcher-default CE2 pass two steps at `3,103 tok/GPU/s`, `21.12 s/step`, and `58.71 TFLOPs/GPU` with loss movement `12.20713 -> 8.75234`. Under the updated `277 TFLOPs/GPU` target this is `21.2%` of target and needs about `4.72x` speedup, so it is a guarded/prototype systems baseline, not a passing performance anchor or final sparse-MLA/LI result.

Current 8x performance scoreboard: [`run_artifacts/torchtitan_current_8x_performance_scoreboard_20260603.json`](run_artifacts/torchtitan_current_8x_performance_scoreboard_20260603.json), generated by [`tools/dsv4_runtime_probe/summarize_current_8x_performance.py`](tools/dsv4_runtime_probe/summarize_current_8x_performance.py), disambiguates the retained default, the nearby manual CE2 A/B, and the older guarded dense-fallback systems baseline. The retained launcher-default anchor is `3,103 tok/GPU/s`, `21.12 s/step`, `58.71 TFLOPs/GPU`, and `2.54%` MI350 MFU. The updated pass/fail target is `277 TFLOPs/GPU`, which corresponds to about `14,640 tok/GPU/s`, `117,122 tok/node/s`, and `4.48 s/step` only under the retained FLOPs/token accounting. The nearby manual CE2 A/B reached `3,136-3,137 tok/GPU/s` or about `59.35 TFLOPs/GPU`, while the older `204.865 TFLOPs/GPU` line belongs to the separate `10,827 tok/GPU/s` guarded dense-fallback systems baseline. The retained and old runs have nearly identical estimated TFLOP/token accounting (`0.018920` vs `0.018922`), so the lower retained TFLOPs is explained by lower retained throughput, not a units mismatch; both retained and old dense-fallback baselines remain below the updated target.

The profile evidence should be read as a critical-path ranking, not an additive wall-time sum. Torch profiler ranges, GPU user annotations, CUDA event sidecars, and kernel-family summaries have different synchronization scopes. The current ladder is:

| Rank | Bucket | Evidence | Highest-return action |
| --- | --- | --- | --- |
| 1 | MoE backward `dX` return skew/overlap | EP8 `combine_dx` is about `2.7 s/rank/step`; barrier diagnostic shows raw registered combine around `0.62 ms/call` while pre-combine wait is around `32.61 ms/call` | Start `dX` return earlier/in chunks, reduce owner-rank VJP skew, require shared route/order metadata before retrying AsyncLL |
| 2 | Routed MoE owner-rank VJP GEMMs | EP8 local VJP GEMM family is about `1.9-2.1 s/rank/step`; standalone owner-rank floor is still `6-7 ms` per local VJP | Replace padded/recompute VJP with fused or grouped training GEMM and a better AITER/MORI forward-backward intermediate contract |
| 3 | Loss/lm-head chunked CE | CE2 improves retained step 2 from `3,026` to `3,103 tok/GPU/s`; CE1/CE4 regress; sidecar puts the CE/lm-head backward region around `~2.35 s/step` | Keep CE2 for this exact 8x canary shape; do not let canary-relative loss size displace MoE/attention kernel work |
| 4 | Attention and final sparse MLA/LI | Attention kernel family is about `2.1 s/rank/step`; current path still has guarded/prototype sparse backward | Continue compressed dKV many-to-one accumulation and final fused sparse MLA/LI backward work |
| 5 | FSDP/RCCL/NCCL and copy/bookkeeping | NCCL kernel family around `1.0 s/rank/step`, copy/gather/fill around `1.4 s/rank/step`, FSDP ranges can overlap | Keep tags and re-rank after MoE and attention changes alter overlap |
| 6 | Shared expert dense backward scope check | Broad trace can label shared expert backward around `1.9-2.0 s/rank/step`, but the later dense/loss sidecar puts owner-tagged shared expert linears near `~0.05 s/step` | Keep labels in future profiles, but do not prioritize shared expert dense tuning without stronger exclusive GPU-event evidence |

Read: the biggest optimization target is not simply "MoE or attention." The current highest-return path is MoE backward overlap/skew plus fused owner-rank VJP, while CE2 remains the retained canary loss setting and final sparse MLA/LI remains mandatory for production correctness/perf. More blind MORI launch tuple sweeps, pure SwiGLU-only tuning, shared-expert tuning from broad non-exclusive labels, or independent-handle AsyncLL throughput A/Bs are lower-return or rejected until stronger evidence exists.

AsyncLL route-identity follow-up: [`run_artifacts/mori_asyncll_route_flatten_probe_contract_20260603.json`](run_artifacts/mori_asyncll_route_flatten_probe_contract_20260603.json) records the next MoE-overlap gate. Source inspection of MORI shows `get_dispatch_src_token_pos()` is a flat source input-row id, not a selected top-k route-slot id. That explains why normal `topk=6` learned-router rows can still be ambiguous: one source token may route multiple expert slots to the same destination rank, so `src_pos` and `(src_pos, full top-k ids)` are not guaranteed unique per row. The standalone AsyncLL probe now has `--route-flatten`, which dispatches logical `[tokens, topk]` as `[tokens * topk, 1]`, uses MORI `decode_send_flat_idx()` for metadata decoding, and reports duplicate `src_pos`/route-key rows. The first probe-only 4xMI350 S4096 gate passed: [`run_artifacts/mori_asyncll_route_flatten_s4096_h4096_topk6_4x_20260603.json`](run_artifacts/mori_asyncll_route_flatten_s4096_h4096_topk6_4x_20260603.json) has `dispatch_tokens_per_rank=24576`, `dispatch_topk=1`, zero duplicate `src_pos` rows on all four ranks, zero duplicate route-key rows, and both x-vs-grad and x-vs-topk route keys match after source-position sorting. This promotes flattened `topk=1` routing to the next seeded sync-vs-AsyncLL training parity branch; it is not a throughput result yet.

Route-flattened AsyncLL training gate: [`run_artifacts/mori_asyncll_flattened_training_gate_20260603.json`](run_artifacts/mori_asyncll_flattened_training_gate_20260603.json) carries that route identity into the TorchTitan autograd bridge. `sources/wip/torchtitan/torchtitan/models/common/mori_aiter_moe.py` now has a guarded `TORCHTITAN_MORI_AITER_DX_RETURN_MODE=asyncll_flattened` branch that expands `[T,K]` into `[T*K,1]`, creates flattened AsyncLL x/grad/top-k ops with `top_k=1` and `max_tokens=T*K`, computes the padded owner-rank VJP in route rows, combines dX back as `[T*K,H] -> [T,H]`, and combines dTopk back as `[T*K,1] -> [T,K]`. Local `py_compile` and launcher `check-source` pass. The 4xMI350 `S128`, `GBS4`, no-AC smoke passes with loss `12.20844` and grad norm `12.1875`; the paired sync reference also passes with grad norm `12.1875`. The first `S4096`, `GBS4`, full-AC flattened attempt with `MORI_SHMEM_HEAP_SIZE=32G` fails only on MORI static heap capacity (`1.61 GB` allocation against a used `32G` heap), not on the old source-position/route-key mismatch. Retrying at `MORI_SHMEM_HEAP_SIZE=96G` passes one full-AC step with loss `12.27059`, grad norm `11.0625`, and `127 tok/GPU/s`; the same-seed sync context run passes with loss `12.27053`, grad norm `11.0625`, and `215-217 tok/GPU/s`. Read: the per-route identity contract is now validated inside TorchTitan at long sequence, but route flattening is a diagnostic stepping stone, not a retained performance path. The production overlap design should carry shared route/order metadata or compact per-route identity without duplicating `[T*K,H]` x/grad dispatches or requiring a `96G` MORI heap for this tiny `GBS4` gate.

## Latest EP4/EP8 Full-Run Profile

Artifacts: [`run_artifacts/torchtitan_latest_ce2_ep4_ep8_profile_summary_20260603.json`](run_artifacts/torchtitan_latest_ce2_ep4_ep8_profile_summary_20260603.json), filtered parser output [`run_artifacts/torchtitan_latest_ce2_ep4_ep8_full_profile_breakdown_step2_filtered_20260603.json`](run_artifacts/torchtitan_latest_ce2_ep4_ep8_full_profile_breakdown_step2_filtered_20260603.json), and MORI step-2 skew analyzer [`run_artifacts/torchtitan_latest_ce2_ep4_ep8_mori_step2_skew_20260603.json`](run_artifacts/torchtitan_latest_ce2_ep4_ep8_mori_step2_skew_20260603.json).

Both profiles use the latest retained TorchTitan direct canary shape: `MBS=1`, `S=4096`, 4 DSv4-style layers, full activation checkpointing, CE2, MORI routed padded VJP, registered backward combine, and Triton SwiGLU. This is the guarded/default attention canary path, not the final sparse-MLA/LightningIndexer override.

| Run | Step wall note | MoE local VJP | MoE dX combine | Loss/CE sidecar | Attention projection linears | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| EP4/FSDP4, GBS64 | profiled rank0 step `19.22 s`, includes `1.94 s` Kineto export | `2.52 s/rank` step total, `41.35 ms/call` | `1.25 s/rank` step total, `14.20 ms/call` | `0.584 s/rank` | `0.243 s/rank` | local owner-rank VJP is the larger MoE wall |
| EP8/FSDP8, GBS128 | profiled rank0 step `24.31 s`, includes `1.99 s` Kineto export; retained no-profile anchor remains `3103 tok/GPU/s`, `21.12 s/step` | `2.01 s/rank` step total, `30.78 ms/call` | `1.78 s/rank` step total, `23.81 ms/call` | `0.459 s/rank` | `0.235 s/rank` | EP8 smooths local VJP but dX return/arrival wait grows |

Kernel-family profiler medians are useful for ranking but not additive wall time: EP8 shows about `2.50 s/rank` GEMM-family, `1.20 s/rank` copy, `1.19 s/rank` RCCL/NCCL, and `5.60 s/rank` other cumulative GPU events across streams/ranges. The actionable ladder is: reduce/overlap MORI dX return and owner-rank VJP skew first, then improve fused/padded routed VJP math, then keep final sparse-MLA/LI backward work as the attention-specific production gap.

MORI skew refinement: the corrected step-2 analyzer dedupes the `mori_timing/` and `timing/mori/` symlinked views and keeps exactly `64 * ranks` records (`256` EP4, `512` EP8). On EP8, per-record `recv_count` correlates strongly with local owner VJP time (`r=0.743`) but negatively with `combine_dx` time (`r=-0.492`), and local VJP also negatively correlates with combine time (`r=-0.564`). The worst EP8 call (`call_id=68`) has rank 6 receiving `28,902` rows and spending `151.30 ms` in local VJP while rank 7 receives only `4,265` rows but spends `139.38 ms` in `combine_dx`; rank totals show the same inversion (`rank6 local_vjp=2.464 s`, `combine_dx=1.303 s`; `rank7 local_vjp=1.603 s`, `combine_dx=2.146 s`). EP4 shows the same pattern at smaller scale. Read: the registered `combine_dx` wall is mostly peer-arrival skew induced by imbalanced owner-rank VJP, not raw MORI combine math. The next MoE optimization should either reduce the local VJP tail on high-recv owner ranks or return `dX` earlier/in route chunks with shared route/order metadata; another launch-parameter-only combine sweep is low-return.

Skew benefit model: [`run_artifacts/torchtitan_latest_ce2_ep4_ep8_mori_skew_benefit_model_20260603.json`](run_artifacts/torchtitan_latest_ce2_ep4_ep8_mori_skew_benefit_model_20260603.json) fits `combine_dx ~= raw_combine_floor + max(0, max(local_vjp) - local_vjp_rank)` on the same step-2 sidecars. Fit is tight (`r=0.997` EP8, median absolute error `0.28 ms`; `r=0.99995` EP4, median absolute error `0.14 ms`). The inferred EP8 MORI critical path is `3.762 s`, made of `3.734 s` max owner-local VJP plus only `28.5 ms` raw combine floor (`0.76%`). Scenario reads: eliminating raw combine compute alone can save only `28.5 ms`; a uniform `25%` owner-VJP speedup saves `933 ms`; a `25%` speedup only on ranks above the per-call median receive count saves `823 ms`; capping each call to p75 local VJP saves `1.268 s`. On the retained no-profiler `21.12 s/step` anchor, those modeled EP8 saves are roughly `0.1%`, `4.6%`, `4.1%`, and `6.4%` upper-bound step-throughput levers respectively. This narrows the next branch: first try owner-rank VJP fusion/load-tail reduction, while AsyncLL/chunked dX should be pursued only when it can overlap the same per-call VJP tail with shared route/order metadata.

Owner-VJP tail component model: [`run_artifacts/torchtitan_latest_ce2_ep4_ep8_mori_vjp_tail_components_20260603.json`](run_artifacts/torchtitan_latest_ce2_ep4_ep8_mori_vjp_tail_components_20260603.json) selects the per-call max-local-VJP rank and recomputes critical-path savings after reducing individual VJP substages. On EP8, the max-owner tail ranks are evenly split across ranks `1`, `2`, `6`, and `7` (`16` calls each) and have high receive counts (`p50=25,171.5`, `p90=28,870.7`). Tail composition is recompute-forward `1.172 s` (`31.4%`), wgrad `0.900 s` (`24.1%`), xgrad/reduce `0.754 s` (`20.2%`), W2-dgrad+SwiGLU `0.491 s` (`13.1%`), pack/scatter `0.348 s` (`9.3%`), and dTopkWeight `0.061 s` (`1.6%`). Scenario reads: removing recompute-forward from the owner-VJP tail would save `1.172 s`; removing wgrad saves `0.900 s`; removing xgrad/reduce saves `0.754 s`; even removing all pack/scatter saves only `0.348 s`. A realistic first kernel contract should therefore target the recompute+GEMM family first: either save/expose the right forward intermediates inside AITER/CK, or fuse recompute, wgrad, and xgrad/reduce into a compact grouped training-GEMM backward. Dispatch packing cleanup alone is too small to be the lead branch.

Owner-VJP saved-intermediate contract decision: [`run_artifacts/torchtitan_mori_owner_vjp_saved_intermediate_contract_20260603.json`](run_artifacts/torchtitan_mori_owner_vjp_saved_intermediate_contract_20260603.json). Source inspection of `mori_aiter_moe.py` shows the current forward saves only `x`, local expert weights, `topk_weight`, and `topk_ids`; backward then recomputes route-local packing, W1/W3 gate/up, SwiGLU hidden, W2 output for `dTopkWeight`, W2 dgrad, W1/W3 wgrad, and W1/W3 xgrad/reduce. The public AITER `fused_moe` surface returns only the final MoE output; its two-stage path can expose post-SwiGLU `a2` hidden, but not gate pre-activation, up branch, or unweighted compact expert output. The standalone EP8 owner-rank probe says why that matters: retained Triton-SwiGLU recompute local VJP is `6.577 ms`; public stage1-hidden-only is `6.591 ms` (neutral/regressed); stage1 hidden plus unweighted output is `5.859 ms`; ideal gate/up/hidden is `5.388 ms`; and ideal gate/up/hidden plus unweighted output is `4.774 ms`. Decision: do not lead with a TorchTitan Python-side "save AITER stage1 hidden only" path. The next MoE branch should define an AITER/CK training backward contract that exposes/preserves gate, up, hidden, and optionally unweighted compact output, or implement a fused compact grouped-GEMM owner VJP that recomputes gate/up once internally and reuses them across W2 dgrad, SwiGLU backward, W1/W3 wgrad, and W1/W3 xgrad/reduce.

AITER stage1 capture probe: [`run_artifacts/aiter_stage1_capture_s4096_h4096_i2048_e64_top6_20260605.json`](run_artifacts/aiter_stage1_capture_s4096_h4096_i2048_e64_top6_20260605.json), generated by [`tools/dsv4_runtime_probe/probe_aiter_fused_moe_stage1_capture.py`](tools/dsv4_runtime_probe/probe_aiter_fused_moe_stage1_capture.py), validates the source-level contract above on one MI350. Shape is local owner-rank style: `tokens=4096`, `hidden=4096`, `intermediate=2048`, `experts=64`, `topk=6`, BF16, random top-k. AITER selects `run_1stage=false`, `ck_moe_stage1`, `ck_moe_stage2_fwd`, and `block_m=128`. Manual `moe_sorting -> stage1 -> stage2` matches public `fused_moe` with `max_abs_diff=3.8147e-6`, `mean_abs_diff=1.16e-9`, finite outputs, and cosine distance `5.96e-8`. The manual two-stage capture path is not slower in this probe: median `2.292 ms` versus public fused MoE median `2.456 ms` after warmup, excluding JIT build. Captured stage1 is post-SwiGLU hidden `[4096,6,2048]`, BF16, `96 MiB`. Read: exposing/saving stage1 is technically viable without replacing the fast AITER schedule, but it is still only a partial training state. It can avoid recomputing hidden for W2 dgrad/wgrad, but raw gate/up and per-route W2 output are still missing, so the next production branch remains an AITER/CK training backward contract or fused compact grouped owner VJP that makes those tensors available or recomputes them once inside the owner kernel.

Owner-VJP candidate projection: [`run_artifacts/torchtitan_mori_owner_vjp_candidate_projection_20260603.json`](run_artifacts/torchtitan_mori_owner_vjp_candidate_projection_20260603.json), generated by [`tools/dsv4_runtime_probe/model_mori_owner_vjp_candidate_projection.py`](tools/dsv4_runtime_probe/model_mori_owner_vjp_candidate_projection.py), maps standalone owner-rank VJP latencies onto the retained no-profiler EP8 anchor (`3,103 tok/GPU/s`, `21.12 s/step`, `58.71 TFLOPs/GPU`) using the MORI step-2 tail model. This is a model from existing sidecars, not a new training run. Public stage1-hidden-only remains rejected (`6.591 ms`, projected no gain). Stage1 hidden plus unweighted output (`5.859 ms`) projects only to `3,164 tok/GPU/s`. Ideal gate/up/hidden (`5.388 ms`) projects to `3,205 tok/GPU/s`; ideal gate/up/hidden plus output (`4.774 ms`) projects to `3,261 tok/GPU/s`. A fused owner-VJP candidate around `4.5 ms` projects to `3,286 tok/GPU/s`, and `4.0 ms` projects to `3,334 tok/GPU/s`. Promotion gate: a candidate must beat the `5.86 ms` public-stage1-plus-output point before integration is worth the churn; a serious production target is `4.0-4.5 ms`, followed by 4x/8x CE2 canary reruns and fresh MORI sidecar ranking.

## Reference Gate Status

Artifact: [`run_artifacts/dsv4_reference_gate_status_20260603.json`](run_artifacts/dsv4_reference_gate_status_20260603.json), generated by [`tools/dsv4_runtime_probe/summarize_reference_gate_status.py`](tools/dsv4_runtime_probe/summarize_reference_gate_status.py).

Read: the reference-first gate is still incomplete even though the local clean-reference evidence is stronger. We have the HF DeepSeek-V4-Flash config/shape contract, a local PyTorch full-layer reference harness with sparse MLA, LightningIndexer, routed MoE, the new `DeepSeekV4HyperConnectionLayerRef`, tiny float64 attention/layer gradcheck, S128 MI350 BF16 sparse-MLA mode parity, small-shape EP MoE/MORI+AITER gradient-semantics smokes, an MI350 full Flash-dimension rank-3 `B=1 S=1` local PyTorch oracle smoke for FP32 plus BF16, and now an MI350 full Flash-dimension rank-4 Hyper-Connection local oracle smoke for FP32 plus BF16. The harness has a first-class `flash` preset matching the HF Flash dimensions, and the retained Flash helper uses `--preset flash` instead of a long override list. What is not yet persisted is deterministic HF matched-weight numerical full-layer parity or S4096 reference coverage. Do not claim HF validation until the one-layer HF-vs-local numerical parity gate is recorded.

HF architecture parity preflight: [`run_artifacts/dsv4_hf_reference_parity_preflight_20260603.json`](run_artifacts/dsv4_hf_reference_parity_preflight_20260603.json), generated by [`tools/dsv4_runtime_probe/audit_hf_reference_parity_preflight.py`](tools/dsv4_runtime_probe/audit_hf_reference_parity_preflight.py), made the HF blocker sharper before the local HC oracle was added. Current HF repo state is `deepseek-ai/DeepSeek-V4-Flash` sha `6976c7ff1b30a1b2cb7805021b8ba4684041f136`, last modified `2026-05-06T04:18:09.000Z`; it publishes `inference/model.py` but not repo-root `modeling_deepseek_v4.py` or `configuration_deepseek_v4.py`, and the configured Transformers `v4.57.1` raw paths for those files return `404`. Source-contract comparison shows HF and Ascend both use Hyper-Connections with rank-4 hidden state. The local clean reference now has an HC full-layer smoke, but TorchTitan WIP still needs an HC parity gate and HF matched-weight numerical parity remains unrecorded.

Hyper-Connection primitive reference: [`run_artifacts/dsv4_hyperconnection_reference_smoke_20260603.json`](run_artifacts/dsv4_hyperconnection_reference_smoke_20260603.json), generated by [`tools/dsv4_layer_microbench/verify_hyperconnection_reference.py`](tools/dsv4_layer_microbench/verify_hyperconnection_reference.py), covers the local pure-PyTorch HC boundary copied from the HF/Ascend contract. The smoke passes split/shape checks, small FP64 gradcheck, and MI350 Flash-hidden-size backward smokes for FP32 and BF16 with input shape `[1,1,4,4096]` and output shape `[1,1,4,4096]`.

Flash Hyper-Connection local layer oracle: [`run_artifacts/dsv4_flash_hyperconnection_layer_oracle_20260603.json`](run_artifacts/dsv4_flash_hyperconnection_layer_oracle_20260603.json) records the MI350 `--run-layer --use-hyperconnection --device cuda --dtypes float32,bfloat16 --batch 1 --seqlen 1 --router balanced` gate from [`tools/dsv4_layer_microbench/verify_flash_reference_oracle.py`](tools/dsv4_layer_microbench/verify_flash_reference_oracle.py). The run threads HC around the local attention and MoE block with input/output shape `[1,1,4,4096]`, uses `attention_impl=dense-mask` and `moe_impl=grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad`, and passes finite input plus parameter gradients for both dtypes. FP32 loss is `0.479573` with peak `63.21 GiB`; BF16 loss is `0.478702` with peak `34.68 GiB`. This removes the local full-block HC smoke blocker, but it is not HF matched-weight numerical parity, S4096 reference coverage, TorchTitan HC parity, or production sparse-MLA/LI performance evidence.

Flash preset config oracle: [`run_artifacts/dsv4_flash_reference_oracle_config_20260603.json`](run_artifacts/dsv4_flash_reference_oracle_config_20260603.json), generated by [`tools/dsv4_layer_microbench/verify_flash_reference_oracle.py`](tools/dsv4_layer_microbench/verify_flash_reference_oracle.py), passes config-only parity against the HF Flash contract: hidden `4096`, heads `64`, `head_dim=512`, q/o LoRA ranks `1024`, MoE intermediate `2048`, routed experts `256`, top-k `6`, LI top-k `512`, and CSA/HCA ratio set `{4,128}` all match.

Flash local layer oracle: [`run_artifacts/dsv4_flash_reference_oracle_layer_20260603.json`](run_artifacts/dsv4_flash_reference_oracle_layer_20260603.json) records the MI350 `--run-layer --device cuda --dtypes float32,bfloat16 --batch 1 --seqlen 1 --router balanced` gate from the same tool. The run used `attention_impl=dense-mask` and `moe_impl=grouped-expert-batches-dynamic-compact-reduce-bmm-out-dgrad-dense-wgrad`, output shape `[1,1,4096]`, finite input and parameter gradients for both dtypes, FP32 loss `1.011379` with peak `63.20 GiB`, and BF16 loss `1.011453` with peak `34.67 GiB`. This is a useful local clean-oracle smoke, not HF numerical parity, S4096 reference coverage, or production sparse-MLA/LI performance evidence. Raw remote scratch staging for this oracle was under `/scratch/sonle5/dsv4_pretrain_canary_20260527/reference_oracle_20260603`.

## Current Caveats

- FP4 is simulated with fake-quantized weights and block-scaled dequantization; it is not packed FP4 GEMM.
- Explicit FP4 packed storage helpers now exist for two E2M1 values per byte plus UE8M0 scale bytes, but the training path still dequantizes through PyTorch tensors. Real packed FP4 grouped GEMM is still pending.
- `fp4-packed-sim` caches dequantized FP4 weights to model prepacked weights and uses straight-through gradients; it is a useful performance ceiling/profiling mode, not the final packed FP4 training kernel. The cache validates weak tensor owners to avoid stale entries after data-pointer reuse, chunks large 3D expert prefixes to avoid prepack-time OOM, and caches fixed-router grouped expert route metadata by tensor owner/version.
- The sparse MLA PyTorch reference caches static positions, invalid masks, and optional sparse-gather indices by shape/device. This makes fixed-shape microbench timing less noisy and now permits a physical sparse-key score axis, but it is not a substitute for a real sparse MLA forward/backward kernel.
- `sparse-gather-manual-bwd` is a custom-autograd PyTorch surface for sparse MLA backward. It now avoids the sink-extended zero-gradient softmax temporary, computes sink gradients directly, and reduces aligned CSA/HCA compressed-key gradients without indexed scatter. It still saves gathered KV/probs and uses PyTorch tensor reductions for vanilla sliding-window key gradients; useful for a fused backward kernel contract, but still not a production sparse MLA kernel. `sparse-gather-manual-bwd-bshk` is a selectable layout variant that keeps sparse probabilities as `[B, S, H, K]`; it is near-neutral in PyTorch and mainly preserves a swappable fused-kernel contract experiment. `sparse-gather-manual-bwd-split-comp` avoids physically gathering CSA/HCA compressed KV for every query, computes backward softmax dot/score gradients partitioned by window/compressed keys, and runs compressed score/value/gradient contractions through explicit `torch.bmm`; it still materializes scores/probabilities and keeps the vanilla sliding-window gather in PyTorch. `sparse-gather-manual-bwd-split-comp-no-kv-cat` avoids materializing the compact `kv_all` concat and reports separate `d_kv`, `d_csa`, and `d_hca` edges. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-bf16-contract-window-tail-sum` is the current S2048 and S4096 best because it keeps scores/probs in the existing FP32 Triton softmax path while running sparse and compressed contractions in the BF16/FP16 input dtype for BF16/FP16 layers. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-window-tail-sum` remains the prior FP32-contract best; it writes score GEMM outputs directly into the preallocated sink-extended score slab, fuses forward scale/mask/sink softmax in-place with Triton, applies q/kv RoPE tails in-place with a Triton in-place kernel, replaces q-head RMSNorm with Triton forward/backward, reuses backward score-gradient temporaries in-place, accumulates compressed value products into the existing output/`dQ` buffers, accumulates the second selected-window KV-gradient product into the existing `grad_selected` buffer with `torch.baddbmm(..., out=...)`, replaces the vanilla `dKV` `index_add_` with a pad-light body/tail diagonal window sum, and fuses the backward softmax-dot/score-gradient update in a Triton row kernel. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-triton-rope-qnorm-sinkgrad-window-tail-sum` folds sink-gradient accumulation into the Triton row kernel with atomics; it is correctness-safe but slower on both active shapes, so keep it only as a sink-gradient kernel-contract probe. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-fast-rope-window-tail-sum` remains the prior PyTorch-custom fast-RoPE baseline. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-fwd-bwd-window-tail-sum` remains the prior no-fast-RoPE baseline. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-window-tail-sum` remains the best backward-softmax-only Triton baseline. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-window-tail-sum` remains the best pure-PyTorch softmax-backward baseline. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-window-sum` remains selectable as the padded diagonal-reducer baseline. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-vecdot-window-tail-sum` is correctness-safe but slower on S2048/S4096, so keep it only as a softmax-dot reduction negative control. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm` remains selectable as the pre-window-sum baseline. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-split-ch` is correctness-safe and avoids the compact CSA+HCA concat, but extra PyTorch branch launches regress S2048/S4096, so it remains only a negative-control variant. `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-baddbmm` remains selectable as the compressed-value-only baseline, and `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out` remains selectable as the direct-score baseline. `sparse-gather-manual-bwd-split-comp-bshk` is correctness-safe but slower on both S2048 and S4096, so keep it only as a selectable layout-contract probe. `sparse-gather-manual-bwd-split-comp-manual-softmax` avoids the concatenated softmax slab and is correctness-safe, but it regresses total timing because the manual PyTorch exp/reduction chain loses more forward time than it saves in backward.
- `sparse-gather-manual-bwd-split-comp-no-kv-cat-prealloc-bmm-out-comp-sel-baddbmm-triton-softmax-triton-window-sum` adds a Triton `grad_selected -> d_kv` reducer on top of the retained Triton softmax-score path. It is correctness-safe but slower than the retained path on S2048/S4096, so keep it only as a negative-control kernel-contract probe.
- RoPE cosine/sine tables are cached for fixed-shape reference runs and compressed CSA/HCA group-end position tensors are reused inside attention. This is also a harness/reference cleanup; real kernels should fuse or precompute RoPE according to their production dataflow.
- RMSNorm now has an explicit custom backward that saves the forward reciprocal RMS statistic. This is useful for future boundary-fusion experiments, but the current implementation is still PyTorch tensor math and should not be treated as a final optimized RMSNorm kernel. After the q-head RMSNorm attention update, `memory-light` is the best observed S2048 policy and `saved-accum` is the best observed S4096 policy; keep the policy explicit per shape until a fused backward/statistic-passing kernel changes the tradeoff. `boundary_impl=fused-attn-residual-moe-norm` is also correctness-safe, but it regresses both active retained timings, so it remains only a selectable contract probe.
- `grouped-assignments` is a PyTorch implementation that gathers selected expert weights. It is useful for ranking the MoE work, but a real grouped FP4 kernel should avoid the selected-weight materialization traffic.
- `grouped-contiguous` is a fixed-router canary for contiguous expert assignment layouts; it is a ceiling probe, not a general router implementation. In packed-sim mode it now returns sparse selected-row gradients for touched routed experts, so downstream optimizer support is not implied.
- `grouped-contiguous-fused` is an experimental negative-control path. It validates a fused autograd surface, but the paired `w1/w3` projection is slower than the best sparse-wgrad path for the current `B=1 S=16` shape.
- `grouped-expert-batches` is a fixed-router equal-count canary. It is the best current PyTorch ceiling path, but it still depends on balanced assignments and PyTorch `bmm`; it is not a production router or packed FP4 grouped GEMM implementation.
- `grouped-expert-batches` now gathers routed backward upstream gradients directly from sorted token ids instead of selecting from an expanded assignment-gradient view. This reduces unnecessary source traffic but remains a PyTorch packed-sim ceiling path.
- `grouped-expert-batches-index-add` is a negative-control variant. It validates direct token-id `index_add_` reduction for routed outputs and routed input gradients, but the adjacent S2048 sparse-gather baseline was faster, so the default remains `grouped-expert-batches`.
- `grouped-expert-batches-dense-wgrad` is the best general non-uniform PyTorch packed-sim MoE ceiling path at `B=1 S=2048` with manual sparse attention. It returns dense routed expert wgrads when every expert is touched, but it is still not a real packed FP4 grouped GEMM.
- `grouped-expert-batches-cyclic-uniform-dense-wgrad-baddbmm-xgrad` plus BF16-contract split-compressed sparse MLA is now the current fastest fixed-router PyTorch packed-sim family for both active shapes. With RMSNorm `saved-accum`, the S2048 best is `47.0589 ms`, `43.52k tokens/s`, `206.892852 TFLOPs`; S4096 is `68.2089 ms`, `60.05k tokens/s`, `310.415755 TFLOPs`. The current S2048 gap to the `43.3 ms` development target is `3.76 ms` (`1.09x`), while S4096 clears the `86.6 ms` target by `18.39 ms` (`0.79x target time`).
- These current best paths assume the benchmark's uniform cyclic fixed-router layout with no router gradient and are not learned-router training paths. The q-head RMSNorm Triton branch is a selectable attention-impl probe, not a replacement for the remaining high-leverage work: fused sparse MLA forward/backward and real packed-FP4 grouped MoE training GEMM.
- `grouped-expert-batches-cyclic-uniform-fused-w13-dense-wgrad` is a correctness-safe negative-control variant. It fuses W1/W3 into one packed-sim batched projection for the fixed cyclic router, but it regresses full-layer timing to S2048 `71.7978 ms` and S4096 `113.3559 ms`, mainly from slower backward. Keep the separate W1/W3 cyclic path with `baddbmm` routed xgrad as the retained ceiling.
- `grouped-expert-batches-cyclic-uniform-shared-x-dense-wgrad` is a correctness-safe negative-control variant that avoids physically expanding repeated fixed-router `x_group` before expert matmuls. Under the current attention/output-RoPE stack it regresses the full-layer timing to S2048 `57.3460 ms` and S4096 `100.6940 ms`, so keep the non-shared-x cyclic path as the retained PyTorch packed-sim ceiling.
- `grouped-expert-batches-cyclic-uniform-loop-k-dense-wgrad-baddbmm-xgrad` is a correctness-safe negative-control variant that avoids repeated fixed-router `x_group` and routed `y_unscaled` materialization by launching per-top-k grouped bmm slices. It regresses S2048 to `78.0783 ms` because the backward path fragments into too many smaller GEMM launches, so keep the larger batched expert GEMMs in the retained cyclic path.
- `grouped-expert-batches-cyclic-uniform-dense-w13-wgrad-baddbmm-xgrad` is a correctness-safe negative-control variant that fuses only W1/W3 routed wgrad into one larger grouped `bmm` while leaving forward and routed xgrad on the retained path. It regresses S2048 to `68.8743 ms`, mainly from a much slower backward, so keep separate W1/W3 wgrad `bmm`s in the retained cyclic path.
- `grouped-expert-batches-cyclic-uniform-topk-wgrad-baddbmm-xgrad` is a correctness-safe negative-control variant that groups the six fixed-router top-k experts per token block into taller per-group routed wgrad BMMs. It regresses S2048 to `60.4491 ms`; attribution shows routed MoE dgrad/wgrad around `33.1 ms` each, so keep the retained per-expert dense-wgrad cyclic path.
- `grouped-expert-batches-cyclic-uniform-dense-wgrad-bmm-out-baddbmm-xgrad` is a correctness-safe negative-control variant that writes cyclic MoE `bmm` outputs into explicit preallocated buffers. It is near-neutral but slower at both active shapes, so keep the simpler retained path.
- `grouped-expert-batches-cyclic-uniform-contig-wgrad-lhs-baddbmm-xgrad` is a correctness-safe negative-control layout variant that materializes the routed wgrad LHS transposes before `torch.bmm`. Isolated BMM probes suggested a small possible gain, but the adjacent full-layer S2048 timing was neutral/slightly slower (`47.5242 ms` vs retained adjacent `47.4972 ms`), so keep the retained strided-transpose path.
- `grouped-expert-batches-fused-w13` is also a negative-control path: it validates the fused W1/W3 surface but regresses backward time at `S=64`, `S=1024`, and `S=2048`.
- `grouped-expert-batches-fused-w13-dense-wgrad` combines the fused W1/W3 surface with dense routed expert wgrads. It is correctness-safe, but it remains a negative control because S2048 total time regresses to `81.1608 ms` versus `66.8982 ms` for the adjacent dense-wgrad baseline.
- `shared-expert-impl=fused-w13` is a negative-control path: it validates an independently swappable shared expert W1/W3/SwiGLU fused surface, but the full-layer S2048 timing regresses slightly despite a small forward shared-expert block improvement.
- AITER FlyDSL FP4x2/MXFP4 routed MoE is now proven shape-viable in forward-only synthetic-packed mode at DSv4 dimensions, but it is not yet numerically integrated with the 128x128 FP4 training path and provides no backward/wgrad coverage in this harness.
- AITER Triton BF16 `gmm`/`ptgmm`/`nptgmm` wrappers are correctness-clean for the cyclic-uniform routed MoE BMM shapes, but slower than the existing `torch.bmm`/hipBLASLt path on S2048 and S4096, so they are a negative-control integration path.
- `profile_microbench.py` has `--ready-file`, `--trigger-file`, and `--sleep-after-iters-s` controls for attach-style profiling attempts. On this ROCm image, rocprofv3 attach reported success but did not emit trace files; use `--roctx-profile-region` with rocprofv3 `--selected-regions` for clean timed-loop kernel traces.
- Fixed-router mode intentionally bypasses router weight gradients so optimized MoE kernels can be compared without routing noise.
- Backward attribution now exists at the major-block level plus first attention, grouped expert-batch routed MoE, cyclic uniform grouped expert-batch routed MoE, and shared expert sub-block splits. Profiler-side kernel grouping still needs deeper decomposition.
- HF one-layer validation, real packed FP4 grouped GEMM, and FP4 activation/gradient scale policy are still pending.
