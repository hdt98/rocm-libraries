# Reference Sources

Comparison-only source anchors for DeepSeek V4 training stacks.

- `miles-deepseek-v4-pr1045/`: vendored snapshot of `yueming-yuan/miles:deepseek-v4` at `032721cd61bf7164955f084425eb9f315352fd26`, the head of `radixark/miles#1045`.
- `megatron-lm-deepseek-v4-pr28/`: vendored snapshot of `yueming-yuan/Megatron-LM:deepseek-v4` at `1c6e5b7bcde0097ebe193b6258115ff3558f69d6`, the head of `radixark/Megatron-LM#28`.
- `ascend-cann-recipes-train/`: sparse snapshot of `cann/cann-recipes-train` at `55315e2ee4e6b1f21ff6b040631075bbb84628fb`, containing the TorchTitan-NPU DeepSeek-V4 deployment recipe and CANN AutoFuse performance report.
- `ascend-cann-dsv4-image-cann900-v30/`: source-only snapshot extracted from the recipe image `dsv4_train_torchtitan:cann9.0.0_v3.0`, including the DSv4 custom Ascend C `SparseAttnSharedkvGrad` and `SparseLightningIndexerGradKLLoss` implementation sources.
- `torchtitan-npu-v0.2.2-dev/`: vendored snapshot of `cann/torchtitan-npu:v0.2.2-dev` at `a239d25ba71a6c2ab99e2ed57a9b2de7ac8ebdf5`, containing the Ascend TorchTitan-NPU sparse attention and LightningIndexer operator wrappers used to map `SparseAttnSharedkvGrad` and `SparseLightningIndexerGradKLLoss` to the AMD WIP.
- `ascend-mindspeed-llm-upstream/`: shallow snapshot of `ascend/MindSpeed-LLM` at `df924c58496196341c4578fe95be9641ce0f496c`, used to inspect DeepSeek-style router load balancing, group-limited routing, expert-bias updates, and communication auxiliary losses.
- `ascend-mindspeed-upstream-gitee-lite/`: shallow snapshot of `ascend/MindSpeed` at `bb583c908bc35feddee2ac94a5704cb313471859`, used to inspect MoE all-to-all overlap and `balanced_moe` hot-expert replication / gradient-reduce mechanisms.
- `mori/`: vendored snapshot of `ROCm/mori:main` at `5dd02d6815e9113381a72f6de6b22034fc09b5a3`, used as the AMD EP dispatch/combine and split send/recv overlap reference.
- `hipkittens/`: shallow snapshot of `HazyResearch/HipKittens:main` at `cd090ae98ee4e7b8d3d5291fc62cfd716aecb946`, used to inspect AMD MI350 attention-backward, MFMA, swizzled memory, scheduling, and packed BF16 atomic primitives for possible sparse SharedKV backward reuse.
- `day0-rl-notes.md`: compact notes from the LMSYS Day-0 RL section and radixark Miles roadmap issue.

These trees are source references only. AMD changes should be made under `sources/wip/` or captured as patch files under `changesets/`.
