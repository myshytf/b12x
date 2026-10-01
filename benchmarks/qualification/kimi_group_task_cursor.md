# Carried FC2 task-column cursor

Status: **implemented; qualified; serving**.

The selected inline FC2 path restarts a prefix search through 28 immutable column counts for every grid-strided task. The candidate carries the column index, local task and count forward. Each CTA still executes exactly `cta + iteration * grid_x`; every descriptor, K interval, partial merge and floating-point operation is retained. The invariant is `task == sum(counts[:column]) + local_task`. Empty columns advance normally and an empty queue skips the search.

`B12X_W4A16_GROUP_TASK_CURSOR=1` enables the FC2-only option. It is default off and included in the compile key. No new buffer, launch, shared-memory reservation or synchronization is added. Source is `/home/g0san/worktrees/b12x-group-task-cursor-20260915`, revision `81c899d`. The selected inline package is the reference and restore target.

## Native evidence

The target is nine RTX PRO 6000 Blackwell SM120 GPUs using PyTorch 2.13.0, CUDA 13.3 and CUTLASS DSL 4.6.2. Native checkpoint probes use layer 1 of `/mnt/models/Kimi-K3-QSRT-K2`, FP16 MoE operands and scratch, FP32 accumulation, and BF16 outputs. Both physical devices retain production power/clock limits: GPU 5 UUID `GPU-28828840-7cc9-ffe0-787a-99b704f6ff51`, GPU 6 UUID `GPU-92f7a0e8-3a39-9a02-0605-131d9d0cde5b`.

The reference jobs mount the actual deployed kernel file over the experiment tree, so their timings do not inherit candidate changes. Both arms use the same benchmark and checkpoint, seeds, shapes and native launch path.

All 54 configurations and 270 snapshots match exactly, including all-invalid routes. Retained rotation, FC1, activation, FC2 and final outputs match in the representative stage probe. GPU 6 A/B/B/A repeats exact outputs. The 28-object census verifies raw identities and reports unchanged 256-thread/101376-byte launches, 115–125 GPRs and zero stack/frame/local accesses. The migration-specific typed-OptLevel normalization limitation is retained without weakening raw integrity checks.

The table uses physical GPU 6, M4, two pass medians per arm from A/B/B/A; positive time change is slower. These are native component measurements, not whole-model throughput.

| Local width | Sharing | Selected inline µs | Cursor µs | Time change |
|---:|---|---:|---:|---:|
| 384 | independent | 136.984 | 133.056 | -2.867% |
| 384 | shared | 106.112 | 104.984 | -1.063% |
| 384 | partial | 119.944 | 116.824 | -2.601% |
| 256 | independent | 104.152 | 100.440 | -3.564% |
| 256 | shared | 79.192 | 79.816 | +0.788% |
| 256 | partial | 93.456 | 90.944 | -2.688% |

The fully shared width-256 case regresses about 0.8% on GPU 6, despite gains in the other five cases. It remains in the evidence. The actual model observation contains many more remaining tasks than this extreme shared fixture; selection uses the separate complete-model measurements below.

The source-level lookup model on 23,552 actual MoE observations predicts a reduction from about 23,154 to 4,993 column reads per MoE (78.4%). These are logical source reads, not measured memory transactions or a throughput claim.

The first native attempt exposed a CuTe loop-state type-join error before computation. Explicit Int32 initialization fixes it. The retry passes all native gates. Raw commands, source manifests, stage tensors, output hashes, timings, GPU snapshots and exact resource reports are in `gpu-r2/`. `model-r1` compares against the selected inline package using matched top-5-logprob requests and nine-worker dispatch evidence.

## Complete-model qualification

`model-r1` compares the actual selected inline kernel with cursor revision `81c899d4a2c6e43fd520d7f01df238c676598eb8`, SHA-256 `78078a22002d029942c6a26d23662a3ace689466d7b5ce0924e2b067a0da3ebe`. The existing TP9/DCP9 production image, 30-file communication overlay, power/clock limits, model/checkpoint formats, draft proposals, L2 residency and request contract remain the same. The precision specification is in `/home/g0san/kimi-k3-production/ACTIVE-PRODUCTION.md`.

The unprofiled A/B/B/A comparison has four samples per arm and length: two A-before, two B-first, two B-second and two A-after samples. Requests use a fixed prompt, seed 20260912, temperature zero, top-5 logprobs, returned token IDs and a fresh cache salt. Decode throughput excludes the first delivered token. All 16 requests have exactly equal token IDs, chosen/top-5 logprobs, draft steps, accepted draft tokens and prompt hashes.

| Input / output tokens | Selected inline tok/s | Cursor tok/s | Cursor / reference | Change |
|---|---:|---:|---:|---:|
| 8192 / 1024 | 119.715250 | 120.982084 | 1.01058206 | +1.0582% |
| 65536 / 512 | 92.306555 | 93.008210 | 1.00760135 | +0.7601% |

All six examined 8K delivery windows improve, from +0.932% in tokens 1–67 to +1.121% in tokens 770–1024. Their chunk boundaries match across all eight runs. These short windows include transport jitter and are diagnostic; the complete-generation medians determine selection. The earlier inline-grouping comparison's short-input regression remains disclosed in its own report. Ratios from different campaigns are not multiplied into an unmeasured cumulative speed claim.

Nine-worker startup diagnostics identify cursor-enabled native M4 launches at both local widths. Separate target traces retain 92 MoE kernels, 92 regular header fills, no standalone group builder and 186-by-256 launch geometry. Profile duration is not used as throughput evidence. The reference before/after bracket and original gateway/mode were restored before activation.

Raw artifacts are `model-r1/comparison.json`, `model-gates.json`, `stream-window-comparison.json`, per-phase responses and traces. `deployment-decision.json` pins the native, resource and model evidence hashes. Performance qualification covers the two documented fixed-prompt request lengths, not all possible prompts or the configured one-million-token context limit.

## Review and runtime composition

This branch is stacked on `perf/kimi-inline-groups-20260915` (personal fork PR #11). Its kernel and native benchmark bytes match the qualified research source. The complete serving runtime retains the selected communication overlay identified by `/home/g0san/kimi-k3-production/candidates/k3-static-peers-20260914/source-manifest.json`; four communication files differ from or are absent in the review base. The MoE source port does not establish complete-checkout runtime identity.

The raw experiment root is `/home/g0san/kimi-k3-production/research/decode-optimization-20260914/group-task-cursor-20260915`. The native launcher uses `run_window.py --job-plan jobs-r2.json`; every job records its complete command and environment in `gpu-r2/`. The reference jobs mount the actual deployed inline kernel over the experiment tree. `model_screen.py --out model-r1` conducts the qualified model bracket. GPU probes require a drained owner and an exclusive finite measurement window.

The review checkout passes 17 CPU policy/interval tests and Ruff. These are source/host checks, separate from native qualification. No new dedicated sanitizer run was conducted for the cursor change; the preceding inline implementation retains its targeted audit.

## Production activation

`activation-r1/activation.json` records successful activation at `2026-09-14T20:29:41.531903+00:00`. Both startup requests reproduce all tokens, chosen/top-5 logprobs and draft acceptance. The old-source cache bridge restores 13,824 tokens exactly. Cold/continuation/external-restore cache paths, four active requests and a 4096-by-4096 image pass; vision content/logprobs match the prior source. Authenticated gateway streaming returns the expected canary and terminal SSE event. The selected package is `/home/g0san/kimi-k3-production/candidates/k3-group-task-cursor-20260915`, with the inline package as immediate rollback. The retained namespace is `iso-k3-stable-routes-20260914`; its fingerprint is not added to the global registry.
