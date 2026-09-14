# Inline FC2 reference-group preparation

Status: **implemented, research-only; native and model correctness qualified; mixed model performance, unselected**.

The Kimi TP9 decode path can reuse FC2 weight tiles across routes without changing their numerical reduction contract. Routes share a tile only when their expert, output tile and complete ordered K intervals agree. Partial results retain the deployed descending-K merge order, FP16 operands/stores, FP32 accumulation and BF16 top-k output.

`B12X_W4A16_REFERENCE_GROUPED=1`, `B12X_W4A16_REFERENCE_GROUPED_PHASES=fc2` and `B12X_W4A16_GROUP_BUILDER_INLINE=1` enable the candidate. The inline path requires M2–M8, latent width 3584, local intermediate width 256 or 384, 896 K2 experts, top-16, direct expert-map routing, coupled rotation, small-M stripe split-K, 256-thread CTAs, direct LUT staging and the qualified 186-CTA execution budget. On the measured 188-SM devices the existing two-SM reserve is retained. Defaults remain off; other configurations keep the existing standalone grouping or ordinary dispatch as applicable.

The 28 FC2 metadata CTAs reuse 1536 bytes at the start of ordinary GEMM shared scratch while the disjoint 64 KiB LUT copy is in flight. Builder CTAs begin at 7 for token-major rotation or 112 for route-major rotation. All 256 threads participate in the CTA barriers, but only the first 128 own route records. The regular graph header fill and existing post-rotation grid barrier remain. The candidate adds no shared-memory reservation or graph metadata-builder node.

The arena planner exposes the already deployed shared-input-rotation contract and reserves grouping metadata/partials before graph capture. Graph replay reuses the same arena and updates integer descriptors from current route IDs. In-kernel construction never resets the resident-grid barrier header.

## Native evidence

The qualification environment uses nine RTX PRO 6000 Blackwell devices, TP9/DCP9, PyTorch 2.13.0, CUDA 13.3, CUTLASS DSL 4.6.2 and native checkpoint layer 1 from `/mnt/models/Kimi-K3-QSRT-K2`. Tests preserve the production power/clock configuration. Physical GPU 5 UUID is `GPU-28828840-7cc9-ffe0-787a-99b704f6ff51`; physical GPU 6 is `GPU-92f7a0e8-3a39-9a02-0605-131d9d0cde5b`.

- Metadata/LUT fixture: 56 passing GPU cases against an independent CPU interval oracle; 28 inline-FC1 cases skipped as unsupported. Cases cover M2–M8, both widths, duplicate/invalid/mutated routes and maps, poisoned workspace, graph replay, unchanged header and exact LUT bytes.
- Native checkpoint: 54 configurations with four snapshots each, all 216 outputs exact against ungrouped execution. M1 and M16 exercise fallbacks. Retained rotation, FC1, activation, FC2 and final outputs are byte-identical for representative token-major and route-major execution.
- Native traces identify the actual chosen compile object, one integer fill, no standalone builder and 256-thread launches.
- Exact-object census: 28 MoE objects with verified raw manifests and object hashes, 101376-byte compiler-bound dynamic shared-memory launches, zero stack/frame/local loads/local stores and 115–125 allocated GPRs. Native ordered resource records match per arm.
- Targeted Compute Sanitizer: six native M4 configurations, 24 exact snapshots and zero memory errors. `--external-sanitizer` disables the competing Torch CUPTI subscriber and emits no timings. The filter is `kns=W4A16FusedMoeKernel`; unchanged Triton planner kernels are outside this targeted memory audit.
- Review checkout: 17 CPU policy/interval tests and Ruff pass. CPU policy tests construct the actual kernel with device metadata; they are not GPU arithmetic evidence.

Raw records are retained at `/home/g0san/kimi-k3-production/research/decode-optimization-20260914/inline-groups-20260915/`. `gpu-r4/source.json` binds the native source; `model-r1/native-preflight/gate.json` binds the targeted memory check. `jobs-r4.json`, per-job command files, output hashes, timings, GPU snapshots and traces reproduce the component comparisons. Native benchmark modes are 0 (ungrouped), 1 (separate FC2 builder) and 2 (inline FC2 builder).

## Component timing and scope

Physical GPU 6 uses M4, both local widths, 60 unprofiled graph samples per configuration in A/B/B/A order. Values below are medians of the two pass medians per arm. Positive time change is slower; these are component diagnostics, not model throughput.

| Local width | Route sharing | Reference µs | Inline µs | Time change |
|---:|---|---:|---:|---:|
| 384 | independent | 125.648 | 137.200 | +9.19% |
| 384 | shared | 126.136 | 106.424 | -15.63% |
| 384 | partial | 125.608 | 118.872 | -5.36% |
| 256 | independent | 96.256 | 104.416 | +8.48% |
| 256 | shared | 95.608 | 79.760 | -16.58% |
| 256 | partial | 96.256 | 93.720 | -2.63% |

The independent-route regression is material and is retained in the decision evidence. Full-model A/B/B/A and exact token/logprob/acceptance checks are required before serving selection.

This review branch contains the exact tested MoE and compiler source, but is not a complete serving checkout. The model experiment composes the three changed MoE files over the selected production package, including its unchanged 30-file communication overlay. That overlay's source manifest is `/home/g0san/kimi-k3-production/candidates/k3-static-peers-20260914/source-manifest.json`; four communication files differ from or are absent in the review base. `review-source-comparison.json` records the distinction. Do not infer full-runtime identity from the MoE port.

The legacy compiler-migration comparison helper rejects typed `OptLevel` option objects. The raw resource census retains that limitation and validates the complete raw identities and launch evidence separately; it neither rewrites cache identities nor claims compiler-migration qualification.

## Completed model comparison

The unconditional candidate is unselected. Both request contracts use four unprofiled samples per arm and input length. All 32 timed requests preserve tokens and draft acceptance; all 16 logged requests preserve chosen/top-5 logprobs. Six matched-position profile requests also preserve their complete outputs.

| Request contract | Input | Reference tok/s | Inline tok/s | Change |
|---|---:|---:|---:|---:|
| Top-5 logprobs | 8192 | 120.256738 | 119.781198 | -0.3954% |
| Top-5 logprobs | 65536 | 91.182381 | 92.260384 | +1.1822% |
| No logprobs; token IDs | 8192 | 121.017668 | 120.476682 | -0.4470% |
| No logprobs; token IDs | 65536 | 91.758753 | 92.825972 | +1.1631% |

The original selection captures started early and omitted requested logprobs. New captures match the request contract and trigger at generated-token positions 0 and 768 (actual late trigger 770/774 in both arms). In the late four-step captures, target spans increase 252–261 µs and routed-MoE category sums increase 225–261 µs. The early capture improves. Category sums overlap and are not throughput metrics.

Unprofiled delivery windows agree: initial 66-token windows improve about 0.63–0.65%, while final 254-token windows regress about 0.56–0.70%. All eight runs within each request contract have identical token chunk boundaries. Do not select a context-length heuristic from this prompt pair; collect actual grouping and cache evidence before an adaptive exact FC2 policy.

The selected production package and original gateway/mode are restored and authenticated streaming generation passes. The candidate has no activation/cache/concurrency/vision qualification because it is unselected. Complete raw evidence and the decision are under the documented experiment root in `REPORT.md`, `qualification.json`, `deployment-decision.json`, `model-r1/`, `contracts-r1/` and `post-model/`.
