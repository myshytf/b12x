# Shared input-rotation scratch for coupled QSRT experts

Status: implemented; qualified by planner checks and native GPU microbenchmarks.
Full-model serving qualification is a separate deployment gate.

The full-rotation W4A16 MoE executes input rotation, FC1, activation and FC2 in
one persistent kernel. Global phase barriers order these operations. With
coupled Hadamard rotation and a single shared input-scale row, the token-major
rotation writes `tokens * hidden` FP16 elements. FC1 consumes these elements
and writes only `tokens * top_k * fc1_columns` elements at the start of the
FC1/FC2 intermediate buffer. The remaining tail becomes live only when FC2
writes its routed outputs.

`TPMoEScratchCaps.w4a16_shared_input_rotation=True` declares that the weight
package has shared input scales. The planner places the token-major operand
in that unused tail when it fits, coupled rotation is present, token-major
rotation is enabled, and activation calibration is disabled. Binding checks
the actual scale tables and rejects an incompatible package or a disabled
token-major mode. Other inputs retain separately allocated rotation storage.
No weight format, arithmetic operation, rounding, reduction order, route
geometry, or kernel synchronization changes.

For Kimi-K3 at 4,608 tokens, top-k 16, hidden width 3,584 and local intermediate
width 256 or 384, the arena reserves 528,482,304 fewer bytes per rank (504 MiB).
This does not reduce the KV-cache budget or the configured token capacity.

Validation:

- Nine planner/binding cases cover capacity pricing, disjoint FC1 writes,
  scale metadata, calibration and insufficient-tail fallbacks, and environment
  drift. The scratch suite has 47 passing cases. A separate packed-W4A16
  global-route-map case fails identically on the parent revision: a requested
  12-expert route map is validated against eight local experts. The QSRT path
  does not use that failing contract.
- `benchmarks/benchmark_qsrt_rotation_reuse.py` exercises native K2 decoding
  with hidden width 3,584 and 16 experts, equal eager and CUDA-graph outputs,
  NaN-poisoned scratch, immutable weights, and unchanged replay allocation.
  Intermediate width 256 covers active token counts 1, 7, 8, 9, 47, 48, 49,
  and 82. Decode uses route block eight; prefill uses route block 48.
- On an RTX PRO 6000 Blackwell Max-Q, interleaved width-384, 82-token graph
  samples measure 163.906 microseconds with separate rotation storage and
  163.779 microseconds with shared storage. Width 256 measures 97.181 versus
  97.171 microseconds. These establish no material kernel-time regression;
  they do not establish a complete-model throughput gain.

Numerical precision and output quality must be preserved or improved before
serving a change. Memory savings alone do not waive that acceptance condition.
