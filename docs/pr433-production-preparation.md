# Common Trellis preparation for the Kimi serving branch

Status: **qualified for GPU preparation and component output equality**;
serving-model checks are pending.
The reference is B12X `0439b80e` and vLLM `08ac2c8732`. This backport takes
the uniform preparation, descriptors and EXL3 adapter from upstream B12X
PR433, source `437cead4d902dc549d5c9744630090fc142d9847`, and adds a QSRT
atoms-v2 adapter for the deployed Kimi checkpoint.

Checkpoint adapters run on CPU and return `TrellisSource` and `TrellisWeights`.
The source declares bitrate, transforms and source-global intermediate
coordinates. The weight bundle carries codewords, FP16 scale vectors and
per-expert signs. Common preparation validates that contract, assembles the
codewords in bounded expert batches, applies the existing FP16 scale boundaries
and generates signs in global coordinates before slicing the rank extent.
It transfers final native weights to the selected CUDA device before planning
or graph capture. It does not decode and requantize the checkpoint.

QSRT interleaves scale bytes between expert codeword bundles. The common
codeword tensor therefore also accepts an explicit expert axis with a physical
stride between bundles. The adapter returns a view of the original CPU extent;
it does not allocate another extent-sized payload. Ordinary two-dimensional
native/EXL3 slot rows use the same assembly function. Staging bounds apply to
the logical codeword bytes, excluding the interleaved scales.

The public input API is `plan_weights(source=..., activation=ActivationSpec(...),
geometry=MoEGeometry(...))` followed by `prepare_weights(weights=..., device=...,
staging=TrellisStaging(...))`. Public I/O and rotation arithmetic are separate:
the deployed contract is BF16 I/O with explicitly requested FP16 rotations.
A BF16 rotation request is rejected because the preserved execution backend
does not implement that arithmetic contract; it is never silently narrowed.

This is a preparation backport into the served execution API. Returned plans
and expert owners retain the branch's `MoEWeightPreparationPlan` and
`B12XFP4ExpertWeights` contracts. The upstream prepared-launch API, new query
schema and vLLM EXL3 auto-detection are not substituted for the existing runtime.
The production vLLM QSRT loader now uses the common preparation API for its
uniform K2 TP9 W4A16 path. Other existing loader profiles retain their existing
paths. Projection-tiered MCG is unsupported through the new common frontend;
the legacy entry points remain available. Uniform native and EXL3 callers can
use the shared frontend without depending on QSRT container fields.

The full-codeword decoder from the PR436 backport, bulk staging inside each
kernel, stable route packing, FC2 grouping, reduction order, launch geometry,
scratch ownership, and target/draft cache formats remain the serving reference's.
No CUDA kernel or intrinsic implementation changes in this preparation delta.

Qualification compares all prepared tensor bytes, gate/up allocation sharing,
resident and peak CUDA allocation, eager/graph outputs for real TP9 checkpoint
extents, decode/boundary/prefill shapes, and model token log probabilities.
The local package and raw receipts are at
`/home/g0san/kimi-k3-production/candidates/k3-pr433-trellis-20261001/`.

GPU qualification passed four synthetic tests using the full 896-expert,
3584-channel contract and four source extents, including both FC1 halves.
Real checkpoint layer-1 TP9 extents (384/256 channels) have identical prepared
tensor hashes and shared gate/up storage. All 46 decode, boundary and prefill
cases (184 initial/mutated output hashes) match, including CUDA Graph replay
at prefill row counts 256 and 1536 with 48-row route blocks. Compiled decode
specializations and resources are identical. Resident CUDA allocation does
not increase; peak preparation allocation increases by 21,964,800 bytes for
the 384-channel extent and 14,638,080 bytes for the 256-channel extent.

Raw source-pinned receipts are tracked in
`benchmarks/qualification/pr433_common_trellis_20261001.json`. The preparation
time samples are unbalanced separate-process measurements and do not establish
a speedup. Five stale BTX fixture failures in the broader host suite reproduce
on the untouched reference; 205 targeted common/host tests and 27 vLLM tests
pass. The vLLM file's pre-commit hooks pass.

Scratch sharing uses an execution compatibility key. Full preparation-plan
equality still includes checkpoint-global source coordinates, but prepared
layers with equal local geometry, codebook, transforms and precision may share
an arena. The GPU regression binds weights from one extent to an arena planned
for another extent and compares its output with the preserved preparation.
Codebook and I/O dtype mismatches remain incompatible.
