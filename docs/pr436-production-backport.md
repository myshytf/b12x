# Full-codeword decoder backport for the Kimi serving branch

Status: **implemented**. GPU qualification is recorded in the deployment
package at `/home/g0san/kimi-k3-production/candidates/k3-pr436-lut-20261001/`.

The serving B12X reference is
`a3d58239` (`feat/k5-dcp-geometry-20260927`). It already stages a 64 KiB
precomposed SQG-XOR-Cheb-T12 table for eligible uniform-rate MoE kernels,
with bulk asynchronous staging and retained device tables from the earlier
direct-table implementation. `B12X_SQG_XOR_CHEB_T12_DIRECT_SMEM=1` selects
this path in production.

[B12X PR 436](https://github.com/local-inference-lab/b12x/pull/436), head
`cfb27fbf2d42426e019c2d8ec4ea28cf4ee53d75`, implements a related table on
the newer common-Trellis preparation API. This backport applies the remaining
instruction changes to the existing shared and global decoder entry points:
six `prmt.b32` instructions pack the eight lookup bytes, and the shared
decoder reuses each window register for its byte address. The codeword
mapping, byte order, E4M3-to-FP16/BF16 conversions, GEMM operations, route
order, shared-memory capacity and launch geometry remain the reference's.

The preparation API and `B12X_TRELLIS_DECODE_TABLE` control from the upstream
PR are not added to this legacy branch. Its existing residency checks,
large-prefill fallback and capture-time table retention remain authoritative.
Upstream compact-versus-full throughput results do not measure this change:
the deployed reference already uses the full table.

`tests/moe/test_trellis_direct_lut_decode.py` compares shared and global
direct lookups against the compact decoder at K2/K3/K4, including all 65,536
codewords, random crossing windows, poisoned outputs and graph replay with
changed inputs and stable tensor addresses. Whole-MoE qualification must
also compare actual checkpoint extents against the preserved serving tree.

The shared-table probe is adapted from `6af4b96706eb582ef1098032a16e77de52b2d46d`;
the exhaustive codeword coverage and byte-pack sequence follow PR 436.
