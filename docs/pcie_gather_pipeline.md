# Lossless PCIe gather materialization overlap

Status: research-only until native multi-GPU qualification completes.
`B12X_PCIE_DMA_PIPELINED_GATHER=1` enables the schedule at ring construction;
the default is disabled. All ranks in a channel must use the same setting.

The uncompressed all-reduce ring retains each received all-gather payload in a
per-step scratch region until the closing neighbor handshake. The overlapped
schedule forwards that immutable payload directly to the next rank while a
separate copy stream writes the local output. Reception events gate both
readers. The main stream joins the forwarding, flag-publication and output-copy
streams before publishing the closing handshake, so a following collective
cannot overwrite data still in use.

Paired all-gather uses the same lifetime: forwarded rank blocks stay in the
receive scratch while the separate stream scatters their two components into
the output tensors. Reduction arithmetic, payload bytes, FP32/BF16 rounding,
rank order, and monotonic flag counters are unchanged. Compressed all-reduce
modes retain their existing implementation and are not part of this optimization.

The CPU schedule model checks overlapping reads and writes across streams,
events and device flags. Its repeated 9-rank all-reduce, paired-gather and mixed
all-reduce/gather/reduce-scatter cases report exact reference values and no
unordered conflicting accesses. The communication unit suite has 76 passing
cases. Native timing and CUDA-graph qualification use
`benchmarks/benchmark_pcie_dma_pipeline.py`, with all GPUs idle and at least
2 GiB free per rank after context initialization. A model occupying almost all
device memory leaves insufficient headroom for a second 9-rank IPC runtime.

Speed alone is not an acceptance criterion. Outputs must match the reference
bit for bit, including graph replay with changed input contents and in-place
all-reduce. The deployment must preserve or improve model quality and precision.
