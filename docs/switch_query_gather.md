# Query gathering across a five/four PCIe switch split

Status: implemented and qualified for the recorded native tests; full-model
serving qualification is pending. The option is disabled by default.

`B12X_PCIE_DCP_GATHER_GROUPS="0,1,2,3,8;4,5,6,7"` describes logical ranks sharing
each switch on a cascaded PEX88096 fabric. The group order follows the
application's CUDA rank order, not physical GPU numbering. The two groups must
partition nine ranks into five and four, and the DCP transport must be `push`.
The setting applies only to head/query gathering; LSE reduction arithmetic and
paired-projection gathering retain their existing implementation.

Each 16-byte source packet crosses the inter-switch link once to a relay.
The relay distributes it inside the receiving switch. A small fraction of
five-rank local traffic returns through a relay to balance GPU transmit load.
Relay selection uses upper packet-index bits; the returned packet selection
uses lower bits, so the return traffic is distributed among relays.

For a source contribution of B bytes on every rank, all-peer push sends 20B in
each direction across the switch boundary. The two-stage schedule sends 5B
and 5.25B. Every GPU still receives exactly 8B and maximum GPU transmission
is 8.0625B, close to the eight-copy port lower bound. Reducing traffic on the
shared link does not eliminate each GPU's own port or memory-copy limits.

Both stages preserve the original row-to-block/warp assignment. Paired block
barriers publish the direct copies and then the relay copies before the
original copy-out phase. Existing double-buffered staging and device graph
epochs remain in use. There is no additional tensor allocation, quantization,
floating-point computation, or reduction reordering.

Validation is recorded in `evidence/kimi_switch_query_gather_tp9.json`:
95 CPU cases pass, including packet-delivery and byte-volume checks. Native
TP9 tests gather arbitrary BF16 bit patterns exactly at batches 1, 2, 4, 8 and
16, verify eager execution and CUDA graph replay with changed contents, and
perform 200 mixed query-gather/LSE-reduction graph executions. At the Kimi
96-head model's padded 99-head exchange and 576-element BF16 query, batch-four
interleaved graph medians are 51.103 microseconds for flat push and 36.998 for
the relay schedule (relay/reference 0.724). This is a query-kernel measurement,
not an end-to-end model speedup.

The serving host has a separate history of GPU0 PCIe link loss. Its memory
clock offset is under controlled testing at zero. Native correctness does not
qualify host or driver stability; full-model measurements must use the same
hardware policy and reject concurrent-request contamination.
