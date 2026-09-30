# Graph conditional

This example builds a CUDA Graph whose IF conditional node is decided by a
kernel. `select` reads a flag from device memory and passes it to
`cuda_device::graph::set_conditional`; the node then runs or skips `body`,
which writes 42.

```text
select(handle, flag) ──▶ IF(handle) { body(out) }
```

The same instantiated graph is launched with the flag set to 1, 0 and 7, so
the branch is taken on the device, not by rebuilding the graph.

`set_conditional` calls the CUDA device runtime API `cudaGraphSetConditional`.
The CUDA driver resolves the call when it loads the module, so the example
builds on the ordinary PTX route with no extra link input.

Run it with CUDA 12.3 or newer:

```bash
cargo oxide run graph_conditional
```
