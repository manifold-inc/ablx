# Qwen3.6 Tensor Map

`ablx` classifies tensors by name before any transform:

- `*.mlp.experts.<id>.*gate_up_proj*`: routed expert fused gate/up projection.
- `*.mlp.experts.<id>.*gate_proj*`: routed expert gate projection.
- `*.mlp.experts.<id>.*up_proj*`: routed expert up projection.
- `*.mlp.experts.<id>.*down_proj*`: routed expert down projection.
- `*shared_expert*`: shared expert tensors, widened with the same intermediate
  expansion as routed experts.
- `*router*` or sparse MoE gate tensors: router weights, repeat-expanded during
  expert fission.
- `*vision*` / `*visual*`: passthrough tensors; copied unchanged.
- MTP tensors are transformed when `include_mtp: true`.

The transform report records every tensor role plus old and new slices. The
upsampler uses that report as the source of truth for where new capacity lives.
