# ablx

`ablx` is a research scaffold for deterministic weight-space upscaling of sparse MoE
LLM checkpoints. V1 focuses on Qwen3.6-style fused MoE tensors and produces
function-preserving or near-function-preserving expanded checkpoints that can be
verified before slime/Megatron/SGLang post-training.

## Commands

```bash
ablx inspect --model /path/to/checkpoint
ablx upscale --source /path/to/checkpoint --recipe examples/qwen36_expand_768.yaml --out /path/to/out
ablx probe --source /path/to/checkpoint --candidate /path/to/out --prompts prompts.jsonl
ablx slime-plan --candidate /path/to/out --out /path/to/slime-plan
```

Remote Hugging Face model IDs are supported when `huggingface_hub` is installed.
Large-model logit probing requires optional `torch` and `transformers`.

## V1 Upscaling Strategy

- Keep hidden size, tokenizer, attention, layer count, expert count, top-k, MTP,
  RoPE, and vocab unchanged.
- Expand MoE intermediate width, defaulting from `512` to `768`.
- Preserve old fused expert gate/up/down slices exactly.
- Add deterministic small-noise capacity to new gate/up rows and zero the new
  down columns so the new neurons start silent.
- Emit a manifest describing every changed tensor, expected trainable slices, and
  post-upscale verification gates.

`clone_experts` is present as an experimental gated transform for dry-runs of
true total-parameter expansion. It is not the default first training path.
