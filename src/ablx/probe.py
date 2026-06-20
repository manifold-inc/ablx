from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .inspection import inspect_model
from .models import ProbeReport


def probe_checkpoints(source: str | Path, candidate: str | Path, prompts: str | Path | None = None, max_prompts: int = 16) -> ProbeReport:
    warnings: List[str] = []
    metadata_metrics = metadata_probe(source, candidate)

    if prompts is None:
        return ProbeReport(
            source=str(source),
            candidate=str(candidate),
            runtime="metadata-only",
            prompt_count=0,
            metrics=metadata_metrics,
            warnings=["no prompts supplied; logit probe skipped"],
        )

    try:
        return torch_logit_probe(source, candidate, prompts, max_prompts=max_prompts, metadata_metrics=metadata_metrics)
    except ImportError as exc:
        warnings.append(f"torch/transformers unavailable; logit probe skipped: {exc}")
    except Exception as exc:  # noqa: BLE001 - probe should produce a report, not crash large runs.
        warnings.append(f"logit probe failed; metadata probe completed: {exc}")

    prompt_count = len(load_prompts(prompts, max_prompts=max_prompts))
    return ProbeReport(
        source=str(source),
        candidate=str(candidate),
        runtime="metadata-only",
        prompt_count=prompt_count,
        metrics=metadata_metrics,
        warnings=warnings,
    )


def metadata_probe(source: str | Path, candidate: str | Path) -> Dict[str, object]:
    src = inspect_model(source)
    cand = inspect_model(candidate)
    src_tensors = {tensor.name: tensor for tensor in src.tensors}
    cand_tensors = {tensor.name: tensor for tensor in cand.tensors}
    common = sorted(set(src_tensors) & set(cand_tensors))
    changed_shape = [
        {
            "name": name,
            "source_shape": src_tensors[name].shape,
            "candidate_shape": cand_tensors[name].shape,
        }
        for name in common
        if src_tensors[name].shape != cand_tensors[name].shape
    ]
    return {
        "source_model_type": src.model_type,
        "candidate_model_type": cand.model_type,
        "source_tensor_count": len(src.tensors),
        "candidate_tensor_count": len(cand.tensors),
        "missing_in_candidate": sorted(set(src_tensors) - set(cand_tensors)),
        "added_in_candidate": sorted(set(cand_tensors) - set(src_tensors)),
        "changed_shape_count": len(changed_shape),
        "changed_shapes": changed_shape[:200],
    }


def load_prompts(path: str | Path, max_prompts: int) -> List[str]:
    prompts: List[str] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) >= max_prompts:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    text = item.get("prompt") or item.get("text") or item.get("input")
                    if text is not None:
                        prompts.append(str(text))
                elif isinstance(item, str):
                    prompts.append(item)
            except json.JSONDecodeError:
                prompts.append(line)
    return prompts


def torch_logit_probe(
    source: str | Path,
    candidate: str | Path,
    prompts: str | Path,
    max_prompts: int,
    metadata_metrics: Dict[str, object],
) -> ProbeReport:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt_texts = load_prompts(prompts, max_prompts=max_prompts)
    if not prompt_texts:
        return ProbeReport(
            source=str(source),
            candidate=str(candidate),
            runtime="torch",
            prompt_count=0,
            metrics=metadata_metrics,
            warnings=["prompt file contained no prompts"],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
    src_model = AutoModelForCausalLM.from_pretrained(str(source), trust_remote_code=True, torch_dtype="auto").to(device).eval()
    cand_model = AutoModelForCausalLM.from_pretrained(str(candidate), trust_remote_code=True, torch_dtype="auto").to(device).eval()

    kl_values: List[float] = []
    top1_matches = 0
    token_positions = 0
    top5_overlaps: List[float] = []
    with torch.no_grad():
        for prompt in prompt_texts:
            batch = tokenizer(prompt, return_tensors="pt").to(device)
            src_logits = src_model(**batch).logits[:, :-1, :].float()
            cand_logits = cand_model(**batch).logits[:, :-1, :].float()
            src_log_probs = torch.log_softmax(src_logits, dim=-1)
            cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
            src_probs = src_log_probs.exp()
            kl = (src_probs * (src_log_probs - cand_log_probs)).sum(dim=-1)
            kl_values.extend(float(item) for item in kl.flatten().detach().cpu())
            src_top1 = src_logits.argmax(dim=-1)
            cand_top1 = cand_logits.argmax(dim=-1)
            top1_matches += int((src_top1 == cand_top1).sum().detach().cpu())
            token_positions += int(src_top1.numel())
            src_top5 = torch.topk(src_logits, k=5, dim=-1).indices
            cand_top5 = torch.topk(cand_logits, k=5, dim=-1).indices
            overlap = torch.zeros(src_top5.shape[:-1], device=device)
            for idx in range(5):
                overlap += (src_top5[..., idx : idx + 1] == cand_top5).any(dim=-1).float()
            top5_overlaps.extend(float(item / 5.0) for item in overlap.flatten().detach().cpu())

    metrics = dict(metadata_metrics)
    metrics["logit_probe"] = {
        "device": device,
        "mean_kl": sum(kl_values) / max(1, len(kl_values)),
        "max_kl": max(kl_values) if kl_values else None,
        "top1_agreement": top1_matches / max(1, token_positions),
        "mean_top5_overlap": sum(top5_overlaps) / max(1, len(top5_overlaps)),
        "token_positions": token_positions,
    }
    return ProbeReport(
        source=str(source),
        candidate=str(candidate),
        runtime="torch",
        prompt_count=len(prompt_texts),
        metrics=metrics,
        warnings=[],
    )
