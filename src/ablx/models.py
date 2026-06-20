from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: List[int]
    data_offsets: List[int]
    shard: str

    @property
    def nbytes(self) -> int:
        return int(self.data_offsets[1]) - int(self.data_offsets[0])

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    path: str
    config: JsonDict
    tensors: List[TensorInfo]
    weight_map: Dict[str, str] = field(default_factory=dict)
    total_tensor_bytes: int = 0

    @property
    def model_type(self) -> str:
        text_config = self.config.get("text_config")
        if isinstance(text_config, Mapping) and text_config.get("model_type"):
            return str(text_config["model_type"])
        return str(self.config.get("model_type", "unknown"))

    def tensor(self, name: str) -> TensorInfo:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def to_dict(self) -> JsonDict:
        return {
            "path": self.path,
            "model_type": self.model_type,
            "total_tensor_bytes": self.total_tensor_bytes,
            "tensor_count": len(self.tensors),
            "weight_map_count": len(self.weight_map),
            "config_summary": summarize_config(self.config),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


@dataclass(frozen=True)
class TransformOp:
    type: str
    params: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {"type": self.type, **self.params}


@dataclass(frozen=True)
class UpscaleRecipe:
    name: str
    description: str = ""
    transforms: List[TransformOp] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "transforms": [op.to_dict() for op in self.transforms],
        }


@dataclass(frozen=True)
class TensorMapping:
    name: str
    source_shape: List[int]
    target_shape: List[int]
    transform: str
    trainable_slices: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ExpansionReport:
    source: str
    output: str
    recipe: JsonDict
    changed_tensors: List[TensorMapping]
    copied_tensors: int
    warnings: List[str] = field(default_factory=list)
    slime_hints: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "source": self.source,
            "output": self.output,
            "recipe": self.recipe,
            "changed_tensors": [mapping.to_dict() for mapping in self.changed_tensors],
            "copied_tensors": self.copied_tensors,
            "warnings": list(self.warnings),
            "slime_hints": self.slime_hints,
        }


@dataclass(frozen=True)
class ProbeReport:
    source: str
    candidate: str
    runtime: str
    prompt_count: int = 0
    metrics: JsonDict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkReport:
    stage: str
    source: str
    candidate: str
    out: str
    suite: str
    backend: str
    commands: List[str] = field(default_factory=list)
    command_results: List[JsonDict] = field(default_factory=list)
    metrics: JsonDict = field(default_factory=dict)
    gates: JsonDict = field(default_factory=dict)
    passed: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class TrainPlanReport:
    source: str
    candidate: str
    out: str
    backend: str
    mode: str
    launch_training: bool
    commands: List[str] = field(default_factory=list)
    command_results: List[JsonDict] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class PipelineStepReport:
    name: str
    status: str
    out: str = ""
    artifacts: List[str] = field(default_factory=list)
    metrics: JsonDict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class PipelineReport:
    name: str
    source: str
    candidate: str
    out: str
    dry_run: bool
    accepted: bool
    steps: List[PipelineStepReport]
    artifacts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "source": self.source,
            "candidate": self.candidate,
            "out": self.out,
            "dry_run": self.dry_run,
            "accepted": self.accepted,
            "steps": [step.to_dict() for step in self.steps],
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
        }


def summarize_config(config: Mapping[str, Any]) -> JsonDict:
    text_config = config.get("text_config")
    active = text_config if isinstance(text_config, Mapping) else config
    keys = [
        "architectures",
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "mtp_num_hidden_layers",
        "num_nextn_predict_layers",
    ]
    summary: JsonDict = {}
    for key in keys:
        if key in active:
            summary[key] = active[key]
        elif key in config:
            summary[key] = config[key]
    if text_config is not active and isinstance(text_config, Mapping):
        summary["text_config_model_type"] = text_config.get("model_type")
    return summary


def path_str(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def tensor_names(tensors: Sequence[TensorInfo]) -> List[str]:
    return [tensor.name for tensor in tensors]
