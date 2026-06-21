from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .errors import ConfigError
from .utils import read_yaml


class ParentConfig(BaseModel):
    hf_id: str | None = None
    path: str | None = None
    dtype: str = "bfloat16"

    @field_validator("hf_id", "path")
    @classmethod
    def empty_to_none(cls, value: str | None) -> str | None:
        return value or None

    def reference(self) -> str:
        if self.path:
            return self.path
        if self.hf_id:
            return self.hf_id
        raise ConfigError("parent.hf_id or parent.path is required")


class ExpandTransformConfig(BaseModel):
    type: Literal["expand_moe_intermediate", "clone_experts"]
    old_intermediate_size: int | None = None
    new_intermediate_size: int | None = None
    factor: int = 1
    zero_mean_deltas: bool = True
    include_shared_expert: bool = True
    include_mtp: bool = True


class ExpandConfig(BaseModel):
    source: str | None = None
    transforms: list[ExpandTransformConfig] = Field(default_factory=list)


class UpsampleConfig(BaseModel):
    generator: str = "constrained_noise"
    noise_std: float = 1.0e-6
    noise_seed: int = 1729
    project_to_constraint_set: bool = True
    max_delta_norm: float | None = None


class ProbeGatesConfig(BaseModel):
    max_kl: float = 0.05
    top1_agreement: float = 0.98
    top5_overlap: float = 0.98
    max_logit_abs_diff: float = 1.0e-3
    max_router_load_l1: float = 0.25


class ProbeConfig(BaseModel):
    text_prompts: str | None = None
    vision_prompts: str | None = None
    gates: ProbeGatesConfig = Field(default_factory=ProbeGatesConfig)
    max_prompts: int | None = None


class BenchmarkConfig(BaseModel):
    suites: list[Any] = Field(default_factory=lambda: ["core"])
    soft_gate: bool = True
    force_continue: bool = False
    hard_fail_on_regression: bool = False
    regression_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"coding_aggregate": -0.02, "math_aggregate": -0.02}
    )
    compare_to: str = "parent"
    limit: int | None = None


class TrainStageConfig(BaseModel):
    name: str
    steps: int = 0
    freeze: str = "old_slices"
    lambda_kl: float = 1.0
    lambda_ce: float = 0.0
    lambda_act: float = 0.0
    lambda_anchor: float = 0.0
    train_router: str | None = None


class TrainConfig(BaseModel):
    launch: bool = False
    backend: str = "accelerate_fsdp"
    num_gpus: int = 8
    micro_batch: int = 1
    seq_len: int = 4096
    grad_accum: int = 8
    stages: list[TrainStageConfig] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    checkpoint_every: int = 1000


class PipelineConfig(BaseModel):
    name: str = "ablx_run"
    parent: ParentConfig = Field(default_factory=ParentConfig)
    expand: ExpandConfig = Field(default_factory=ExpandConfig)
    upsample: UpsampleConfig = Field(default_factory=UpsampleConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    output_dir: str = "runs/ablx"

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir).expanduser()


def load_config(path: str | Path) -> PipelineConfig:
    data = read_yaml(path)
    return PipelineConfig.model_validate(data)
