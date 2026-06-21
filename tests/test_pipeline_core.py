from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import ablx.checkpoint.resolve as resolve_mod
from ablx.checkpoint.resolve import resolve_checkpoint_ref
from ablx.benchmark.compare import compare_reports
from ablx.benchmark.runner import run_benchmark
from ablx.compute.estimate import compute_plan
from ablx.config import ProbeGatesConfig, load_config
from ablx.fixtures import create_tiny_checkpoint
from ablx.pipeline import run_pipeline
from ablx.probe.gates import evaluate_gate, next_token_metrics
from ablx.train.checkpoint import trained_checkpoint_dir
from ablx.train.trainer import build_training_commands, train_model
from ablx.transforms.expand import expand_checkpoint
from ablx.upsample.constrained_noise import upsample_checkpoint


def write_config(tmp_path: Path, *, launch: bool = False, backend: str = "local_smoke") -> Path:
    parent = tmp_path / "parent"
    out = tmp_path / "run"
    create_tiny_checkpoint(parent)
    cfg = {
        "name": "tiny_moe_smoke",
        "parent": {"path": str(parent), "dtype": "float32"},
        "expand": {
            "source": str(parent),
            "transforms": [
                {
                    "type": "expand_moe_intermediate",
                    "old_intermediate_size": 2,
                    "new_intermediate_size": 3,
                    "include_shared_expert": True,
                    "include_mtp": True,
                },
                {"type": "clone_experts", "factor": 2, "zero_mean_deltas": True},
            ],
        },
        "upsample": {"generator": "constrained_noise", "noise_std": 0.0, "noise_seed": 7},
        "benchmark": {"suites": ["core"], "soft_gate": True, "limit": 4},
        "train": {
            "launch": launch,
            "backend": backend,
            "num_gpus": 8,
            "seq_len": 128,
            "micro_batch": 1,
            "grad_accum": 1,
            "data": {"text": ["tiny training sample"]},
        },
        "output_dir": str(out),
    }
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_expand_widens_and_clones_tiny_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path))
    report = expand_checkpoint(cfg)
    assert report["config_patch"]["moe_intermediate_size"] == 3
    assert report["config_patch"]["num_experts"] == 4
    tensors = load_file(str(cfg.out_dir / "expanded" / "model.safetensors"))
    assert tensors["model.layers.0.mlp.experts.0.gate_up_proj.weight"].shape == (6, 4)
    assert tensors["model.layers.0.mlp.experts.0.down_proj.weight"].shape == (4, 3)
    assert "model.layers.0.mlp.experts.3.down_proj.weight" in tensors
    assert tensors["model.layers.0.mlp.router.weight"].shape == (4, 4)


def test_upsample_zero_noise_preserves_tensors(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path))
    expand_checkpoint(cfg)
    before = load_file(str(cfg.out_dir / "expanded" / "model.safetensors"))
    report = upsample_checkpoint(cfg)
    after = load_file(str(cfg.out_dir / "upsampled" / "model.safetensors"))
    assert report["affected_tensor_count"] > 0
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name])


def test_probe_metrics_and_gate_failure() -> None:
    parent = torch.tensor([[0.0, 5.0, 1.0]])
    child = torch.tensor([[5.0, 0.0, 1.0]])
    metrics = next_token_metrics(parent, child)
    gate = evaluate_gate(metrics, ProbeGatesConfig())
    assert metrics["top1_agreement"] == 0.0
    assert gate["passed"] is False


def test_benchmark_compare_reports_regressions(tmp_path: Path) -> None:
    parent = {"model": "parent", "scores": {"humaneval_mini_pass@1": 0.8, "mbpp_mini_pass@1": 0.8}}
    child = {"model": "child", "scores": {"humaneval_mini_pass@1": 0.6, "mbpp_mini_pass@1": 0.6}}
    delta = compare_reports(parent, child, thresholds={"coding_aggregate": -0.02}, out_dir=tmp_path)
    assert delta["warnings"]
    assert (tmp_path / "BENCH_SUMMARY.md").exists()
    assert (tmp_path / "BENCH_SUMMARY_candidate.md").exists()


def test_compute_plan_and_pipeline(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path))
    estimate = compute_plan(cfg)
    assert estimate["hardware"]["num_gpus"] == 8
    report = run_pipeline(cfg, config_path=tmp_path / "config.yaml", dry_run=True)
    assert report["stages"]["train"]["status"] == "launch_disabled_plan_emitted"
    assert report["stages"]["bench_final"]["skipped"] is True
    assert report["benchmark_targets"]["final"]["benchmarked"] is False


def test_native_benchmark_report(tmp_path: Path) -> None:
    report = run_benchmark("model-a", out=tmp_path, limit=2)
    assert len(report["scores"]) == 2
    assert (tmp_path / "bench_report.json").exists()
    saved = json.loads((tmp_path / "bench_report.json").read_text(encoding="utf-8"))
    assert saved["suite"] == "core"


def test_hf_repo_id_is_not_treated_as_relative_path(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "hf-snapshot"
    target.mkdir()

    def fake_download(repo_id, *, parent=None):
        assert repo_id == "Qwen/Qwen3.6-35B-A3B"
        return target

    monkeypatch.setattr(resolve_mod, "download_hf_checkpoint", fake_download)
    assert resolve_checkpoint_ref("Qwen/Qwen3.6-35B-A3B") == target


def test_build_commands_use_real_config_path_and_worker(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, launch=True)
    cfg = load_config(config_path)
    command = build_training_commands(cfg, config_path=config_path)[0]
    assert "${CONFIG}" not in " ".join(command)
    assert str(config_path.resolve()) in command
    assert "train-worker" in command


def test_train_launch_local_worker_writes_checkpoint(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, launch=True)
    cfg = load_config(config_path)
    expand_checkpoint(cfg)
    upsample_checkpoint(cfg)
    report = train_model(cfg, config_path=config_path)
    assert report["status"] == "completed"
    assert trained_checkpoint_dir(cfg).exists()
    assert (trained_checkpoint_dir(cfg) / "model.safetensors").exists()


def test_pipeline_launch_benchmarks_trained_checkpoint(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, launch=True)
    cfg = load_config(config_path)
    report = run_pipeline(cfg, config_path=config_path)
    assert report["stages"]["train"]["status"] == "completed"
    assert report["stages"]["bench_final"]["checkpoint"]["role"] == "trained"
    assert report["stages"]["bench_final"]["model"] == str(trained_checkpoint_dir(cfg))
    assert report["stages"]["bench_final"]["model"] != str(cfg.out_dir / "upsampled")
    assert (cfg.out_dir / "delta_vs_parent.json").exists()
    assert (cfg.out_dir / "BENCH_SUMMARY_final.md").exists()
