import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from ablx.cli import main
from ablx.errors import AblxError
from ablx.inspection import inspect_model
from ablx.models import TransformOp
from ablx.pipeline import run_pipeline
from ablx.probe import probe_checkpoints
from ablx.recipes import UpscaleRecipe
from ablx.safetensors_io import TensorPayload, read_tensor_payload, list_tensor_infos, write_safetensors
from ablx.slime import emit_slime_plan
from ablx.transforms import upscale_checkpoint


class AblxCoreTests(unittest.TestCase):
    def test_inspect_reads_config_and_safetensor_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = make_tiny_model(Path(tmp) / "src")

            spec = inspect_model(model_dir)

            self.assertEqual(spec.model_type, "qwen3_5_moe_text")
            self.assertEqual(len(spec.tensors), 8)
            self.assertEqual(spec.tensor("model.language_model.layers.0.mlp.experts.gate_up_proj").shape, [2, 4, 3])

    def test_inspect_accepts_fp8_safetensors_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = make_tiny_model(Path(tmp) / "fp8", dtype="F8_E4M3")

            spec = inspect_model(model_dir)

            tensor = spec.tensor("model.language_model.layers.0.mlp.experts.gate_up_proj")
            self.assertEqual(tensor.dtype, "F8_E4M3")
            self.assertEqual(tensor.nbytes, 2 * 4 * 3)

    def test_expand_moe_intermediate_maps_fused_and_shared_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "out"
            recipe = UpscaleRecipe(
                name="tiny_expand",
                transforms=[
                    TransformOp(
                        "expand_moe_intermediate",
                        {
                            "old_intermediate_size": 2,
                            "new_intermediate_size": 3,
                            "noise_std": 0.0,
                            "noise_seed": 7,
                            "include_shared_expert": True,
                        },
                    )
                ],
            )

            report = upscale_checkpoint(source, recipe, out)

            self.assertTrue((out / "ablx_expansion_report.json").exists())
            self.assertGreaterEqual(len(report.changed_tensors), 5)
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["text_config"]["moe_intermediate_size"], 3)
            self.assertEqual(config["text_config"]["shared_expert_intermediate_size"], 3)

            infos = {info.name: info for info in list_tensor_infos(out / "model.safetensors")}
            gate_up = tensor_array(out / "model.safetensors", infos["model.language_model.layers.0.mlp.experts.gate_up_proj"])
            source_gate_up = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
            self.assertEqual(list(gate_up.shape), [2, 6, 3])
            np.testing.assert_array_equal(gate_up[:, :2, :], source_gate_up[:, :2, :])
            np.testing.assert_array_equal(gate_up[:, 3:5, :], source_gate_up[:, 2:4, :])
            np.testing.assert_array_equal(gate_up[:, 2:3, :], np.zeros((2, 1, 3), dtype=np.float32))
            np.testing.assert_array_equal(gate_up[:, 5:6, :], np.zeros((2, 1, 3), dtype=np.float32))

            down = tensor_array(out / "model.safetensors", infos["model.language_model.layers.0.mlp.experts.down_proj"])
            source_down = np.arange(100, 100 + 2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
            self.assertEqual(list(down.shape), [2, 3, 3])
            np.testing.assert_array_equal(down[:, :, :2], source_down)
            np.testing.assert_array_equal(down[:, :, 2:], np.zeros((2, 3, 1), dtype=np.float32))

            shared_gate = tensor_array(out / "model.safetensors", infos["model.language_model.layers.0.mlp.shared_expert.gate_proj"])
            self.assertEqual(list(shared_gate.shape), [3, 3])
            np.testing.assert_array_equal(shared_gate[:2], np.arange(200, 206, dtype=np.float32).reshape(2, 3))
            np.testing.assert_array_equal(shared_gate[2:], np.zeros((1, 3), dtype=np.float32))

            index = json.loads((out / "model.safetensors.index.json").read_text())
            self.assertEqual(index["weight_map"]["model.language_model.layers.0.mlp.experts.down_proj"], "model.safetensors")
            self.assertGreater(index["metadata"]["total_size"], 0)

    def test_clone_experts_expands_expert_and_router_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "clone"
            recipe = UpscaleRecipe(
                name="tiny_clone",
                transforms=[
                    TransformOp(
                        "clone_experts",
                        {
                            "old_num_experts": 2,
                            "new_num_experts": 3,
                            "source_indices": [1],
                            "router_init": "clone",
                        },
                    )
                ],
            )

            report = upscale_checkpoint(source, recipe, out)

            self.assertTrue(any("experimental" in warning for warning in report.warnings))
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["text_config"]["num_experts"], 3)
            infos = {info.name: info for info in list_tensor_infos(out / "model.safetensors")}
            gate_up = tensor_array(out / "model.safetensors", infos["model.language_model.layers.0.mlp.experts.gate_up_proj"])
            self.assertEqual(list(gate_up.shape), [3, 4, 3])
            np.testing.assert_array_equal(gate_up[2], gate_up[1])
            router = tensor_array(out / "model.safetensors", infos["model.language_model.layers.0.mlp.gate.weight"])
            self.assertEqual(list(router.shape), [3, 3])
            np.testing.assert_array_equal(router[2], router[1])

    def test_expand_bf16_preserves_raw_words_and_zeroes_new_down_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src", dtype="BF16")
            out = root / "bf16"
            recipe = UpscaleRecipe(
                name="tiny_bf16_expand",
                transforms=[
                    TransformOp(
                        "expand_moe_intermediate",
                        {"old_intermediate_size": 2, "new_intermediate_size": 3, "noise_std": 0.0},
                    )
                ],
            )

            upscale_checkpoint(source, recipe, out)

            infos = {info.name: info for info in list_tensor_infos(out / "model.safetensors")}
            gate_up = tensor_words(out / "model.safetensors", infos["model.language_model.layers.0.mlp.experts.gate_up_proj"])
            source_gate_up = np.arange(2 * 4 * 3, dtype=np.uint16).reshape(2, 4, 3)
            self.assertEqual(infos["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype, "BF16")
            np.testing.assert_array_equal(gate_up[:, :2, :], source_gate_up[:, :2, :])
            np.testing.assert_array_equal(gate_up[:, 3:5, :], source_gate_up[:, 2:4, :])

            down = tensor_words(out / "model.safetensors", infos["model.language_model.layers.0.mlp.experts.down_proj"])
            np.testing.assert_array_equal(down[:, :, 2:], np.zeros((2, 3, 1), dtype=np.uint16))

    def test_probe_and_slime_plan_emit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "out"
            plan_dir = root / "slime"
            recipe = UpscaleRecipe(
                name="tiny_expand",
                transforms=[
                    TransformOp(
                        "expand_moe_intermediate",
                        {"old_intermediate_size": 2, "new_intermediate_size": 3, "noise_std": 0.0},
                    )
                ],
            )
            upscale_checkpoint(source, recipe, out)

            probe = probe_checkpoints(source, out)
            self.assertEqual(probe.runtime, "metadata-only")
            self.assertGreater(probe.metrics["changed_shape_count"], 0)

            plan = emit_slime_plan(out, plan_dir)
            self.assertEqual(sorted(plan["files"]), ["README.md", "commands.json", "freeze_masks.yaml", "slime_phases.yaml"])
            self.assertTrue((plan_dir / "slime_phases.yaml").exists())

    def test_pipeline_executes_all_stages_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "run"
            config = pipeline_config(source, out)

            report = run_pipeline(config)

            self.assertFalse(report.dry_run)
            self.assertTrue(report.accepted)
            self.assertEqual([step.name for step in report.steps], ["inspect", "upscale", "probe", "bench-pre", "train", "bench-post"])
            self.assertTrue((out / "candidate" / "ablx_expansion_report.json").exists())
            self.assertTrue((out / "probe" / "ablx_probe_report.json").exists())
            self.assertTrue((out / "bench" / "pre" / "summary.json").exists())
            self.assertTrue((out / "train" / "reverse_distill_plan.yaml").exists())
            self.assertTrue((out / "bench" / "post" / "summary.json").exists())
            self.assertTrue((out / "ablx_pipeline_report.json").exists())

    def test_pipeline_preserves_hf_source_id_until_model_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_model = make_tiny_model(root / "resolved")
            out = root / "hf"
            config = pipeline_config(local_model, out)
            config["source"] = "Qwen/Qwen3.6-35B-A3B-FP8"

            with patch("ablx.pipeline.inspect_model") as inspect_mock:
                inspect_mock.return_value = inspect_model(local_model)
                report = run_pipeline(config, {"stop_after": "inspect"})

            inspect_mock.assert_called_once_with("Qwen/Qwen3.6-35B-A3B-FP8")
            self.assertEqual(report.steps[0].out, str(local_model.resolve()))

    def test_upscale_checkpoint_resolves_hf_source_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_model = make_tiny_model(root / "resolved")
            out = root / "hf_upscale"
            recipe = UpscaleRecipe(
                name="tiny_expand",
                transforms=[
                    TransformOp(
                        "expand_moe_intermediate",
                        {"old_intermediate_size": 2, "new_intermediate_size": 3, "noise_std": 0.0},
                    )
                ],
            )

            with patch("ablx.transforms.resolve_model_path", return_value=local_model.resolve()) as resolver:
                report = upscale_checkpoint("Qwen/Qwen3.6-35B-A3B-FP8", recipe, out)

            resolver.assert_called_once_with("Qwen/Qwen3.6-35B-A3B-FP8")
            self.assertEqual(report.source, str(local_model.resolve()))
            self.assertTrue((out / "ablx_expansion_report.json").exists())

    def test_pipeline_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "dry"
            config = pipeline_config(source, out)
            config["dry_run"] = True

            report = run_pipeline(config)

            self.assertTrue(report.dry_run)
            self.assertFalse(out.exists())
            self.assertTrue(all(step.status in {"planned", "skipped"} for step in report.steps))

    def test_pipeline_gate_failure_stops_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "gate"
            config = pipeline_config(source, out)
            config["benchmark"]["gates"]["pre"]["changed_shape_count_min"] = 99

            with self.assertRaises(AblxError):
                run_pipeline(config)

            self.assertTrue((out / "bench" / "pre" / "summary.json").exists())
            self.assertTrue((out / "ablx_pipeline_report.json").exists())
            self.assertFalse((out / "train").exists())

    def test_pipeline_continue_on_gate_failure_warns_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "warn"
            config = pipeline_config(source, out)
            config["continue_on_gate_fail"] = True
            config["benchmark"]["gates"]["pre"]["changed_shape_count_min"] = 99

            report = run_pipeline(config)

            self.assertFalse(report.accepted)
            self.assertTrue(any("pre-benchmark" in warning for warning in report.warnings))
            self.assertTrue((out / "train" / "reverse_distill_plan.yaml").exists())
            self.assertTrue((out / "bench" / "post" / "summary.json").exists())

    def test_pipeline_missing_training_backend_fails_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "missing_backend"
            config = pipeline_config(source, out)
            config["reverse_distill"] = {"enabled": True, "backend": "slime", "launch_training": True, "emit_slime_plan": False}

            with self.assertRaises(AblxError):
                run_pipeline(config)

    def test_pipeline_skip_train_emits_no_training_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "skip_train"
            config = pipeline_config(source, out)
            config["skip_train"] = True

            report = run_pipeline(config)

            train_step = next(step for step in report.steps if step.name == "train")
            self.assertEqual(train_step.status, "skipped")
            self.assertFalse((out / "train" / "commands.json").exists())

    def test_cli_pipeline_returns_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_tiny_model(root / "src")
            out = root / "cli"
            config_path = root / "pipeline.yaml"
            config_path.write_text(yaml.safe_dump(pipeline_config(source, out), sort_keys=False), encoding="utf-8")
            stdout = StringIO()

            with redirect_stdout(stdout):
                code = main(["pipeline", "--config", str(config_path)])

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["name"], "tiny_pipeline")
            self.assertTrue(payload["accepted"])


def make_tiny_model(model_dir: Path, dtype: str = "F32") -> Path:
    model_dir.mkdir(parents=True)
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 3,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "vocab_size": 16,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = [
        payload("model.language_model.layers.0.mlp.experts.gate_up_proj", array_data(0, 2 * 4 * 3, (2, 4, 3), dtype), dtype),
        payload("model.language_model.layers.0.mlp.experts.down_proj", array_data(100, 2 * 3 * 2, (2, 3, 2), dtype), dtype),
        payload("model.language_model.layers.0.mlp.gate.weight", array_data(300, 2 * 3, (2, 3), dtype), dtype),
        payload("model.language_model.layers.0.mlp.shared_expert.gate_proj", array_data(200, 2 * 3, (2, 3), dtype), dtype),
        payload("model.language_model.layers.0.mlp.shared_expert.up_proj", array_data(210, 2 * 3, (2, 3), dtype), dtype),
        payload("model.language_model.layers.0.mlp.shared_expert.down_proj", array_data(220, 3 * 2, (3, 2), dtype), dtype),
        payload("model.language_model.layers.0.self_attn.q_proj.weight", array_data(400, 3 * 3, (3, 3), dtype), dtype),
        payload("lm_head.weight", array_data(500, 16 * 3, (16, 3), dtype), dtype),
    ]
    write_safetensors(model_dir / "model.safetensors", tensors, metadata={"format": "pt"})
    weight_map = {tensor.name: "model.safetensors" for tensor in tensors}
    index = {"metadata": {"total_size": sum(tensor.nbytes for tensor in tensors)}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model_dir


def pipeline_config(source: Path, out: Path) -> dict:
    return {
        "name": "tiny_pipeline",
        "source": str(source),
        "out": str(out),
        "upscale": {
            "enabled": True,
            "out": "candidate",
            "recipe": {
                "name": "tiny_expand",
                "transforms": [
                    {
                        "type": "expand_moe_intermediate",
                        "old_intermediate_size": 2,
                        "new_intermediate_size": 3,
                        "noise_std": 0.0,
                    }
                ],
            },
        },
        "probe": {"enabled": True, "max_prompts": 2},
        "benchmark": {
            "enabled": True,
            "suite": "preserve",
            "serve_backend": "metadata",
            "run_commands": True,
            "gates": {
                "pre": {"missing_in_candidate_count_max": 0, "changed_shape_count_min": 1},
                "post": {"missing_in_candidate_count_max": 0},
            },
        },
        "reverse_distill": {
            "enabled": True,
            "backend": "local",
            "mode": "light",
            "launch_training": True,
            "emit_slime_plan": False,
        },
    }


def array_data(start: int, count: int, shape, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        np_dtype = np.uint16
    elif dtype.startswith("F8_"):
        return (np.arange(start, start + count, dtype=np.uint16) % 256).astype(np.uint8).reshape(shape)
    else:
        np_dtype = np.float32
    return np.arange(start, start + count, dtype=np_dtype).reshape(shape)


def payload(name: str, array: np.ndarray, dtype: str = "F32") -> TensorPayload:
    return TensorPayload(name=name, dtype=dtype, shape=list(array.shape), data=array.tobytes(order="C"))


def tensor_array(shard: Path, info):
    return np.frombuffer(read_tensor_payload(shard, info).data, dtype=np.float32).reshape(info.shape)


def tensor_words(shard: Path, info):
    return np.frombuffer(read_tensor_payload(shard, info).data, dtype=np.uint16).reshape(info.shape)


if __name__ == "__main__":
    unittest.main()
