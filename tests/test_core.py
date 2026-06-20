import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ablx.inspection import inspect_model
from ablx.probe import probe_checkpoints
from ablx.recipes import UpscaleRecipe
from ablx.models import TransformOp
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
