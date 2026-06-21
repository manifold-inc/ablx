from __future__ import annotations

from typing import Any


def data_manifest(data_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "mixture": data_config,
        "contamination_policy": "benchmark and preservation probes must be held out",
        "required_domains": [
            "parent_preservation",
            "high_quality_text",
            "uncertainty_amplifying",
            "failure_mode",
            "long_context",
            "multimodal",
            "safety_adversarial",
        ],
    }
