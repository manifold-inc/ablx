from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    domain: str
    metric: str
    higher_is_better: bool = True
    contamination_tag: str = "default_clean_split"


CORE_TASKS = [
    BenchmarkTask("humaneval_mini_pass@1", "coding_aggregate", "pass@1"),
    BenchmarkTask("mbpp_mini_pass@1", "coding_aggregate", "pass@1"),
    BenchmarkTask("gsm8k_mini_exact_match", "math_aggregate", "exact_match"),
    BenchmarkTask("math_mini_exact_match", "math_aggregate", "exact_match"),
    BenchmarkTask("mmmu_mini_accuracy", "multimodal_aggregate", "accuracy"),
    BenchmarkTask("chartqa_mini_accuracy", "multimodal_aggregate", "accuracy"),
    BenchmarkTask("ruler_32k_accuracy", "long_context_aggregate", "accuracy"),
    BenchmarkTask("needle_128k_recall", "long_context_aggregate", "recall"),
]


def suite_tasks(name: str) -> list[BenchmarkTask]:
    if name == "core":
        return CORE_TASKS
    raise KeyError(f"unknown benchmark suite: {name}")


def aggregate_domains(scores: dict[str, float]) -> dict[str, float]:
    aggregates: dict[str, list[float]] = {}
    task_lookup = {task.name: task.domain for task in CORE_TASKS}
    for task, score in scores.items():
        domain = task_lookup.get(task)
        if domain:
            aggregates.setdefault(domain, []).append(float(score))
    return {domain: sum(values) / len(values) for domain, values in aggregates.items() if values}
