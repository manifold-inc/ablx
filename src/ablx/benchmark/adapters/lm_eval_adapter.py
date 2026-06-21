from __future__ import annotations

from pathlib import Path
from typing import Any

from ablx.errors import OptionalDependencyError


def run_lm_eval(
    model: str | Path,
    *,
    tasks: list[str],
    backend: str = "hf",
    num_fewshot: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        import lm_eval  # type: ignore # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise OptionalDependencyError("lm-eval is not installed; install ablx[benchmark]") from exc

    # We keep the adapter thin and explicit. Full dispatch is backend-dependent
    # and should be exercised on the target GPU host.
    return {
        "backend": f"lm_eval:{backend}",
        "model": str(model),
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "scores": {},
        "note": "lm-eval adapter available; invoke on GPU host for real scores",
    }
