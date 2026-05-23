from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


METHODS = ["random", "monte_carlo", "bnd"]


def ensure_experiment_dirs(storage_root: str | Path) -> None:
    """Create only runtime storage folders: cache, outputs, and summaries."""
    storage_root = Path(storage_root)
    for path in [
        storage_root / "outputs",
        storage_root / "outputs" / "summary",
        storage_root / "cache" / "hf_cache",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def result_path(storage_root: str | Path, run_name: str, method: str, prompt_id: str) -> Path:
    path = Path(storage_root) / "outputs" / "results" / run_name / method / f"{prompt_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def image_path(storage_root: str | Path, run_name: str, method: str, prompt_id: str, suffix: str = "best") -> Path:
    path = Path(storage_root) / "outputs" / "images" / run_name / method / f"{prompt_id}_{suffix}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def already_done(storage_root: str | Path, run_name: str, method: str, prompt_id: str) -> bool:
    return result_path(storage_root, run_name, method, prompt_id).exists()


def summarize_results(storage_root: str | Path, run_name: str, methods: list[str] | None = None):
    storage_root = Path(storage_root)
    methods = methods or METHODS
    rows = []

    for method in methods:
        result_dir = storage_root / "outputs" / "results" / run_name / method
        for fp in sorted(result_dir.glob("*.json")):
            data = load_json(fp)
            rows.append(
                {
                    "prompt_id": data["prompt_id"],
                    "prompt": data["prompt"],
                    "group": data.get("group"),
                    "method": data["method"],
                    "score": data["score"],
                    "runtime_sec": data["runtime_sec"],
                    "gpu": data["gpu"],
                    "image_path": data["image_path"],
                }
            )

    df = pd.DataFrame(rows)
    summary_dir = storage_root / "outputs" / "summary" / run_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / "results_all.csv", index=False)

    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    method_summary = (
        df.groupby("method")
        .agg(
            mean_clipscore=("score", "mean"),
            std_clipscore=("score", "std"),
            mean_runtime=("runtime_sec", "mean"),
            count=("score", "count"),
        )
        .reset_index()
    )
    group_summary = (
        df.groupby(["group", "method"])
        .agg(
            mean_clipscore=("score", "mean"),
            mean_runtime=("runtime_sec", "mean"),
            count=("score", "count"),
        )
        .reset_index()
    )

    method_summary.to_csv(summary_dir / "summary_by_method.csv", index=False)
    group_summary.to_csv(summary_dir / "summary_by_group.csv", index=False)
    return df, method_summary, group_summary
