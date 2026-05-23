from __future__ import annotations

from pathlib import Path

from .algorithm import run_bnd, run_monte_carlo, run_random_baseline
from .config import load_config, load_prompts
from .io import ensure_experiment_dirs, summarize_results
from .sampling import get_gpu_name, load_sd_pipeline
from .scoring import ClipScorer


def run_experiment(
    code_root: str | Path,
    config_name: str,
    storage_root: str | Path | None = None,
    prompt_limit: int | None = None,
):
    """
    Run a full BND experiment.

    code_root contains versioned code/configs/prompts. storage_root stores only
    runtime artifacts: model cache, generated images, JSON results, and summaries.
    """
    code_root = Path(code_root)
    storage_root = Path(storage_root) if storage_root is not None else code_root
    ensure_experiment_dirs(storage_root)

    cfg = load_config(code_root, config_name)
    prompts = load_prompts(code_root)
    if prompt_limit is not None:
        prompts = prompts[:prompt_limit]

    run_name = cfg["name"]
    cache_dir = storage_root / "cache" / "hf_cache"

    print("Config:", run_name)
    print("GPU:", get_gpu_name())
    print("Prompts:", len(prompts))
    print("Storage root:", storage_root)

    pipe = load_sd_pipeline(cfg, cache_dir)
    scorer = ClipScorer(cache_dir)

    for item in prompts:
        run_random_baseline(item, cfg, pipe, scorer, storage_root, run_name)
        run_monte_carlo(item, cfg, pipe, scorer, storage_root, run_name)
        run_bnd(item, cfg, pipe, scorer, storage_root, run_name)

    return summarize_results(storage_root, run_name)
