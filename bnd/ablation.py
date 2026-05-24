from __future__ import annotations

from pathlib import Path

from .algorithm import run_bnd, run_bnd_no_full_rerank, run_bnd_rounds, run_monte_carlo
from .io import summarize_results


def ablation_methods(rounds: tuple[int, ...]) -> list[str]:
    """Return the method names produced by the ablation runner."""
    return ["monte_carlo", "bnd", "bnd_no_full_rerank"] + [f"bnd_R{r}" for r in rounds]


def run_ablation_studies(
    prompts: list[dict],
    cfg: dict,
    pipe,
    scorer,
    storage_root: str | Path,
    run_name: str,
    prompt_limit: int | None = None,
    rounds: tuple[int, ...] = (0, 1, 2, 3),
):
    """Run BND ablation variants and save raw and summarized results."""
    items = prompts if prompt_limit is None else prompts[:prompt_limit]

    for item in items:
        # Ablation 1: remove beam-guided refinement by using Monte-Carlo only.
        run_monte_carlo(item, cfg, pipe, scorer, storage_root, run_name)
        run_bnd(item, cfg, pipe, scorer, storage_root, run_name)

        # Ablation 2: remove full-sampling re-ranking.
        run_bnd_no_full_rerank(item, cfg, pipe, scorer, storage_root, run_name)

        # Ablation 3: vary the number of refinement rounds.
        for r in rounds:
            run_bnd_rounds(item, cfg, pipe, scorer, storage_root, run_name, rounds=r)

    return summarize_results(
        storage_root,
        run_name,
        methods=ablation_methods(rounds),
        output_prefix="ablation_",
    )
