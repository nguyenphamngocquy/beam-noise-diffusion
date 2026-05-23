from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(root: str | Path, config_name: str) -> dict[str, Any]:
    root = Path(root)
    cfg = load_json(root / "configs" / f"{config_name}.json")
    validate_config(cfg)
    return cfg


def load_prompts(root: str | Path, filename: str = "prompts_30.json") -> list[dict[str, Any]]:
    prompts = load_json(Path(root) / "prompts" / filename)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt file must contain a non-empty list")
    for idx, item in enumerate(prompts):
        if "id" not in item or "prompt" not in item:
            raise ValueError(f"Prompt at index {idx} must contain 'id' and 'prompt'")
    return prompts


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "name",
        "model_id",
        "resolution",
        "guidance_scale",
        "seed_base",
        "eval_batch_size",
        "B_init",
        "n_batch",
        "k1",
        "R",
        "a_r",
        "m_r",
        "q_r",
        "k_r",
        "gamma",
        "T_preview",
        "T_mid",
        "T_full",
        "b",
        "epsilon",
        "patience",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    positive_ints = [
        "resolution",
        "eval_batch_size",
        "B_init",
        "n_batch",
        "k1",
        "T_preview",
        "T_mid",
        "T_full",
        "b",
        "patience",
    ]
    for key in positive_ints:
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")

    if not isinstance(cfg["R"], int) or cfg["R"] < 0:
        raise ValueError("R must be a non-negative integer")
    if cfg["resolution"] % 8 != 0:
        raise ValueError("resolution must be divisible by 8 for Stable Diffusion latents")
    if not (0 < float(cfg["gamma"]) <= 1):
        raise ValueError("gamma should be in (0, 1]")
    if float(cfg["guidance_scale"]) <= 0:
        raise ValueError("guidance_scale must be > 0")
    if float(cfg["epsilon"]) < 0:
        raise ValueError("epsilon must be >= 0")
    if cfg["T_preview"] > cfg["T_mid"] or cfg["T_mid"] > cfg["T_full"]:
        raise ValueError("Expected T_preview <= T_mid <= T_full")
    if cfg["k1"] > cfg["B_init"] * cfg["n_batch"]:
        raise ValueError("k1 cannot exceed B_init * n_batch")

    R = cfg["R"]
    for key in ["a_r", "m_r", "q_r", "k_r"]:
        values = cfg[key]
        if not isinstance(values, list) or len(values) != R:
            raise ValueError(f"{key} length must match R={R}")
        if any((not isinstance(v, int) or v <= 0) for v in values):
            raise ValueError(f"{key} must contain positive integers")

    prev_k = cfg["k1"]
    for r in range(R):
        if cfg["a_r"][r] > prev_k:
            raise ValueError(f"a_r[{r}] cannot exceed previous beam size")
        max_proposals = cfg["a_r"][r] * cfg["m_r"][r]
        if cfg["q_r"][r] > max_proposals:
            raise ValueError(f"q_r[{r}] cannot exceed a_r[{r}] * m_r[{r}]")
        if cfg["k_r"][r] < cfg["b"]:
            raise ValueError(f"k_r[{r}] should be >= b for final re-ranking")
        prev_k = cfg["k_r"][r]

    mode = cfg.get("perturbation_mode", "additive")
    if mode not in {"additive", "prior_mix"}:
        raise ValueError("perturbation_mode must be either 'additive' or 'prior_mix'")
    if mode == "prior_mix":
        if "alpha_0" not in cfg:
            raise ValueError("alpha_0 is required when perturbation_mode='prior_mix'")
        if not (0 < float(cfg["alpha_0"]) < 1):
            raise ValueError("alpha_0 should be in (0, 1)")
    if mode == "additive":
        if "sigma_0" not in cfg:
            raise ValueError("sigma_0 is required when perturbation_mode='additive'")
        if float(cfg["sigma_0"]) <= 0:
            raise ValueError("sigma_0 should be > 0")
