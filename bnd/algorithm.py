from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch

from .io import already_done, image_path, result_path, save_json
from .sampling import get_gpu_name, make_latents, make_proposals, sample_images_from_latents


def top_k_candidates(candidates: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Keep the highest-scoring candidates."""
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:k]


def candidates_to_latents(candidates: list[dict[str, Any]]):
    """Move stored CPU latents back to GPU for another diffusion pass."""
    return torch.stack([c["latent"] for c in candidates]).to("cuda", dtype=torch.float16)


@torch.no_grad()
def score_latents(pipe, scorer, prompt: str, latents, cfg: dict, num_steps: int, batch_size: int | None = None):
    """
    ScoreLatents from the BND pseudocode.

    It samples preview/mid images from the provided latents and stores only
    (latent, score). The generated preview/mid images are discarded to save memory.
    """
    results = []
    batch_size = batch_size or cfg["eval_batch_size"]

    for start in range(0, latents.shape[0], batch_size):
        sub_latents = latents[start : start + batch_size]
        images = sample_images_from_latents(pipe, prompt, sub_latents, cfg, num_steps)
        scores = scorer.score_images(images, prompt)

        for i, score in enumerate(scores):
            results.append({"latent": sub_latents[i].detach().cpu(), "score": float(score)})

        del sub_latents, images

    return results


def deduplicate_candidates(candidates: list[dict[str, Any]], atol: float = 1e-6) -> list[dict[str, Any]]:
    """Remove exact or near-exact duplicate latent candidates."""
    unique = []

    for cand in candidates:
        z = cand["latent"]
        if not any(torch.allclose(z, u["latent"], atol=atol, rtol=0) for u in unique):
            unique.append(cand)

    return unique


def _round_cfg(cfg: dict, rounds: int) -> dict:
    """Return a copy of cfg using only the first `rounds` refinement rounds."""
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if rounds > len(cfg["a_r"]):
        raise ValueError(f"rounds={rounds} exceeds configured rounds={len(cfg['a_r'])}")

    out = copy.deepcopy(cfg)
    out["R"] = rounds
    for key in ["a_r", "m_r", "q_r", "k_r"]:
        out[key] = out[key][:rounds]
    return out


def run_random_baseline(prompt_item: dict, cfg: dict, pipe, scorer, root: str | Path, run_name: str):
    """Generate one image from one random latent and score it."""
    method = "random"
    prompt_id = prompt_item["id"]
    prompt = prompt_item["prompt"]

    if already_done(root, run_name, method, prompt_id, cfg):
        print(f"[SKIP] {method} {prompt_id}")
        return

    start_time = time.time()
    seed = cfg["seed_base"] + int(prompt_id)
    latents = make_latents(batch_size=1, seed=seed, cfg=cfg)

    images = sample_images_from_latents(pipe, prompt, latents, cfg, cfg["T_full"])
    score = scorer.score_images(images, prompt)[0]

    out_img = image_path(root, run_name, method, prompt_id)
    images[0].save(out_img)

    result = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "group": prompt_item.get("group"),
        "method": method,
        "score": float(score),
        "runtime_sec": time.time() - start_time,
        "gpu": get_gpu_name(),
        "image_path": str(out_img),
        "config": cfg,
    }
    save_json(result_path(root, run_name, method, prompt_id), result)
    print(f"[DONE] {method} {prompt_id} score={score:.4f}")


def run_best_of_n(prompt_item: dict, cfg: dict, pipe, scorer, root: str | Path, run_name: str, n: int):
    """Best-of-N baseline: full-sample N independent latents and keep the best CLIP score."""
    method = f"best_of_{n}"
    prompt_id = prompt_item["id"]
    prompt = prompt_item["prompt"]

    if already_done(root, run_name, method, prompt_id, cfg):
        print(f"[SKIP] {method} {prompt_id}")
        return

    start_time = time.time()
    seed_base = cfg["seed_base"] + int(prompt_id) * 1000 + 50000
    all_scores = []
    all_seeds = []
    best_image = None
    best_score = None
    best_seed = None

    for i in range(n):
        seed = seed_base + i
        latents = make_latents(batch_size=1, seed=seed, cfg=cfg)
        images = sample_images_from_latents(pipe, prompt, latents, cfg, cfg["T_full"])
        score = float(scorer.score_images(images, prompt)[0])

        all_scores.append(score)
        all_seeds.append(seed)

        if best_score is None or score > best_score:
            best_score = score
            best_seed = seed
            best_image = images[0]

        del latents, images

    out_img = image_path(root, run_name, method, prompt_id)
    best_image.save(out_img)

    result = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "group": prompt_item.get("group"),
        "method": method,
        "n": int(n),
        "score": float(best_score),
        "all_scores": all_scores,
        "best_seed": int(best_seed),
        "runtime_sec": time.time() - start_time,
        "gpu": get_gpu_name(),
        "image_path": str(out_img),
        "config": cfg,
    }
    save_json(result_path(root, run_name, method, prompt_id), result)
    print(f"[DONE] {method} {prompt_id} score={best_score:.4f}")


def run_monte_carlo(prompt_item: dict, cfg: dict, pipe, scorer, root: str | Path, run_name: str):
    """Monte-Carlo baseline: search random initial latents without refinement."""
    method = "monte_carlo"
    prompt_id = prompt_item["id"]
    prompt = prompt_item["prompt"]

    if already_done(root, run_name, method, prompt_id, cfg):
        print(f"[SKIP] {method} {prompt_id}")
        return

    start_time = time.time()
    candidates = []
    seed_base = cfg["seed_base"] + int(prompt_id) * 1000

    for u in range(cfg["B_init"]):
        z = make_latents(cfg["n_batch"], seed=seed_base + u, cfg=cfg)
        scored = score_latents(pipe, scorer, prompt, z, cfg, cfg["T_preview"])
        candidates = top_k_candidates(candidates + scored, cfg["k1"])

    z0 = candidates_to_latents(candidates)
    medium = score_latents(pipe, scorer, prompt, z0, cfg, cfg["T_mid"])
    medium = top_k_candidates(medium, cfg["k1"])

    final_candidates = top_k_candidates(medium, cfg["b"])
    z_final = candidates_to_latents(final_candidates)

    images = sample_images_from_latents(pipe, prompt, z_final, cfg, cfg["T_full"])
    scores = scorer.score_images(images, prompt)
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))

    out_img = image_path(root, run_name, method, prompt_id)
    images[best_idx].save(out_img)

    result = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "group": prompt_item.get("group"),
        "method": method,
        "score": float(scores[best_idx]),
        "all_final_scores": [float(s) for s in scores],
        "runtime_sec": time.time() - start_time,
        "gpu": get_gpu_name(),
        "image_path": str(out_img),
        "config": cfg,
    }
    save_json(result_path(root, run_name, method, prompt_id), result)
    print(f"[DONE] {method} {prompt_id} score={scores[best_idx]:.4f}")


def _run_bnd_core(
    prompt_item: dict,
    cfg: dict,
    pipe,
    scorer,
    root: str | Path,
    run_name: str,
    method: str,
    full_rerank: bool,
):
    """Shared BND implementation used by the main method and ablation variants."""
    prompt_id = prompt_item["id"]
    prompt = prompt_item["prompt"]

    if already_done(root, run_name, method, prompt_id, cfg):
        print(f"[SKIP] {method} {prompt_id}")
        return

    start_time = time.time()
    seed_base = cfg["seed_base"] + int(prompt_id) * 1000
    candidates = []

    # Stage 1: batched Monte-Carlo initialization using cheap preview scores.
    for u in range(cfg["B_init"]):
        z = make_latents(cfg["n_batch"], seed=seed_base + u, cfg=cfg)
        scored = score_latents(pipe, scorer, prompt, z, cfg, cfg["T_preview"])
        candidates = top_k_candidates(candidates + scored, cfg["k1"])

    # Convert preview winners into the main beam using mid-level scores only.
    z0 = candidates_to_latents(candidates)
    beam = score_latents(pipe, scorer, prompt, z0, cfg, cfg["T_mid"])
    beam = top_k_candidates(beam, cfg["k1"])
    anchor_beam = top_k_candidates(beam, cfg["b"])

    s_best = max(c["score"] for c in beam)
    c_stop = 0
    stopped_round = cfg["R"]
    refinement_trace = []

    # Stage 2: beam-based latent refinement. Skipped automatically when R = 0.
    for r in range(cfg["R"]):
        active = top_k_candidates(beam, cfg["a_r"][r])
        active_latents = candidates_to_latents(active)

        gen = torch.Generator(device="cuda").manual_seed(seed_base + 10000 + r)
        proposals = make_proposals(active_latents, cfg["m_r"][r], r, cfg, gen)

        # Preview scores only filter which proposals receive the mid-level budget.
        prev_scored = score_latents(pipe, scorer, prompt, proposals, cfg, cfg["T_preview"])
        q_prev = top_k_candidates(prev_scored, cfg["q_r"][r])

        # The persistent beam is updated only with mid-level scores.
        z_q = candidates_to_latents(q_prev)
        mid_scored = score_latents(pipe, scorer, prompt, z_q, cfg, cfg["T_mid"])
        beam = top_k_candidates(beam + mid_scored, cfg["k_r"][r])

        prev_best = max(c["score"] for c in q_prev)
        mid_best = max(c["score"] for c in mid_scored)
        s_r_max = max(c["score"] for c in beam)
        improved = s_r_max > s_best + cfg["epsilon"]

        refinement_trace.append(
            {
                "round": r + 1,
                "num_active": len(active),
                "num_proposals": int(proposals.shape[0]),
                "preview_best": float(prev_best),
                "mid_best": float(mid_best),
                "beam_best": float(s_r_max),
                "improved": bool(improved),
            }
        )

        if improved:
            s_best = s_r_max
            c_stop = 0
        else:
            c_stop += 1

        if c_stop >= cfg["patience"]:
            stopped_round = r + 1
            break

    final_pool = deduplicate_candidates(beam + anchor_beam)

    if full_rerank:
        # Full sampling and final re-ranking over exactly b final candidates.
        final_candidates = top_k_candidates(final_pool, cfg["b"])
        z_final = candidates_to_latents(final_candidates)
        images = sample_images_from_latents(pipe, prompt, z_final, cfg, cfg["T_full"])
        scores = scorer.score_images(images, prompt)
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        out_img = image_path(root, run_name, method, prompt_id)
        images[best_idx].save(out_img)
        final_score = float(scores[best_idx])
        all_scores = [float(s) for s in scores]
    else:
        # No full rerank: choose the best mid-score latent and save a T_mid image.
        final_candidates = top_k_candidates(final_pool, 1)
        z_final = candidates_to_latents(final_candidates)
        images = sample_images_from_latents(pipe, prompt, z_final, cfg, cfg["T_mid"])
        scores = scorer.score_images(images, prompt)
        out_img = image_path(root, run_name, method, prompt_id)
        images[0].save(out_img)
        final_score = float(scores[0])
        all_scores = [float(scores[0])]

    result = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "group": prompt_item.get("group"),
        "method": method,
        "score": final_score,
        "all_final_scores": all_scores,
        "final_mid_scores": [float(c["score"]) for c in final_candidates],
        "full_rerank": bool(full_rerank),
        "num_final_candidates": len(final_candidates),
        "num_anchor_candidates": len(anchor_beam),
        "best_mid_score": float(s_best),
        "stopped_round": int(stopped_round),
        "refinement_rounds": int(cfg["R"]),
        "refinement_trace": refinement_trace,
        "runtime_sec": time.time() - start_time,
        "gpu": get_gpu_name(),
        "image_path": str(out_img),
        "config": cfg,
    }
    save_json(result_path(root, run_name, method, prompt_id), result)
    print(f"[DONE] {method} {prompt_id} score={final_score:.4f}")


def run_bnd(prompt_item: dict, cfg: dict, pipe, scorer, root: str | Path, run_name: str):
    """Run the main BND method with refinement and full re-ranking."""
    return _run_bnd_core(prompt_item, cfg, pipe, scorer, root, run_name, method="bnd", full_rerank=True)


def run_bnd_no_full_rerank(prompt_item: dict, cfg: dict, pipe, scorer, root: str | Path, run_name: str):
    """Ablation: run BND but select the final output from the medium-score beam."""
    return _run_bnd_core(
        prompt_item,
        cfg,
        pipe,
        scorer,
        root,
        run_name,
        method="bnd_no_full_rerank",
        full_rerank=False,
    )


def run_bnd_rounds(prompt_item: dict, cfg: dict, pipe, scorer, root: str | Path, run_name: str, rounds: int):
    """Ablation: run BND with a specified number of refinement rounds."""
    round_cfg = _round_cfg(cfg, rounds)
    return _run_bnd_core(
        prompt_item,
        round_cfg,
        pipe,
        scorer,
        root,
        run_name,
        method=f"bnd_R{rounds}",
        full_rerank=True,
    )
