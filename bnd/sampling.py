from __future__ import annotations

from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def get_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def load_sd_pipeline(cfg: dict, cache_dir: str | Path):
    """Load the text-to-image diffusion pipeline used by all methods."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this experiment")

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=str(cache_dir),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    # Attention slicing trades speed for memory and is useful on T4-sized GPUs.
    if "T4" in get_gpu_name():
        pipe.enable_attention_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled.")
    except Exception as exc:
        print("xFormers not enabled:", str(exc)[:120])

    return pipe


@torch.no_grad()
def sample_images_from_latents(pipe, prompt: str, latents, cfg: dict, num_steps: int):
    """Run diffusion from explicit initial latents."""
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    batch_size = latents.shape[0]
    negative_prompt = cfg.get("negative_prompt", "")

    return pipe(
        prompt=[prompt] * batch_size,
        negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
        latents=latents,
        height=cfg["resolution"],
        width=cfg["resolution"],
        num_inference_steps=num_steps,
        guidance_scale=cfg["guidance_scale"],
        output_type="pil",
    ).images


def make_latents(batch_size: int, seed: int, cfg: dict):
    """Create standard Gaussian initial latents for Stable Diffusion."""
    latent_size = cfg["resolution"] // 8
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    return torch.randn(
        (batch_size, 4, latent_size, latent_size),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )


def make_proposals(base_latents, m: int, round_idx: int, cfg: dict, generator):
    """Create local proposal latents around the active beam."""
    proposals = []
    mode = cfg.get("perturbation_mode", "additive")

    if mode == "prior_mix":
        # prior_mix keeps proposals closer to the N(0, I) initial-noise prior.
        alpha = cfg["alpha_0"] * (cfg["gamma"] ** round_idx)
        alpha = max(0.0, min(float(alpha), 0.999))
        keep = (1.0 - alpha**2) ** 0.5
    else:
        sigma = cfg["sigma_0"] * (cfg["gamma"] ** round_idx)

    for i in range(base_latents.shape[0]):
        base = base_latents[i : i + 1]
        noise = torch.randn(
            (m, *base.shape[1:]),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        base_rep = base.repeat(m, 1, 1, 1)
        z_prop = keep * base_rep + alpha * noise if mode == "prior_mix" else base_rep + sigma * noise
        proposals.append(z_prop)

    return torch.cat(proposals, dim=0)
