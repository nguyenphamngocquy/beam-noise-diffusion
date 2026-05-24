# Beam Noise Diffusion (BND)

This repository contains experiments for Beam Noise Diffusion, a beam-based latent noise search method for text-to-image diffusion models.

## Structure

- `bnd/`: reusable Python package for config validation, sampling, scoring, experiment I/O, and BND logic.
- `configs/`: experiment configurations (`bnd_light.json`, `bnd.json`).
- `prompts/`: prompt sets.
- `notebooks/run_bnd.ipynb`: Colab entrypoint for running the experiments.
- `requirements.txt`: Python dependencies.

Generated artifacts are intentionally not part of the code tree. In Colab, the notebook clones this repository into `/content/beam-noise-diffusion` and stores only runtime artifacts on Google Drive.

## Colab Runtime Layout

- Code/configs/prompts: `/content/beam-noise-diffusion`
- Persistent cache and outputs: `/content/drive/MyDrive/beam-noise-diffusion-storage`
- Model cache: `/content/drive/MyDrive/beam-noise-diffusion-storage/cache/hf_cache`
- Results/images/summaries: `/content/drive/MyDrive/beam-noise-diffusion-storage/outputs`

## Running in Colab

1. Open `notebooks/run_bnd.ipynb` from this repository.
2. Run all cells.
3. Set `CONFIG_NAME` to `bnd_light` for quick checks or T4 runs.
4. Set `CONFIG_NAME` to `bnd` for A100/full experiments.

The notebook installs dependencies from the cloned repository's `requirements.txt`.

## Qualitative Comparison

After running the selected methods, the notebook can build qualitative grids under:

`outputs/summary/<run_name>/qualitative/`

Each grid compares the same prompts across methods. By default, it selects three complete prompts from each prompt group and uses the same method list as the quantitative summary.
