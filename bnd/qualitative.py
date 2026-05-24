from __future__ import annotations

from pathlib import Path
from textwrap import wrap
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .io import load_json


try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def _result_file(storage_root: str | Path, run_name: str, method: str, prompt_id: str) -> Path:
    return Path(storage_root) / "outputs" / "results" / run_name / method / f"{prompt_id}.json"


def _canonical_image_file(storage_root: str | Path, run_name: str, method: str, prompt_id: str) -> Path:
    return Path(storage_root) / "outputs" / "images" / run_name / method / f"{prompt_id}_best.png"


def _load_result(storage_root: str | Path, run_name: str, method: str, prompt_id: str) -> dict[str, Any] | None:
    result_file = _result_file(storage_root, run_name, method, prompt_id)
    if not result_file.exists():
        return None
    return load_json(result_file)


def _resolve_image_path(
    storage_root: str | Path,
    run_name: str,
    method: str,
    prompt_id: str,
    result: dict[str, Any],
) -> Path | None:
    candidates = [_canonical_image_file(storage_root, run_name, method, prompt_id)]
    if result.get("image_path"):
        candidates.append(Path(result["image_path"]))

    for path in candidates:
        if path.exists():
            return path
    return None


def _method_label(method: str) -> str:
    known = {
        "random": "Random",
        "monte_carlo": "Monte-Carlo",
        "bnd": "BND",
        "bnd_no_full_rerank": "BND no full rerank",
    }
    if method in known:
        return known[method]
    if method.startswith("best_of_"):
        return "Best-of-" + method.rsplit("_", maxsplit=1)[-1]
    if method.startswith("bnd_R"):
        return "BND R=" + method.replace("bnd_R", "")
    return method.replace("_", " ").title()


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill, max_chars: int, line_gap: int = 4):
    x, y = xy
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + line_gap
    for line in wrap(text, width=max_chars):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _has_complete_outputs(
    prompt_item: dict,
    storage_root: str | Path,
    run_name: str,
    methods: list[str],
) -> bool:
    prompt_id = prompt_item["id"]
    for method in methods:
        result = _load_result(storage_root, run_name, method, prompt_id)
        if result is None:
            return False
        if _resolve_image_path(storage_root, run_name, method, prompt_id, result) is None:
            return False
    return True


def select_qualitative_prompts(
    prompts: list[dict],
    storage_root: str | Path,
    run_name: str,
    methods: list[str],
    prompts_per_group: int = 3,
    groups: list[str] | None = None,
) -> list[dict]:
    """
    Select the first complete prompts from each group.

    A prompt is complete only when every requested method has both a result JSON
    and a saved output image. This keeps the qualitative grid visually balanced.
    """
    group_order = groups or list(dict.fromkeys(item.get("group", "unknown") for item in prompts))
    selected = []

    for group in group_order:
        group_items = [item for item in prompts if item.get("group", "unknown") == group]
        complete = [
            item
            for item in group_items
            if _has_complete_outputs(item, storage_root, run_name, methods)
        ]
        if len(complete) < prompts_per_group:
            raise ValueError(
                f"Group '{group}' has only {len(complete)} complete prompts for methods {methods}; "
                f"need {prompts_per_group}. Run the missing experiments first or reduce prompts_per_group."
            )
        selected.extend(complete[:prompts_per_group])

    return selected


def _render_group_grid(
    rows: list[dict],
    methods: list[str],
    storage_root: str | Path,
    run_name: str,
    out_path: str | Path,
    image_size: int,
) -> pd.DataFrame:
    label_w = 390
    header_h = 54
    score_h = 32
    cell_w = image_size
    cell_h = image_size + score_h
    pad = 12
    line = (218, 224, 231)
    text = (32, 38, 46)
    muted = (90, 99, 112)

    width = label_w + len(methods) * cell_w
    height = header_h + len(rows) * cell_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(15, bold=True)
    text_font = _font(13)
    small_font = _font(12)

    draw.rectangle((0, 0, width - 1, height - 1), outline=line)
    draw.rectangle((0, 0, width, header_h), fill=(248, 250, 252), outline=line)
    draw.text((pad, 17), "Prompt", font=title_font, fill=text)
    for col, method in enumerate(methods):
        x = label_w + col * cell_w
        draw.rectangle((x, 0, x + cell_w, header_h), outline=line)
        header = _method_label(method)
        bbox = draw.textbbox((0, 0), header, font=title_font)
        draw.text((x + (cell_w - (bbox[2] - bbox[0])) // 2, 17), header, font=title_font, fill=text)

    manifest_rows = []
    for row_idx, item in enumerate(rows):
        y = header_h + row_idx * cell_h
        prompt_id = item["id"]
        prompt = item["prompt"]
        group = item.get("group")

        draw.rectangle((0, y, label_w, y + cell_h), fill=(255, 255, 255), outline=line)
        draw.text((pad, y + pad), f"{prompt_id} | {group}", font=title_font, fill=text)
        _draw_wrapped(draw, (pad, y + pad + 24), prompt, text_font, muted, max_chars=42)

        for col, method in enumerate(methods):
            x = label_w + col * cell_w
            result = _load_result(storage_root, run_name, method, prompt_id)
            if result is None:
                raise FileNotFoundError(_result_file(storage_root, run_name, method, prompt_id))
            image_file = _resolve_image_path(storage_root, run_name, method, prompt_id, result)
            if image_file is None:
                raise FileNotFoundError(_canonical_image_file(storage_root, run_name, method, prompt_id))
            score = float(result["score"])

            draw.rectangle((x, y, x + cell_w, y + cell_h), outline=line)
            image = Image.open(image_file).convert("RGB")
            image.thumbnail((image_size, image_size), RESAMPLE_LANCZOS)
            img_x = x + (cell_w - image.width) // 2
            img_y = y + (image_size - image.height) // 2
            canvas.paste(image, (img_x, img_y))

            score_text = f"CLIPScore: {score:.4f}"
            bbox = draw.textbbox((0, 0), score_text, font=small_font)
            draw.rectangle((x, y + image_size, x + cell_w, y + cell_h), fill=(248, 250, 252), outline=line)
            draw.text(
                (x + (cell_w - (bbox[2] - bbox[0])) // 2, y + image_size + 8),
                score_text,
                font=small_font,
                fill=muted,
            )

            manifest_rows.append(
                {
                    "group": group,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "method": method,
                    "score": score,
                    "image_path": str(image_file),
                    "grid_path": str(out_path),
                }
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return pd.DataFrame(manifest_rows)


def build_qualitative_comparisons(
    prompts: list[dict],
    storage_root: str | Path,
    run_name: str,
    methods: list[str],
    prompts_per_group: int = 3,
    groups: list[str] | None = None,
    image_size: int = 256,
):
    """
    Build qualitative comparison grids: one group, three prompts, multiple methods.

    The output files are saved under outputs/summary/<run_name>/qualitative.
    """
    storage_root = Path(storage_root)
    out_dir = storage_root / "outputs" / "summary" / run_name / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_qualitative_prompts(
        prompts=prompts,
        storage_root=storage_root,
        run_name=run_name,
        methods=methods,
        prompts_per_group=prompts_per_group,
        groups=groups,
    )

    manifest_parts = []
    grid_paths = {}
    for group in dict.fromkeys(item.get("group", "unknown") for item in selected):
        group_rows = [item for item in selected if item.get("group", "unknown") == group]
        grid_path = out_dir / f"qualitative_{group}.png"
        manifest_parts.append(
            _render_group_grid(
                rows=group_rows,
                methods=methods,
                storage_root=storage_root,
                run_name=run_name,
                out_path=grid_path,
                image_size=image_size,
            )
        )
        grid_paths[group] = grid_path

    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest.to_csv(out_dir / "qualitative_manifest.csv", index=False)
    return manifest, grid_paths
