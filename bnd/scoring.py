from __future__ import annotations

from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F


class ClipScorer:
    """CLIP cosine-similarity scorer for image-text alignment."""

    def __init__(self, cache_dir: str | Path):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            cache_dir=str(cache_dir),
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to("cuda").eval()

    @torch.no_grad()
    def score_images(self, images, prompt: str) -> list[float]:
        """Return normalized CLIP image-text similarity scores."""
        if not isinstance(images, list):
            images = [images]

        image_tensors = torch.stack([self.preprocess(img).to("cuda") for img in images])
        text_tokens = self.tokenizer([prompt]).to("cuda")

        image_features = self.model.encode_image(image_tensors)
        text_features = self.model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        scores = image_features @ text_features.T
        return scores.squeeze(-1).detach().float().cpu().tolist()
