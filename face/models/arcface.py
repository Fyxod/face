"""ArcFace iResNet-100 loading, preprocessing, and embedding utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .iresnet import iresnet100, model_summary


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict", "backbone", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload is {type(payload)!r}, not a state dict.")
    state = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        new_key = str(key)
        for prefix in ("module.", "backbone.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        state[new_key] = value
    if not state:
        raise ValueError("No tensor state_dict entries found in checkpoint.")
    return state


class ArcFaceIResNet100(torch.nn.Module):
    """Frozen iResNet-100 ArcFace embedding model.

    The differentiable preprocessing path is:
    RGB `[0, 1]` tensor -> bilinear resize 112x112 -> `[-1, 1]`
    normalization -> frozen iResNet-100 -> L2-normalized 512-D embedding.
    """

    input_size = 112
    architecture = "iresnet100"
    normalization = "minus_one_to_one"
    channel_order = "rgb"
    embedding_normalization = "l2"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        source_url: str | None = None,
        fp16: bool = False,
        strict_ratio: float = 0.95,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing ArcFace iResNet-100 checkpoint: {self.checkpoint_path}")
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.checkpoint_source = source_url or "user_provided_local_checkpoint"
        self.model = iresnet100(fp16=fp16).to(device)
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        state = _extract_state_dict(payload)
        model_keys = set(self.model.state_dict().keys())
        matching = [key for key in state if key in model_keys and tuple(state[key].shape) == tuple(self.model.state_dict()[key].shape)]
        loadable = {key: state[key] for key in matching}
        ratio = len(matching) / max(len(model_keys), 1)
        if ratio < strict_ratio:
            raise RuntimeError(
                f"Checkpoint does not look like iResNet-100: matched {len(matching)}/{len(model_keys)} "
                f"state entries ({ratio:.1%}), required {strict_ratio:.0%}."
            )
        missing, unexpected = self.model.load_state_dict(loadable, strict=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(device)
        self.device = device
        self.embedding_dimension = 512
        self.load_report = {
            "arcface_architecture": self.architecture,
            "arcface_checkpoint_path": str(self.checkpoint_path),
            "arcface_checkpoint_filename": self.checkpoint_path.name,
            "arcface_checkpoint_sha256": self.checkpoint_sha256,
            "arcface_checkpoint_source": self.checkpoint_source,
            "arcface_input_size": self.input_size,
            "arcface_embedding_dimension": self.embedding_dimension,
            "arcface_normalization": self.normalization,
            "arcface_channel_order": self.channel_order,
            "arcface_embedding_normalization": self.embedding_normalization,
            "state_dict_match_ratio": ratio,
            "loaded_state_entries": len(loadable),
            "missing_state_entries": list(missing),
            "unexpected_state_entries": list(unexpected),
            **model_summary(self.model),
        }

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float().clamp(0, 1)
        resized = F.interpolate(image, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return resized * 2.0 - 1.0

    def embedding(self, image: torch.Tensor) -> torch.Tensor:
        raw = self.model(self.preprocess(image))
        return F.normalize(raw.float(), p=2, dim=1)

    def metadata(self) -> dict[str, Any]:
        return dict(self.load_report)


def write_setup_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
