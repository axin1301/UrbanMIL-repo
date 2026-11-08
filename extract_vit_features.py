"""Extract ViT image features in the format expected by UrbanMIL.

Input rows follow the CityLens task JSON order. Each row contains one
satellite image followed by ten street view images in ``images``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch
from PIL import Image


def image_path(image_root: Path, source_path: str) -> Path:
    # Task JSONs can contain absolute paths from a different machine.
    name = str(source_path).replace("\\", "/").rsplit("/", 1)[-1]
    if not name:
        raise ValueError(f"Invalid image path: {source_path!r}")
    path = image_root / name
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path} (source: {source_path})")
    return path


@torch.inference_mode()
def encode_paths(model, transform, paths: list[Path], batch_size: int,
                 device: torch.device) -> torch.Tensor:
    features = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
        output = model.forward_features(torch.stack(batch).to(device))
        if output.ndim == 3:
            output = output[:, 0, :]
        if output.ndim != 2:
            raise ValueError(f"Unexpected ViT output shape: {tuple(output.shape)}")
        features.append(output.float().cpu())
    return torch.cat(features, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_json", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if not args.task_json.is_file() or not args.image_root.is_dir():
        parser.error("--task_json and --image_root must exist")

    rows = json.loads(args.task_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Task JSON must be a nonempty list")
    ids: list[str] = []
    paths: list[Path] = []
    for index, row in enumerate(rows):
        rid = str(row.get("region_id", f"sample_{index:06d}"))
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 11:
            raise ValueError(f"{rid}: expected one satellite and ten street view images")
        ids.append(rid)
        paths.extend(image_path(args.image_root, value) for value in images)
    if len(ids) != len(set(ids)):
        raise ValueError("Task JSON contains duplicate region IDs")

    device = torch.device(args.device)
    model = timm.create_model(args.model_name, pretrained=True).eval().to(device)
    config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**config, is_training=False)
    encoded = encode_paths(model, transform, paths, args.batch_size, device)
    if encoded.shape[0] != len(ids) * 11 or not torch.isfinite(encoded).all():
        raise ValueError("Feature extraction produced missing or nonfinite vectors")
    encoded = encoded.reshape(len(ids), 11, -1)
    records = {
        rid: {"sat": encoded[i, 0].clone(),
              "street": encoded[i, 1:].clone(),
              "street_mask": torch.ones(10, dtype=torch.bool)}
        for i, rid in enumerate(ids)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, args.output)
    print(f"Saved {len(records)} regions, feature dimension {encoded.shape[-1]}: {args.output}")


if __name__ == "__main__":
    main()
