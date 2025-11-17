"""Generate quick-look images showing YOLO boxes + class names.

Example:
    python preview_labels.py --split validation --count 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


# University of Michigan brand palette (Maize/Blue plus approved accents).
DEFAULT_COLORS = [
    "#00274C",  # Michigan Blue
    "#FFCB05",  # Michigan Maize
    "#9A3324",  # Tappan Red
    "#6DA34D",  # Ann Arbor Amethyst (green variant)
    "#6F1D77",  # Rackham Purple
    "#00A1DE",  # Wave Blue
]


def parse_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text().strip().splitlines():
        if not line:
            continue
        cls, cx, cy, w, h = line.split()
        boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return boxes


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image file found for {stem} in {images_dir}")


def denorm_box(
    cx: float, cy: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x0 = (cx - w / 2) * width
    y0 = (cy - h / 2) * height
    x1 = (cx + w / 2) * width
    y1 = (cy + h / 2) * height
    return (
        max(0.0, x0),
        max(0.0, y0),
        min(float(width), x1),
        min(float(height), y1),
    )


def draw_boxes(
    image_path: Path,
    labels: Iterable[tuple[int, float, float, float, float]],
    class_names: list[str],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for cls_id, cx, cy, w, h in labels:
        color = DEFAULT_COLORS[cls_id % len(DEFAULT_COLORS)]
        x0, y0, x1, y1 = denorm_box(cx, cy, w, h, width, height)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = class_names[cls_id] if cls_id < len(class_names) else f"id_{cls_id}"
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except AttributeError:  # Pillow < 8 fallback
            text_w, text_h = font.getsize(label)
        rect = [x0, y0 - text_h - 4, x0 + text_w + 6, y0]
        draw.rectangle(rect, fill=color)
        draw.text((x0 + 3, y0 - text_h - 2), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def pick_label_files(labels_dir: Path, count: int, randomize: bool) -> list[Path]:
    files = sorted(labels_dir.glob("*.txt"))
    if randomize:
        random.shuffle(files)
    return files[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "input",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split folder inside --base-dir (default: validation)",
    )
    parser.add_argument("--count", type=int, default=5, help="Images to render.")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick random label files instead of the first N.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["header", "body", "footer"],
        help="Class names in ID order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "label_previews",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    labels_dir = args.base_dir / args.split / "labels"
    images_dir = args.base_dir / args.split / "images"
    output_dir = args.output_dir / args.split

    selected = pick_label_files(labels_dir, args.count, args.random)
    if not selected:
        raise SystemExit(f"No label files found in {labels_dir}")

    for label_path in selected:
        boxes = parse_labels(label_path)
        image_path = find_image(images_dir, label_path.stem)
        output_path = output_dir / f"{label_path.stem}.jpg"
        draw_boxes(image_path, boxes, list(args.names), output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
