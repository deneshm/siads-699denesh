"""Utility for remapping YOLO label IDs across dataset splits.

The finance parser dataset accidentally labeled headers as footers, footers as
headers, etc.  This script rewrites every label file so that the IDs line up
with the ordering in finance-image-parser.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_mapping_arg(value: str) -> tuple[int, int]:
    """Parse a command line `SRC:DST` mapping option."""
    try:
        src, dst = value.split(":")
        return int(src), int(dst)
    except ValueError as exc:  # pragma: no cover - defensive against bad args
        raise argparse.ArgumentTypeError(
            f"Invalid mapping '{value}'. Expected format SRC:DST (e.g. 2:0)."
        ) from exc


def remap_file(path: Path, mapping: dict[int, int]) -> bool:
    """Rewrite a label file in-place if at least one ID changes."""
    lines = path.read_text().strip().splitlines()
    changed = False
    new_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        src_id = int(parts[0])
        if src_id not in mapping:
            raise ValueError(f"Label {path} contains unknown class id {src_id}.")
        dst_id = mapping[src_id]
        if dst_id != src_id:
            changed = True
        parts[0] = str(dst_id)
        new_lines.append(" ".join(parts))

    if changed:
        path.write_text("\n".join(new_lines) + "\n")
    return changed


def remap_split(base_dir: Path, split: str, mapping: dict[int, int]) -> int:
    """Remap all labels inside `<base_dir>/<split>/labels`."""
    labels_dir = base_dir / split / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    changes = 0
    for label_file in labels_dir.glob("*.txt"):
        if remap_file(label_file, mapping):
            changes += 1
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "input",
        help="Root directory that contains train/val/test splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("training", "validation", "testing"),
        help="Dataset splits to rewrite (default: training validation testing).",
    )
    parser.add_argument(
        "--map",
        dest="mapping",
        action="append",
        type=parse_mapping_arg,
        required=True,
        help="Mapping entries in SRC:DST form (e.g. 2:0). Repeat per class.",
    )
    args = parser.parse_args()

    mapping = dict(args.mapping)
    touched = 0
    for split in args.splits:
        touched += remap_split(args.base_dir, split, mapping)
    print(f"Rewrote {touched} label files.")


if __name__ == "__main__":
    main()
