from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def _read_image_unicode_safe(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def validate_output_dir(output_dir: Path, sample_decode_count: int = 3) -> tuple[bool, str]:
    if not output_dir.exists() or not output_dir.is_dir():
        return False, f"Output directory does not exist: {output_dir}"

    files = sorted([p for p in output_dir.iterdir() if p.suffix.lower() in {".png", ".jpg"}])
    if not files:
        return False, f"No frame image files found in: {output_dir}"

    stem = output_dir.name
    pattern = re.compile(rf"^{re.escape(stem)}_(\d{{6}})\.(png|jpg)$", re.IGNORECASE)
    indexes: list[int] = []
    for f in files:
        m = pattern.match(f.name)
        if not m:
            return False, f"Invalid filename format: {f.name}"
        indexes.append(int(m.group(1)))

    expected = list(range(len(files)))
    if indexes != expected:
        return False, f"Frame indexes are not contiguous from 000000: {indexes[:5]} ... {indexes[-5:]}"

    decode_targets = files[: min(sample_decode_count, len(files))]
    for f in decode_targets:
        img = _read_image_unicode_safe(f)
        if img is None:
            return False, f"Failed to decode frame image: {f}"

    return True, f"Validated {len(files)} frames in {output_dir}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate extracted frame directory.")
    parser.add_argument("--output-dir", required=True, help="Path to one extracted video output directory.")
    parser.add_argument(
        "--sample-decode-count",
        type=int,
        default=3,
        help="Number of first frames to decode for integrity check.",
    )
    args = parser.parse_args()

    ok, message = validate_output_dir(Path(args.output_dir), sample_decode_count=max(1, args.sample_decode_count))
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
