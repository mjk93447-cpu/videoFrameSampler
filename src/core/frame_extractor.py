from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from core.models import ExtractionOptions, ImageFormat, VideoExtractionResult

ProgressCallback = Callable[[str], None]


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def make_unique_output_dir(video_stem: str, root_output: Path) -> Path:
    root_output.mkdir(parents=True, exist_ok=True)
    candidate = root_output / video_stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = 2
    while True:
        candidate = root_output / f"{video_stem}__{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def suggest_fast_mode_interval(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 5
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080.0
    cap.release()

    megapixels = (width * height) / 1_000_000
    if megapixels >= 6 or fps >= 50:
        return 5
    if megapixels >= 2 or fps >= 30:
        return 3
    return 2


def _save_frame(
    frame,
    out_path: Path,
    image_format: ImageFormat,
    jpg_quality: int,
) -> bool:
    if image_format == ImageFormat.JPG:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    else:
        ext = ".png"
        params = []

    ok, encoded = cv2.imencode(ext, frame, params)
    if not ok:
        return False

    try:
        # tofile handles Windows Unicode paths more reliably than imwrite.
        encoded_array: np.ndarray = encoded
        encoded_array.tofile(str(out_path))
        return True
    except OSError:
        return False


def extract_video_frames(
    video_path: Path,
    options: ExtractionOptions,
    progress_cb: ProgressCallback | None = None,
) -> VideoExtractionResult:
    video_path = Path(video_path)
    base_dir = get_runtime_base_dir()
    output_root = base_dir / "output"
    output_dir = make_unique_output_dir(video_path.stem, output_root)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="Failed to open video file.",
        )

    interval = options.normalized_interval()
    image_format = options.image_format
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"
    jpg_quality = options.normalized_jpg_quality()

    frame_index = 0
    saved_count = 0
    failure_message = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % interval == 0:
            filename = f"{video_path.stem}_{saved_count:06d}{ext}"
            out_path = output_dir / filename
            written = _save_frame(frame, out_path, image_format, jpg_quality)
            if not written:
                failure_message = f"Failed to save frame: {out_path}"
                break
            saved_count += 1
            if progress_cb and saved_count % 25 == 0:
                progress_cb(f"Saved {saved_count} frames from {video_path.name}")

        frame_index += 1

    cap.release()

    if failure_message:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=saved_count,
            total_frames_seen=frame_index,
            success=False,
            message=failure_message,
        )

    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=saved_count,
        total_frames_seen=frame_index,
        success=True,
        message="Completed.",
    )


def extract_videos_sequentially(
    video_paths: Iterable[Path],
    options: ExtractionOptions,
    progress_cb: ProgressCallback | None = None,
) -> list[VideoExtractionResult]:
    results: list[VideoExtractionResult] = []
    for idx, video_path in enumerate(video_paths, start=1):
        if progress_cb:
            progress_cb(f"[{idx}] Processing {Path(video_path).name}")
        try:
            result = extract_video_frames(Path(video_path), options, progress_cb=progress_cb)
        except Exception as exc:  # Defensive catch for file/permission/corruption issues
            result = VideoExtractionResult(
                video_path=Path(video_path),
                output_dir=get_runtime_base_dir() / "output",
                saved_count=0,
                total_frames_seen=0,
                success=False,
                message=f"Unhandled extraction error: {exc}",
            )
        results.append(result)
    return results
