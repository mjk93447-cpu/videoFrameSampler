from __future__ import annotations

import importlib
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from core.models import ExtractionOptions, ImageFormat, MotionSamplingOptions, RoiBox, VideoExtractionResult

ProgressCallback = Callable[[str], None]

FFMPEG_TOLERANT_INPUT_PARAMS: list[str] = [
    "-err_detect",
    "ignore_err",
    "-fflags",
    "+genpts+igndts+discardcorrupt",
    "-analyzeduration",
    "200M",
    "-probesize",
    "200M",
]

FFMPEG_RELAXED_INPUT_PARAMS: list[str] = [
    "-err_detect",
    "ignore_err",
    "-fflags",
    "+genpts+igndts+ignidx",
    "-analyzeduration",
    "400M",
    "-probesize",
    "400M",
]

FFMPEG_FORCED_INPUT_FORMATS: tuple[str, ...] = (
    "avi",
    "asf",
    "mpegts",
    "mov",
    "matroska",
    "h264",
)

CV2_BACKEND_ORDER: tuple[str, ...] = (
    "default",
    # Prefer legacy Windows system decoders before OpenCV ffmpeg backend.
    "CAP_DSHOW",
    "CAP_MSMF",
    "CAP_FFMPEG",
)


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


def load_first_frame_preview(video_path: Path) -> np.ndarray | None:
    candidates: list[int | None] = [None]
    for backend_name in CV2_BACKEND_ORDER:
        if backend_name == "default":
            continue
        backend_id = getattr(cv2, backend_name, None)
        if isinstance(backend_id, int):
            candidates.append(backend_id)

    seen: set[int | None] = set()
    for backend_id in candidates:
        if backend_id in seen:
            continue
        seen.add(backend_id)
        cap = cv2.VideoCapture(str(video_path)) if backend_id is None else cv2.VideoCapture(str(video_path), backend_id)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame
    return None


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


def _clamp_roi(frame_shape: tuple[int, ...], roi: RoiBox | None) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    normalized = roi.normalized()
    h, w = frame_shape[:2]
    x = min(max(0, normalized.x), max(0, w - 1))
    y = min(max(0, normalized.y), max(0, h - 1))
    x2 = min(w, x + normalized.width)
    y2 = min(h, y + normalized.height)
    if x2 <= x or y2 <= y:
        return None
    return x, y, x2, y2


def _apply_roi(frame: np.ndarray, roi: RoiBox | None) -> np.ndarray:
    bounds = _clamp_roi(frame.shape, roi)
    if bounds is None:
        return frame
    x, y, x2, y2 = bounds
    return frame[y:y2, x:x2]


def _find_major_motion_window(motion_scores: list[float], fps: float, expected_duration_sec: float) -> tuple[int, int]:
    if not motion_scores:
        return 0, 0
    approx_len = max(1, int(round(expected_duration_sec * fps)))
    min_len = max(1, int(round(approx_len * 0.9)))
    max_len = max(min_len, int(round(approx_len * 1.1)))

    prefix = [0.0]
    for score in motion_scores:
        prefix.append(prefix[-1] + score)

    best_start = 0
    best_len = min_len
    best_sum = -1.0
    total = len(motion_scores)
    for length in range(min_len, min(max_len, total) + 1):
        for start in range(0, total - length + 1):
            current = prefix[start + length] - prefix[start]
            if current > best_sum:
                best_sum = current
                best_start = start
                best_len = length
    return best_start, min(total - 1, best_start + best_len - 1)

def _extract_with_cv2(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
    roi: RoiBox | None,
    motion_sampling: MotionSamplingOptions | None,
    progress_cb: ProgressCallback | None,
) -> VideoExtractionResult | None:
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"

    backend_candidates: list[tuple[str, int | None]] = [("default", None)]
    for backend_name in CV2_BACKEND_ORDER:
        if backend_name == "default":
            continue
        backend_id = getattr(cv2, backend_name, None)
        if isinstance(backend_id, int):
            backend_candidates.append((backend_name.lower(), backend_id))

    seen_backend_ids: set[int | None] = set()
    unique_candidates: list[tuple[str, int | None]] = []
    for name, backend_id in backend_candidates:
        if backend_id in seen_backend_ids:
            continue
        seen_backend_ids.add(backend_id)
        unique_candidates.append((name, backend_id))

    for backend_name, backend_id in unique_candidates:
        if progress_cb:
            progress_cb(f"[cv2] Trying backend: {backend_name}")
        cap = cv2.VideoCapture(str(video_path)) if backend_id is None else cv2.VideoCapture(str(video_path), backend_id)
        if not cap.isOpened():
            cap.release()
            continue

        frame_index = 0
        saved_count = 0
        failure_message = ""

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % interval == 0:
                frame_to_save = _apply_roi(frame, roi)
                filename = f"{video_path.stem}_{saved_count:06d}{ext}"
                out_path = output_dir / filename
                written = _save_frame(frame_to_save, out_path, image_format, jpg_quality)
                if not written:
                    failure_message = f"Failed to save frame: {out_path}"
                    break
                saved_count += 1
                if progress_cb and saved_count % 25 == 0:
                    progress_cb(f"Saved {saved_count} frames from {video_path.name} ({backend_name})")

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

        if saved_count == 0 and frame_index == 0:
            continue

        message = f"Completed with OpenCV backend: {backend_name}."
        if motion_sampling and motion_sampling.enabled:
            motion_result = _extract_motion_segment_with_cv2(
                video_path=video_path,
                output_dir=output_dir,
                image_format=image_format,
                jpg_quality=jpg_quality,
                roi=roi,
                motion_sampling=motion_sampling,
                backend_id=backend_id,
                progress_cb=progress_cb,
            )
            message = f"{message} {motion_result}"

        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=saved_count,
            total_frames_seen=frame_index,
            success=True,
            message=message,
        )

    return None


def _extract_with_imageio_fallback(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
    roi: RoiBox | None,
    progress_cb: ProgressCallback | None,
) -> VideoExtractionResult:
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"
    frame_index = 0
    saved_count = 0

    try:
        imageio = importlib.import_module("imageio")
    except Exception as exc:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"Fallback backend is unavailable: {exc}",
        )

    try:
        with imageio.get_reader(
            str(video_path),
            format="ffmpeg",
            input_params=FFMPEG_TOLERANT_INPUT_PARAMS,
        ) as reader:
            for frame in reader:
                if frame_index % interval == 0:
                    # imageio returns RGB arrays; convert to BGR for cv2 encoding.
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_to_save = _apply_roi(bgr_frame, roi)
                    filename = f"{video_path.stem}_{saved_count:06d}{ext}"
                    out_path = output_dir / filename
                    written = _save_frame(frame_to_save, out_path, image_format, jpg_quality)
                    if not written:
                        return VideoExtractionResult(
                            video_path=video_path,
                            output_dir=output_dir,
                            saved_count=saved_count,
                            total_frames_seen=frame_index,
                            success=False,
                            message=f"Failed to save frame: {out_path}",
                        )
                    saved_count += 1
                    if progress_cb and saved_count % 25 == 0:
                        progress_cb(f"[fallback] Saved {saved_count} frames from {video_path.name}")

                frame_index += 1
    except Exception as exc:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=saved_count,
            total_frames_seen=frame_index,
            success=False,
            message=f"Failed to decode video with fallback backend: {exc}",
        )

    if saved_count == 0 and frame_index == 0:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="Failed to decode any frame with OpenCV and fallback backend.",
        )

    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=saved_count,
        total_frames_seen=frame_index,
        success=True,
        message="Completed with fallback backend.",
    )


def _jpg_quality_to_ffmpeg_qscale(jpg_quality: int) -> int:
    # ffmpeg jpg quality uses 2(best)-31(worst), inverse of 1-100 slider.
    q = max(1, min(100, int(jpg_quality)))
    return int(round(((100 - q) / 99) * 29 + 2))


def _run_ffmpeg_recovery_attempt(
    ffmpeg_exe: str,
    video_path: Path,
    pattern_path: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
    input_params: list[str],
    forced_format: str | None = None,
    decoder_tuning_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd: list[str] = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-loglevel",
        "error",
        *input_params,
    ]
    if forced_format:
        cmd.extend(["-f", forced_format])
    if decoder_tuning_args:
        cmd.extend(decoder_tuning_args)
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-vf",
            f"select=not(mod(n\\,{interval}))",
            "-vsync",
            "vfr",
        ]
    )
    if image_format == ImageFormat.JPG:
        cmd.extend(["-q:v", str(_jpg_quality_to_ffmpeg_qscale(jpg_quality))])
    cmd.extend(["-y", str(pattern_path)])
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")


def _extract_with_ffmpeg_recovery(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
    roi: RoiBox | None,
    progress_cb: ProgressCallback | None,
) -> VideoExtractionResult:
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"
    tmp_dir = output_dir / "__ffmpeg_recovery_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pattern_path = tmp_dir / f"frame_%06d{ext}"

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"Recovery backend is unavailable: {exc}",
        )

    attempt_errors: list[str] = []
    attempts: list[dict[str, object]] = [{"label": "auto", "forced_format": None, "input_params": FFMPEG_TOLERANT_INPUT_PARAMS}]
    attempts.extend(
        {
            "label": f"forced_{fmt}",
            "forced_format": fmt,
            "input_params": FFMPEG_TOLERANT_INPUT_PARAMS,
        }
        for fmt in FFMPEG_FORCED_INPUT_FORMATS
    )
    attempts.extend(
        [
            {
                "label": "aggressive_h264",
                "forced_format": "h264",
                "input_params": FFMPEG_RELAXED_INPUT_PARAMS,
                "decoder_tuning_args": ["-max_error_rate", "1", "-flags2", "+showall"],
            },
            {
                "label": "aggressive_auto",
                "forced_format": None,
                "input_params": FFMPEG_RELAXED_INPUT_PARAMS,
                "decoder_tuning_args": ["-max_error_rate", "1", "-flags2", "+showall"],
            },
        ]
    )
    success_label = "auto"

    for attempt in attempts:
        forced_format = attempt["forced_format"]  # type: ignore[assignment]
        label = str(attempt["label"])
        if progress_cb:
            progress_cb(f"[recovery] Trying profile: {label}")

        run = _run_ffmpeg_recovery_attempt(
            ffmpeg_exe=str(ffmpeg_exe),
            video_path=video_path,
            pattern_path=pattern_path,
            interval=interval,
            image_format=image_format,
            jpg_quality=jpg_quality,
            input_params=list(attempt["input_params"]),  # type: ignore[arg-type]
            forced_format=forced_format,
            decoder_tuning_args=list(attempt.get("decoder_tuning_args", [])),  # type: ignore[arg-type]
        )

        extracted = sorted(tmp_dir.glob(f"frame_*{ext}"))
        if extracted:
            success_label = label
            break

        stderr = (run.stderr or "").strip()
        stdout = (run.stdout or "").strip()
        details = stderr or stdout or f"ffmpeg return code {run.returncode}"
        attempt_errors.append(f"{label}: {details}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"Recovery decoder failed: {' | '.join(attempt_errors)}",
        )

    for idx, src_path in enumerate(extracted):
        dst_path = output_dir / f"{video_path.stem}_{idx:06d}{ext}"
        if roi is None:
            src_path.replace(dst_path)
        else:
            raw = np.fromfile(str(src_path), dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                src_path.replace(dst_path)
            else:
                frame_to_save = _apply_roi(frame, roi)
                if _save_frame(frame_to_save, dst_path, image_format, jpg_quality):
                    src_path.unlink(missing_ok=True)
                else:
                    src_path.replace(dst_path)
        if progress_cb and (idx + 1) % 25 == 0:
            progress_cb(f"[recovery] Saved {idx + 1} frames from {video_path.name}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=len(extracted),
        total_frames_seen=-1,
        success=True,
        message=f"Completed with recovery ffmpeg decoder ({success_label}).",
    )


def _extract_with_h264_salvage(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
    roi: RoiBox | None,
    progress_cb: ProgressCallback | None,
) -> VideoExtractionResult:
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"
    try:
        raw = video_path.read_bytes()
    except OSError as exc:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"H264 salvage failed to read file: {exc}",
        )

    start_4 = raw.find(b"\x00\x00\x00\x01")
    start_3 = raw.find(b"\x00\x00\x01")
    starts = [pos for pos in (start_4, start_3) if pos >= 0]
    if not starts:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="H264 salvage failed: no Annex-B start code found.",
        )
    start = min(starts)

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"H264 salvage backend is unavailable: {exc}",
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        stream_path = tmp_dir / f"{video_path.stem}_salvage.h264"
        stream_path.write_bytes(raw[start:])
        pattern_path = tmp_dir / f"salvage_%06d{ext}"
        run = _run_ffmpeg_recovery_attempt(
            ffmpeg_exe=str(ffmpeg_exe),
            video_path=stream_path,
            pattern_path=pattern_path,
            interval=interval,
            image_format=image_format,
            jpg_quality=jpg_quality,
            input_params=FFMPEG_RELAXED_INPUT_PARAMS,
            forced_format="h264",
            decoder_tuning_args=["-max_error_rate", "1", "-flags2", "+showall"],
        )
        extracted = sorted(tmp_dir.glob(f"salvage_*{ext}"))
        if not extracted:
            stderr = (run.stderr or "").strip()
            stdout = (run.stdout or "").strip()
            details = stderr or stdout or f"ffmpeg return code {run.returncode}"
            return VideoExtractionResult(
                video_path=video_path,
                output_dir=output_dir,
                saved_count=0,
                total_frames_seen=0,
                success=False,
                message=f"H264 salvage decoder failed: {details}",
            )

        for idx, src_path in enumerate(extracted):
            dst_path = output_dir / f"{video_path.stem}_{idx:06d}{ext}"
            if roi is None:
                src_path.replace(dst_path)
            else:
                raw_img = np.fromfile(str(src_path), dtype=np.uint8)
                frame = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)
                if frame is None:
                    src_path.replace(dst_path)
                else:
                    frame_to_save = _apply_roi(frame, roi)
                    if _save_frame(frame_to_save, dst_path, image_format, jpg_quality):
                        src_path.unlink(missing_ok=True)
                    else:
                        src_path.replace(dst_path)
            if progress_cb and (idx + 1) % 25 == 0:
                progress_cb(f"[salvage] Saved {idx + 1} frames from {video_path.name}")

        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=len(extracted),
            total_frames_seen=-1,
            success=True,
            message="Completed with raw H264 salvage decoder.",
        )


def _extract_motion_segment_with_cv2(
    video_path: Path,
    output_dir: Path,
    image_format: ImageFormat,
    jpg_quality: int,
    roi: RoiBox | None,
    motion_sampling: MotionSamplingOptions,
    backend_id: int | None,
    progress_cb: ProgressCallback | None,
) -> str:
    cap = cv2.VideoCapture(str(video_path)) if backend_id is None else cv2.VideoCapture(str(video_path), backend_id)
    if not cap.isOpened():
        cap.release()
        return "Motion sampling skipped: decode pass unavailable."

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    prev_gray: np.ndarray | None = None
    motion_scores: list[float] = []
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi_frame = _apply_roi(frame, roi)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
        prev_gray = gray
        frame_count += 1
    cap.release()

    if frame_count <= 1 or not motion_scores:
        return "Motion sampling skipped: insufficient frames."

    start_idx, end_idx = _find_major_motion_window(
        motion_scores=motion_scores,
        fps=fps,
        expected_duration_sec=motion_sampling.normalized_duration_sec(),
    )

    motion_dir = output_dir / "motion_segment"
    motion_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"

    cap2 = cv2.VideoCapture(str(video_path)) if backend_id is None else cv2.VideoCapture(str(video_path), backend_id)
    if not cap2.isOpened():
        cap2.release()
        return "Motion sampling skipped: write pass unavailable."

    idx = 0
    saved = 0
    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        if start_idx <= idx <= end_idx:
            roi_frame = _apply_roi(frame, roi)
            filename = f"{video_path.stem}_motion_{saved:06d}{ext}"
            out_path = motion_dir / filename
            if not _save_frame(roi_frame, out_path, image_format, jpg_quality):
                cap2.release()
                return f"Motion sampling stopped: failed to save {out_path.name}."
            saved += 1
            if progress_cb and saved % 25 == 0:
                progress_cb(f"[motion] Saved {saved} frames from {video_path.name}")
        idx += 1
        if idx > end_idx:
            break
    cap2.release()
    return f"Motion segment saved {saved} frame(s) in {motion_dir} (frame {start_idx}..{end_idx})."


def extract_video_frames(
    video_path: Path,
    options: ExtractionOptions,
    progress_cb: ProgressCallback | None = None,
) -> VideoExtractionResult:
    video_path = Path(video_path)
    base_dir = get_runtime_base_dir()
    output_root = base_dir / "output"
    output_dir = make_unique_output_dir(video_path.stem, output_root)

    interval = options.normalized_interval()
    image_format = options.image_format
    jpg_quality = options.normalized_jpg_quality()
    roi = options.roi.normalized() if options.roi is not None else None
    motion_sampling = options.motion_sampling or MotionSamplingOptions()

    cv2_result = _extract_with_cv2(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        image_format=image_format,
        jpg_quality=jpg_quality,
        roi=roi,
        motion_sampling=motion_sampling,
        progress_cb=progress_cb,
    )
    if cv2_result is not None:
        return cv2_result

    if progress_cb:
        progress_cb(f"[fallback] OpenCV decode failed for {video_path.name}. Trying ffmpeg backend.")
    imageio_result = _extract_with_imageio_fallback(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        image_format=image_format,
        jpg_quality=jpg_quality,
        roi=roi,
        progress_cb=progress_cb,
    )
    if imageio_result.success:
        return imageio_result

    if progress_cb:
        progress_cb(f"[recovery] Trying tolerant ffmpeg recovery for {video_path.name}.")
    recovery_result = _extract_with_ffmpeg_recovery(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        image_format=image_format,
        jpg_quality=jpg_quality,
        roi=roi,
        progress_cb=progress_cb,
    )
    if recovery_result.success:
        return recovery_result

    if progress_cb:
        progress_cb(f"[salvage] Trying raw H264 salvage for {video_path.name}.")
    salvage_result = _extract_with_h264_salvage(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        image_format=image_format,
        jpg_quality=jpg_quality,
        roi=roi,
        progress_cb=progress_cb,
    )
    if salvage_result.success:
        return salvage_result

    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=0,
        total_frames_seen=0,
        success=False,
        message=f"{imageio_result.message}\n\n{recovery_result.message}\n\n{salvage_result.message}",
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


def _safe_import_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable: {exc}"


def _probe_cv2_backend(video_path: Path, backend_name: str, backend_id: int | None) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video_path)) if backend_id is None else cv2.VideoCapture(str(video_path), backend_id)
    opened = bool(cap.isOpened())
    read_ok = False
    width = 0
    height = 0
    fps = 0.0
    frame_count = 0.0
    if opened:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        ok, frame = cap.read()
        read_ok = bool(ok and frame is not None)
    cap.release()
    return {
        "backend": backend_name,
        "opened": opened,
        "read_first_frame": read_ok,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def _probe_imageio_reader(video_path: Path) -> dict[str, object]:
    try:
        imageio = importlib.import_module("imageio")
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    try:
        with imageio.get_reader(
            str(video_path),
            format="ffmpeg",
            input_params=FFMPEG_TOLERANT_INPUT_PARAMS,
        ) as reader:
            metadata = dict(reader.get_meta_data() or {})
            first_frame_ok = False
            for frame in reader:
                if frame is not None:
                    first_frame_ok = True
                break
            nframes_raw = metadata.get("nframes")
            try:
                nframes_value = float(nframes_raw) if nframes_raw is not None else 0.0
            except (TypeError, ValueError):
                nframes_value = 0.0
            return {
                "available": True,
                "first_frame_ok": first_frame_ok,
                "meta_keys": sorted(metadata.keys()),
                "fps": float(metadata.get("fps") or 0.0),
                "nframes": int(nframes_value) if math.isfinite(nframes_value) else -1,
            }
    except Exception as exc:
        return {"available": True, "first_frame_ok": False, "error": str(exc)}


def _probe_recovery_ffmpeg(video_path: Path) -> dict[str, object]:
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    with tempfile.TemporaryDirectory() as tmp:
        sample_out = Path(tmp) / "sample_%06d.jpg"
        runs: list[dict[str, object]] = []
        probe_attempts: list[dict[str, object]] = [{"label": "auto", "forced_format": None, "input_params": FFMPEG_TOLERANT_INPUT_PARAMS}]
        probe_attempts.extend(
            {
                "label": f"forced_{fmt}",
                "forced_format": fmt,
                "input_params": FFMPEG_TOLERANT_INPUT_PARAMS,
            }
            for fmt in FFMPEG_FORCED_INPUT_FORMATS
        )
        probe_attempts.extend(
            [
                {
                    "label": "aggressive_h264",
                    "forced_format": "h264",
                    "input_params": FFMPEG_RELAXED_INPUT_PARAMS,
                    "decoder_tuning_args": ["-max_error_rate", "1", "-flags2", "+showall"],
                },
                {
                    "label": "aggressive_auto",
                    "forced_format": None,
                    "input_params": FFMPEG_RELAXED_INPUT_PARAMS,
                    "decoder_tuning_args": ["-max_error_rate", "1", "-flags2", "+showall"],
                },
            ]
        )
        for attempt in probe_attempts:
            forced_format = attempt["forced_format"]  # type: ignore[assignment]
            label = str(attempt["label"])
            cmd = [
                str(ffmpeg_exe),
                "-hide_banner",
                "-loglevel",
                "error",
                *list(attempt["input_params"]),  # type: ignore[arg-type]
            ]
            if forced_format:
                cmd.extend(["-f", forced_format])
            cmd.extend(list(attempt.get("decoder_tuning_args", [])))  # type: ignore[arg-type]
            cmd.extend(
                [
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-y",
                    str(sample_out),
                ]
            )
            run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            extracted = list(Path(tmp).glob("sample_*.jpg"))
            runs.append(
                {
                    "profile": label,
                    "format": forced_format or "auto",
                    "returncode": run.returncode,
                    "stdout": (run.stdout or "").strip(),
                    "stderr": (run.stderr or "").strip(),
                    "extracted_first_frame": bool(extracted),
                }
            )
            if extracted:
                break
        return {
            "available": True,
            "attempts": runs,
            "extracted_first_frame": any(bool(item.get("extracted_first_frame")) for item in runs),
        }


def collect_decode_diagnostics(video_path: Path, options: ExtractionOptions, repeat: int = 1) -> dict[str, object]:
    video_path = Path(video_path)
    repeat = max(1, int(repeat))

    backend_candidates: list[tuple[str, int | None]] = [("default", None)]
    for backend_name in CV2_BACKEND_ORDER:
        if backend_name == "default":
            continue
        backend_id = getattr(cv2, backend_name, None)
        if isinstance(backend_id, int):
            backend_candidates.append((backend_name.lower(), backend_id))

    env: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cv2_version": cv2.__version__,
        "imageio_version": _safe_import_version("imageio"),
        "imageio_ffmpeg_version": _safe_import_version("imageio_ffmpeg"),
        "video_path": str(video_path),
        "video_exists": video_path.exists(),
        "video_size_bytes": int(video_path.stat().st_size) if video_path.exists() else 0,
        "cv2_backend_order": [name for name, _ in backend_candidates],
    }

    cv2_probe = [_probe_cv2_backend(video_path, name, backend_id) for name, backend_id in backend_candidates]
    imageio_probe = _probe_imageio_reader(video_path)
    recovery_probe = _probe_recovery_ffmpeg(video_path)

    extraction_runs: list[dict[str, object]] = []
    for run_idx in range(1, repeat + 1):
        messages: list[str] = []
        result = extract_video_frames(video_path, options, progress_cb=messages.append)
        extraction_runs.append(
            {
                "run": run_idx,
                "success": result.success,
                "saved_count": result.saved_count,
                "total_frames_seen": result.total_frames_seen,
                "message": result.message,
                "progress_log": messages,
                "output_dir": str(result.output_dir),
            }
        )

    return {
        "environment": env,
        "probe": {
            "cv2": cv2_probe,
            "imageio": imageio_probe,
            "recovery_ffmpeg": recovery_probe,
        },
        "extraction_runs": extraction_runs,
    }
