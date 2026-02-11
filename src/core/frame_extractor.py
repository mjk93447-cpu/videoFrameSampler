from __future__ import annotations

import importlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
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


def _truncate_error_block(text: str, max_chars: int = 320) -> str:
    cleaned = " ".join(str(text or "").replace("\r", "\n").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _first_line_or_self(message: str) -> str:
    raw = str(message or "").strip()
    if not raw:
        return "empty message"
    first_line = raw.splitlines()[0].strip()
    return first_line or raw


def _collect_decode_hints(stage_messages: dict[str, str]) -> list[str]:
    haystack = "\n".join(stage_messages.values()).lower()
    hints: list[str] = []
    if "format avi detected only with low score" in haystack:
        hints.append("AVI container header/index looks damaged or mislabeled.")
    if "non-existing pps" in haystack or "decode_slice_header error" in haystack:
        hints.append("H264 bitstream likely misses required SPS/PPS metadata.")
    if "moov atom not found" in haystack:
        hints.append("MP4/MOV metadata atom is missing (container not valid as MP4).")
    if "ebml header parsing failed" in haystack:
        hints.append("File does not contain a valid Matroska/WebM header.")
    if "could not find codec parameters" in haystack:
        hints.append("Stream parameters are not discoverable from current container headers.")
    if "container carving found no embedded stream signature" in haystack:
        hints.append("No secondary embedded container signature was detected in raw bytes.")
    if "legacy codec bridge failed" in haystack and "unknown error occurred" in haystack:
        hints.append("AviSynth/DirectShow bridge is unavailable or script parsing failed.")

    dedup: list[str] = []
    for hint in hints:
        if hint not in dedup:
            dedup.append(hint)
    return dedup


def _write_decode_failure_report(
    output_dir: Path,
    video_path: Path,
    stage_messages: dict[str, str],
) -> Path:
    report_path = output_dir / "decode_failure_report.txt"
    file_size = 0
    header_hex = "n/a"
    try:
        raw = video_path.read_bytes()
        file_size = len(raw)
        header_hex = raw[:64].hex(" ")
    except OSError:
        pass

    lines: list[str] = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"video_path: {video_path}",
        f"video_exists: {video_path.exists()}",
        f"video_size_bytes: {file_size}",
        f"header_hex_64: {header_hex}",
        "",
        "stage_summary:",
    ]
    for stage, message in stage_messages.items():
        lines.append(f"- {stage}: {_truncate_error_block(_first_line_or_self(message), 500)}")
    lines.extend(["", "stage_details:"])
    for stage, message in stage_messages.items():
        lines.append(f"[{stage}]")
        lines.append(str(message or ""))
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_decode_failure_report_json(
    output_dir: Path,
    video_path: Path,
    stage_messages: dict[str, str],
    hints: list[str],
) -> Path:
    report_path = output_dir / "decode_failure_report.json"
    file_size = 0
    header_hex = "n/a"
    carved_candidates: list[dict[str, object]] = []
    try:
        raw = video_path.read_bytes()
        file_size = len(raw)
        header_hex = raw[:64].hex(" ")
        carved_candidates = [
            {"offset": offset, "label": label, "extension": ext}
            for offset, label, ext in _scan_container_offsets(raw)
        ][:30]
    except OSError:
        pass

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_path": str(video_path),
        "video_exists": video_path.exists(),
        "video_size_bytes": file_size,
        "header_hex_64": header_hex,
        "hints": hints,
        "carving_candidates": carved_candidates,
        "stage_summary": {stage: _first_line_or_self(message) for stage, message in stage_messages.items()},
        "stage_details": stage_messages,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def _build_compact_failure_message(
    video_path: Path,
    stage_messages: dict[str, str],
    report_path: Path,
    json_report_path: Path,
) -> str:
    stage_order = ("fallback", "recovery", "carving", "repair", "legacy", "salvage")
    compact_lines = [f"All decode pipelines failed for {video_path.name}."]
    hints = _collect_decode_hints(stage_messages)
    if hints:
        compact_lines.append("Likely causes:")
        compact_lines.extend(f"- {hint}" for hint in hints[:5])

    compact_lines.append("Stage summary:")
    for stage in stage_order:
        if stage in stage_messages:
            compact_lines.append(f"- {stage}: {_truncate_error_block(_first_line_or_self(stage_messages[stage]))}")
    compact_lines.append(f"Detailed report: {report_path}")
    compact_lines.append(f"Structured report: {json_report_path}")
    return "\n".join(compact_lines)


def _emit_failure_diagnostics_to_progress(
    progress_cb: ProgressCallback | None,
    stage_messages: dict[str, str],
    hints: list[str],
    report_path: Path,
    json_report_path: Path,
) -> None:
    if progress_cb is None:
        return
    progress_cb("[error] ----- decode failure diagnostics begin -----")
    for hint in hints:
        progress_cb(f"[error][hint] {hint}")
    for stage, message in stage_messages.items():
        progress_cb(f"[error][summary] {stage}: {_truncate_error_block(_first_line_or_self(message), 500)}")
    for stage, message in stage_messages.items():
        progress_cb(f"[error][detail:{stage}] begin")
        detail_text = str(message or "")
        for raw_line in detail_text.splitlines() or [""]:
            if " | " in raw_line and len(raw_line) > 500:
                for part in raw_line.split(" | "):
                    progress_cb(f"[error][detail:{stage}] {part.strip()}")
            else:
                progress_cb(f"[error][detail:{stage}] {raw_line}")
        progress_cb(f"[error][detail:{stage}] end")
    progress_cb(f"[error] Detailed decode report saved: {report_path}")
    progress_cb(f"[error] Structured decode report saved: {json_report_path}")
    progress_cb("[error] ----- decode failure diagnostics end -----")


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


def _scan_container_offsets(raw: bytes) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []

    riff_pos = raw.find(b"RIFF")
    while riff_pos >= 0:
        if riff_pos + 12 <= len(raw):
            marker = raw[riff_pos + 8 : riff_pos + 12]
            if marker == b"AVI ":
                candidates.append((riff_pos, "riff_avi", ".avi"))
        riff_pos = raw.find(b"RIFF", riff_pos + 1)

    ftyp_pos = raw.find(b"ftyp")
    while ftyp_pos >= 0:
        start = max(0, ftyp_pos - 4)
        if start + 12 <= len(raw):
            candidates.append((start, "mp4_ftyp", ".mp4"))
        ftyp_pos = raw.find(b"ftyp", ftyp_pos + 1)

    asf_guid = b"\x30\x26\xB2\x75\x8E\x66\xCF\x11\xA6\xD9\x00\xAA\x00\x62\xCE\x6C"
    asf_pos = raw.find(asf_guid)
    while asf_pos >= 0:
        candidates.append((asf_pos, "asf_guid", ".asf"))
        asf_pos = raw.find(asf_guid, asf_pos + 1)

    ebml_pos = raw.find(b"\x1A\x45\xDF\xA3")
    while ebml_pos >= 0:
        candidates.append((ebml_pos, "ebml", ".mkv"))
        ebml_pos = raw.find(b"\x1A\x45\xDF\xA3", ebml_pos + 1)

    ogg_pos = raw.find(b"OggS")
    while ogg_pos >= 0:
        candidates.append((ogg_pos, "ogg", ".ogv"))
        ogg_pos = raw.find(b"OggS", ogg_pos + 1)

    dedup: dict[int, tuple[int, str, str]] = {}
    for offset, label, ext in sorted(candidates, key=lambda item: item[0]):
        if offset <= 0:
            continue
        dedup.setdefault(offset, (offset, label, ext))
    return list(dedup.values())


def _find_major_motion_window(
    motion_scores: list[float],
    fps: float,
    expected_duration_sec: float,
    tolerance_ratio: float,
) -> tuple[int, int]:
    if not motion_scores:
        return 0, 0
    approx_len = max(1, int(round(expected_duration_sec * fps)))
    min_len = max(1, int(round(approx_len * (1.0 - tolerance_ratio))))
    max_len = max(min_len, int(round(approx_len * (1.0 + tolerance_ratio))))

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


def _normalize_output_filenames(output_dir: Path, source_stem: str, target_stem: str, ext: str) -> None:
    files = sorted(output_dir.glob(f"{source_stem}_*{ext}"))
    for idx, src in enumerate(files):
        dst = output_dir / f"{target_stem}_{idx:06d}{ext}"
        if src != dst:
            src.replace(dst)

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
        if hasattr(cap, "get"):
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        else:
            fps = 30.0
        prev_gray: np.ndarray | None = None
        motion_scores: list[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if motion_sampling and motion_sampling.enabled:
                roi_for_motion = _apply_roi(frame, roi)
                gray = cv2.cvtColor(roi_for_motion, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    motion_scores.append(float(np.mean(diff)))
                prev_gray = gray

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
        if motion_sampling and motion_sampling.enabled and motion_scores:
            start_idx, end_idx = _find_major_motion_window(
                motion_scores=motion_scores,
                fps=fps,
                expected_duration_sec=motion_sampling.normalized_duration_sec(),
                tolerance_ratio=motion_sampling.normalized_tolerance_ratio(),
            )
            motion_result = _save_motion_segment_with_cv2(
                video_path=video_path,
                output_dir=output_dir,
                image_format=image_format,
                jpg_quality=jpg_quality,
                roi=roi,
                start_idx=start_idx,
                end_idx=end_idx,
                backend_id=backend_id,
                progress_cb=progress_cb,
            )
            message = f"{message} {motion_result}"
        elif motion_sampling and motion_sampling.enabled:
            message = f"{message} Motion sampling skipped: insufficient frames."

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
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
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
        reader = imageio_ffmpeg.read_frames(
            str(video_path),
            input_params=FFMPEG_TOLERANT_INPUT_PARAMS,
        )
        meta = next(reader)
        width, height = meta.get("size", (0, 0))
        if not width or not height:
            raise RuntimeError("Fallback ffmpeg pipe did not provide frame size metadata.")

        for frame_bytes in reader:
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            if frame_array.size != width * height * 3:
                frame_index += 1
                continue
            rgb = frame_array.reshape((height, width, 3))
            if frame_index % interval == 0:
                bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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

    def _collect_h264_candidate_offsets(payload: bytes, max_candidates: int = 12) -> list[tuple[int, str]]:
        # Collect multiple Annex-B starts and prioritize offsets that look like SPS/PPS/IDR entry points.
        candidates: list[tuple[int, str]] = []
        seen: set[int] = set()
        i = 0
        size = len(payload)
        while i + 5 < size and len(candidates) < max_candidates:
            start_len = 0
            if payload[i : i + 4] == b"\x00\x00\x00\x01":
                start_len = 4
            elif payload[i : i + 3] == b"\x00\x00\x01":
                start_len = 3
            if start_len == 0:
                i += 1
                continue
            nal_header_idx = i + start_len
            if nal_header_idx >= size:
                break
            nal_type = int(payload[nal_header_idx] & 0x1F)
            if i not in seen:
                seen.add(i)
                if nal_type == 7:
                    candidates.append((i, "sps"))
                elif nal_type == 8:
                    candidates.append((i, "pps"))
                elif nal_type == 5:
                    candidates.append((i, "idr"))
                elif nal_type == 1:
                    candidates.append((i, "slice"))
                else:
                    candidates.append((i, f"nal_{nal_type}"))
            i = nal_header_idx + 1

        # Prefer likely decoder-bootstrap points first.
        priority = {"sps": 0, "pps": 1, "idr": 2, "slice": 3}
        return sorted(candidates, key=lambda item: (priority.get(item[1], 9), item[0]))[:max_candidates]

    def _iter_annexb_units(payload: bytes) -> list[tuple[int, bytes]]:
        markers: list[tuple[int, int]] = []
        i = 0
        end = len(payload)
        while i + 3 < end:
            marker_len = 0
            if payload[i : i + 4] == b"\x00\x00\x00\x01":
                marker_len = 4
            elif payload[i : i + 3] == b"\x00\x00\x01":
                marker_len = 3
            if marker_len:
                markers.append((i, marker_len))
                i += marker_len
            else:
                i += 1

        units: list[tuple[int, bytes]] = []
        for idx, (start, marker_len) in enumerate(markers):
            nal_start = start + marker_len
            next_start = markers[idx + 1][0] if idx + 1 < len(markers) else end
            nal = payload[nal_start:next_start]
            if not nal:
                continue
            units.append((int(nal[0] & 0x1F), bytes(nal)))
        return units

    def _extract_sps_pps(payload: bytes) -> tuple[bytes | None, bytes | None]:
        sps: bytes | None = None
        pps: bytes | None = None
        for nal_type, nal in _iter_annexb_units(payload):
            if nal_type == 7 and sps is None:
                sps = nal
            elif nal_type == 8 and pps is None:
                pps = nal
            if sps is not None and pps is not None:
                break
        return sps, pps

    def _rebuild_h264_with_parameter_sets(payload: bytes, sps: bytes, pps: bytes) -> bytes:
        units = _iter_annexb_units(payload)
        if not units:
            return b""
        start_code = b"\x00\x00\x00\x01"
        rebuilt = bytearray()
        # Bootstrap decoder context once at stream head.
        rebuilt.extend(start_code + sps)
        rebuilt.extend(start_code + pps)
        for nal_type, nal in units:
            if nal_type == 5:
                # Re-announce SPS/PPS before IDR frames for damaged streams.
                rebuilt.extend(start_code + sps)
                rebuilt.extend(start_code + pps)
            rebuilt.extend(start_code + nal)
        return bytes(rebuilt)

    candidates = _collect_h264_candidate_offsets(raw)
    if not candidates:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="H264 salvage failed: no Annex-B start code found.",
        )

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

    sps, pps = _extract_sps_pps(raw)
    if progress_cb:
        progress_cb(
            "[salvage] SPS/PPS scan: "
            + ("found SPS" if sps is not None else "no SPS")
            + ", "
            + ("found PPS" if pps is not None else "no PPS")
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pattern_path = tmp_dir / f"salvage_%06d{ext}"
        attempt_errors: list[str] = []
        extracted: list[Path] = []
        success_mode = "raw"

        for candidate_offset, candidate_label in candidates:
            if progress_cb:
                progress_cb(
                    f"[salvage] Trying candidate offset {candidate_offset} ({candidate_label})."
                )
            for stale in tmp_dir.glob(f"salvage_*{ext}"):
                stale.unlink(missing_ok=True)

            stream_path = tmp_dir / f"{video_path.stem}_salvage_{candidate_offset}.h264"
            stream_path.write_bytes(raw[candidate_offset:])

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
            if extracted:
                success_mode = f"raw_{candidate_label}@{candidate_offset}"
                break

            stderr = (run.stderr or "").strip()
            stdout = (run.stdout or "").strip()
            details = stderr or stdout or f"ffmpeg return code {run.returncode}"
            attempt_errors.append(f"{candidate_label}@{candidate_offset}: {details}")

            # Immediate reconstruction route: prepend/inject SPS/PPS if available.
            if sps is not None and pps is not None:
                if progress_cb:
                    progress_cb(
                        f"[salvage] Trying SPS/PPS reconstruction for offset {candidate_offset} ({candidate_label})."
                    )
                rebuilt = _rebuild_h264_with_parameter_sets(raw[candidate_offset:], sps, pps)
                if rebuilt:
                    for stale in tmp_dir.glob(f"salvage_*{ext}"):
                        stale.unlink(missing_ok=True)
                    rebuilt_path = tmp_dir / f"{video_path.stem}_salvage_rebuild_{candidate_offset}.h264"
                    rebuilt_path.write_bytes(rebuilt)
                    rebuilt_run = _run_ffmpeg_recovery_attempt(
                        ffmpeg_exe=str(ffmpeg_exe),
                        video_path=rebuilt_path,
                        pattern_path=pattern_path,
                        interval=interval,
                        image_format=image_format,
                        jpg_quality=jpg_quality,
                        input_params=FFMPEG_RELAXED_INPUT_PARAMS,
                        forced_format="h264",
                        decoder_tuning_args=["-max_error_rate", "1", "-flags2", "+showall"],
                    )
                    extracted = sorted(tmp_dir.glob(f"salvage_*{ext}"))
                    if extracted:
                        success_mode = f"rebuild_{candidate_label}@{candidate_offset}"
                        break
                    rebuilt_stderr = (rebuilt_run.stderr or "").strip()
                    rebuilt_stdout = (rebuilt_run.stdout or "").strip()
                    rebuilt_details = rebuilt_stderr or rebuilt_stdout or f"ffmpeg return code {rebuilt_run.returncode}"
                    attempt_errors.append(
                        f"rebuild_{candidate_label}@{candidate_offset}: {rebuilt_details}"
                    )

        if not extracted:
            return VideoExtractionResult(
                video_path=video_path,
                output_dir=output_dir,
                saved_count=0,
                total_frames_seen=0,
                success=False,
                message=f"H264 salvage decoder failed: {' | '.join(attempt_errors)}",
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
            message=f"Completed with raw H264 salvage decoder ({success_mode}).",
        )


def _extract_with_legacy_codec_bridge(
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
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"Legacy codec bridge unavailable: {exc}",
        )

    avs_path = str(video_path).replace("\\", "/").replace('"', '\\"')
    profiles: tuple[tuple[str, str], ...] = (
        ("avisource", f'AviSource("{avs_path}", audio=false)'),
        ("directshow", f'DirectShowSource("{avs_path}", audio=false, convertfps=true)'),
    )
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pattern_path = tmp_dir / f"legacy_%06d{ext}"
        for name, script_line in profiles:
            if progress_cb:
                progress_cb(f"[legacy] Trying codec bridge profile: {name}")
            script_path = tmp_dir / f"bridge_{name}.avs"
            script_path.write_text(script_line + "\n", encoding="utf-8")

            cmd: list[str] = [
                str(ffmpeg_exe),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "avisynth",
                "-i",
                str(script_path),
                "-vf",
                f"select=not(mod(n\\,{interval}))",
                "-vsync",
                "vfr",
            ]
            if image_format == ImageFormat.JPG:
                cmd.extend(["-q:v", str(_jpg_quality_to_ffmpeg_qscale(jpg_quality))])
            cmd.extend(["-y", str(pattern_path)])
            run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            extracted = sorted(tmp_dir.glob(f"legacy_*{ext}"))
            if not extracted:
                stderr = (run.stderr or "").strip()
                stdout = (run.stdout or "").strip()
                details = stderr or stdout or f"ffmpeg return code {run.returncode}"
                errors.append(f"{name}: {details}")
                continue

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
                    progress_cb(f"[legacy] Saved {idx + 1} frames from {video_path.name}")

            return VideoExtractionResult(
                video_path=video_path,
                output_dir=output_dir,
                saved_count=len(extracted),
                total_frames_seen=-1,
                success=True,
                message=f"Completed with legacy codec bridge ({name}).",
            )

    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=0,
        total_frames_seen=0,
        success=False,
        message=f"Legacy codec bridge failed: {' | '.join(errors)}",
    )


def _extract_with_container_carving(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
    roi: RoiBox | None,
    progress_cb: ProgressCallback | None,
) -> VideoExtractionResult:
    try:
        raw = video_path.read_bytes()
    except OSError as exc:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"Container carving failed to read file: {exc}",
        )

    candidates = _scan_container_offsets(raw)
    if not candidates:
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="Container carving found no embedded stream signature.",
        )

    errors: list[str] = []
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for offset, label, carved_ext in candidates:
            if progress_cb:
                progress_cb(f"[carve] Trying embedded stream at offset {offset} ({label}).")
            carved_path = tmp_dir / f"{video_path.stem}__carved_{offset}{carved_ext}"
            carved_path.write_bytes(raw[offset:])

            recovery = _extract_with_ffmpeg_recovery(
                video_path=carved_path,
                output_dir=output_dir,
                interval=interval,
                image_format=image_format,
                jpg_quality=jpg_quality,
                roi=roi,
                progress_cb=progress_cb,
            )
            if recovery.success:
                _normalize_output_filenames(output_dir, carved_path.stem, video_path.stem, ext)
                return VideoExtractionResult(
                    video_path=video_path,
                    output_dir=output_dir,
                    saved_count=recovery.saved_count,
                    total_frames_seen=recovery.total_frames_seen,
                    success=True,
                    message=f"Completed with container carving ({label} @ {offset}).",
                )
            errors.append(f"{label}@{offset}: {recovery.message}")

    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=0,
        total_frames_seen=0,
        success=False,
        message=f"Container carving failed: {' | '.join(errors)}",
    )


def _save_motion_segment_with_cv2(
    video_path: Path,
    output_dir: Path,
    image_format: ImageFormat,
    jpg_quality: int,
    roi: RoiBox | None,
    start_idx: int,
    end_idx: int,
    backend_id: int | None,
    progress_cb: ProgressCallback | None,
) -> str:
    cap = cv2.VideoCapture(str(video_path)) if backend_id is None else cv2.VideoCapture(str(video_path), backend_id)
    if not cap.isOpened():
        cap.release()
        return "Motion sampling skipped: write pass unavailable."

    motion_dir = output_dir / "motion_segment"
    motion_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if image_format == ImageFormat.JPG else ".png"

    idx = start_idx
    saved = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_idx))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx <= end_idx:
            roi_frame = _apply_roi(frame, roi)
            filename = f"{video_path.stem}_motion_{saved:06d}{ext}"
            out_path = motion_dir / filename
            if not _save_frame(roi_frame, out_path, image_format, jpg_quality):
                cap.release()
                return f"Motion sampling stopped: failed to save {out_path.name}."
            saved += 1
            if progress_cb and saved % 25 == 0:
                progress_cb(f"[motion] Saved {saved} frames from {video_path.name}")
        idx += 1
        if idx > end_idx:
            break
    cap.release()
    return f"Motion segment saved {saved} frame(s) in {motion_dir} (frame {start_idx}..{end_idx})."


def _run_ffmpeg_repair_attempt(
    ffmpeg_exe: str,
    video_path: Path,
    repaired_path: Path,
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
            "-map",
            "0:v:0?",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(repaired_path),
        ]
    )
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")


def _repair_video_with_ffmpeg(
    video_path: Path,
    output_dir: Path,
    progress_cb: ProgressCallback | None,
) -> tuple[Path | None, str]:
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return None, f"Repair backend is unavailable: {exc}"

    repaired_dir = output_dir / "__repaired_video_tmp"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = repaired_dir / f"{video_path.stem}__repaired.mp4"
    attempts: list[dict[str, object]] = [{"label": "repair_auto", "forced_format": None, "input_params": FFMPEG_TOLERANT_INPUT_PARAMS}]
    attempts.extend(
        {
            "label": f"repair_forced_{fmt}",
            "forced_format": fmt,
            "input_params": FFMPEG_TOLERANT_INPUT_PARAMS,
        }
        for fmt in FFMPEG_FORCED_INPUT_FORMATS
    )
    attempts.extend(
        [
            {
                "label": "repair_aggressive_h264",
                "forced_format": "h264",
                "input_params": FFMPEG_RELAXED_INPUT_PARAMS,
                "decoder_tuning_args": ["-max_error_rate", "1", "-flags2", "+showall"],
            },
            {
                "label": "repair_aggressive_auto",
                "forced_format": None,
                "input_params": FFMPEG_RELAXED_INPUT_PARAMS,
                "decoder_tuning_args": ["-max_error_rate", "1", "-flags2", "+showall"],
            },
        ]
    )
    attempt_errors: list[str] = []

    for attempt in attempts:
        label = str(attempt["label"])
        if progress_cb:
            progress_cb(f"[repair] Trying profile: {label}")
        run = _run_ffmpeg_repair_attempt(
            ffmpeg_exe=str(ffmpeg_exe),
            video_path=video_path,
            repaired_path=repaired_path,
            input_params=list(attempt["input_params"]),  # type: ignore[arg-type]
            forced_format=attempt["forced_format"],  # type: ignore[arg-type]
            decoder_tuning_args=list(attempt.get("decoder_tuning_args", [])),  # type: ignore[arg-type]
        )
        if repaired_path.exists() and repaired_path.stat().st_size > 0:
            return repaired_path, f"Repaired video created with profile {label}."

        stderr = (run.stderr or "").strip()
        stdout = (run.stdout or "").strip()
        details = stderr or stdout or f"ffmpeg return code {run.returncode}"
        attempt_errors.append(f"{label}: {details}")

    return None, f"Repair failed: {' | '.join(attempt_errors)}"


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
        progress_cb(f"[carve] Trying embedded container carving for {video_path.name}.")
    carving_result = _extract_with_container_carving(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        image_format=image_format,
        jpg_quality=jpg_quality,
        roi=roi,
        progress_cb=progress_cb,
    )
    if carving_result.success:
        return carving_result

    if progress_cb:
        progress_cb(f"[repair] Trying video repair transcode for {video_path.name}.")
    repaired_video_path, repair_message = _repair_video_with_ffmpeg(
        video_path=video_path,
        output_dir=output_dir,
        progress_cb=progress_cb,
    )
    if repaired_video_path is not None:
        repaired_result = _extract_with_cv2(
            video_path=repaired_video_path,
            output_dir=output_dir,
            interval=interval,
            image_format=image_format,
            jpg_quality=jpg_quality,
            roi=roi,
            motion_sampling=motion_sampling,
            progress_cb=progress_cb,
        )
        if repaired_result is not None and repaired_result.success:
            ext = ".jpg" if image_format == ImageFormat.JPG else ".png"
            repaired_files = sorted(output_dir.glob(f"{repaired_video_path.stem}_*{ext}"))
            for idx, src in enumerate(repaired_files):
                src.replace(output_dir / f"{video_path.stem}_{idx:06d}{ext}")
            return VideoExtractionResult(
                video_path=video_path,
                output_dir=output_dir,
                saved_count=repaired_result.saved_count,
                total_frames_seen=repaired_result.total_frames_seen,
                success=True,
                message=f"{repair_message} Completed extraction from repaired stream.",
            )

    if progress_cb:
        progress_cb(f"[legacy] Trying legacy codec bridge for {video_path.name}.")
    legacy_result = _extract_with_legacy_codec_bridge(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        image_format=image_format,
        jpg_quality=jpg_quality,
        roi=roi,
        progress_cb=progress_cb,
    )
    if legacy_result.success:
        return legacy_result

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

    stage_messages = {
        "fallback": imageio_result.message,
        "recovery": recovery_result.message,
        "carving": carving_result.message,
        "repair": repair_message,
        "legacy": legacy_result.message,
        "salvage": salvage_result.message,
    }
    report_path = _write_decode_failure_report(
        output_dir=output_dir,
        video_path=video_path,
        stage_messages=stage_messages,
    )
    hints = _collect_decode_hints(stage_messages)
    json_report_path = _write_decode_failure_report_json(
        output_dir=output_dir,
        video_path=video_path,
        stage_messages=stage_messages,
        hints=hints,
    )
    _emit_failure_diagnostics_to_progress(
        progress_cb=progress_cb,
        stage_messages=stage_messages,
        hints=hints,
        report_path=report_path,
        json_report_path=json_report_path,
    )
    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=0,
        total_frames_seen=0,
        success=False,
        message=_build_compact_failure_message(video_path, stage_messages, report_path, json_report_path),
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
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    try:
        reader = imageio_ffmpeg.read_frames(
            str(video_path),
            input_params=FFMPEG_TOLERANT_INPUT_PARAMS,
        )
        metadata = dict(next(reader) or {})
        first_frame_ok = False
        try:
            frame_bytes = next(reader)
            first_frame_ok = frame_bytes is not None and len(frame_bytes) > 0
        except StopIteration:
            first_frame_ok = False
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
        "imageio_version": "removed (using imageio_ffmpeg pipe fallback)",
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
