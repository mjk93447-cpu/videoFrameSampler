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

from core.models import ExtractionOptions, ImageFormat, VideoExtractionResult

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


def _extract_with_cv2(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
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
                filename = f"{video_path.stem}_{saved_count:06d}{ext}"
                out_path = output_dir / filename
                written = _save_frame(frame, out_path, image_format, jpg_quality)
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

        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=saved_count,
            total_frames_seen=frame_index,
            success=True,
            message=f"Completed with OpenCV backend: {backend_name}.",
        )

    return None


def _extract_with_imageio_fallback(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
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
                    filename = f"{video_path.stem}_{saved_count:06d}{ext}"
                    out_path = output_dir / filename
                    written = _save_frame(bgr_frame, out_path, image_format, jpg_quality)
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


def _extract_with_ffmpeg_recovery(
    video_path: Path,
    output_dir: Path,
    interval: int,
    image_format: ImageFormat,
    jpg_quality: int,
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

    cmd: list[str] = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-loglevel",
        "error",
        *FFMPEG_TOLERANT_INPUT_PARAMS,
        "-i",
        str(video_path),
        "-vf",
        f"select=not(mod(n\\,{interval}))",
        "-vsync",
        "vfr",
    ]
    if image_format == ImageFormat.JPG:
        cmd.extend(["-q:v", str(_jpg_quality_to_ffmpeg_qscale(jpg_quality))])
    cmd.extend(["-y", str(pattern_path)])

    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")

    extracted = sorted(tmp_dir.glob(f"frame_*{ext}"))
    if not extracted:
        stderr = (run.stderr or "").strip()
        stdout = (run.stdout or "").strip()
        details = stderr or stdout or f"ffmpeg return code {run.returncode}"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=output_dir,
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"Recovery decoder failed: {details}",
        )

    for idx, src_path in enumerate(extracted):
        dst_path = output_dir / f"{video_path.stem}_{idx:06d}{ext}"
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
        message="Completed with recovery ffmpeg decoder.",
    )


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

    cv2_result = _extract_with_cv2(
        video_path=video_path,
        output_dir=output_dir,
        interval=interval,
        image_format=image_format,
        jpg_quality=jpg_quality,
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
        progress_cb=progress_cb,
    )
    if recovery_result.success:
        return recovery_result

    return VideoExtractionResult(
        video_path=video_path,
        output_dir=output_dir,
        saved_count=0,
        total_frames_seen=0,
        success=False,
        message=f"{imageio_result.message}\n\n{recovery_result.message}",
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
        cmd = [
            str(ffmpeg_exe),
            "-hide_banner",
            "-loglevel",
            "error",
            *FFMPEG_TOLERANT_INPUT_PARAMS,
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(sample_out),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        extracted = list(Path(tmp).glob("sample_*.jpg"))
        return {
            "available": True,
            "returncode": run.returncode,
            "stdout": (run.stdout or "").strip(),
            "stderr": (run.stderr or "").strip(),
            "extracted_first_frame": bool(extracted),
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
