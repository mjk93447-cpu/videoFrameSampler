from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.frame_extractor import (  # noqa: E402
    CV2_BACKEND_ORDER,
    FFMPEG_FORCED_INPUT_FORMATS,
    FFMPEG_RELAXED_INPUT_PARAMS,
    _extract_with_ffmpeg_recovery,
    _extract_with_h264_salvage,
    collect_decode_diagnostics,
    extract_video_frames,
    make_unique_output_dir,
    suggest_fast_mode_interval,
)
import core.frame_extractor as frame_extractor  # noqa: E402
from core.models import ExtractionOptions, ImageFormat, MotionSamplingOptions, RoiBox, VideoExtractionResult  # noqa: E402
from tests.generate_test_video import create_synthetic_video  # noqa: E402


def _clean_output_dir() -> None:
    out = Path.cwd() / "output"
    if out.exists():
        shutil.rmtree(out)


def _read_image_unicode_safe(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def test_extract_video_frames_interval_and_naming(tmp_path: Path) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "sample.mp4", frame_count=24)
    options = ExtractionOptions(interval=3, image_format=ImageFormat.PNG)

    result = extract_video_frames(video_path, options)

    assert result.success is True
    assert result.saved_count == 8
    assert result.output_dir.exists()
    files = sorted(result.output_dir.glob("*.png"))
    assert len(files) == 8
    assert files[0].name == "sample_000000.png"
    assert files[-1].name == "sample_000007.png"

    img = _read_image_unicode_safe(files[0])
    assert img is not None


def test_make_unique_output_dir_suffix(tmp_path: Path) -> None:
    root = tmp_path / "output"
    first = make_unique_output_dir("dup_video", root)
    second = make_unique_output_dir("dup_video", root)

    assert first.name == "dup_video"
    assert second.name == "dup_video__2"


def test_suggest_fast_mode_interval_is_positive(tmp_path: Path) -> None:
    video_path = create_synthetic_video(tmp_path / "speed.mp4", width=1920, height=1080, fps=30)
    suggestion = suggest_fast_mode_interval(video_path)
    assert suggestion >= 1


@pytest.mark.parametrize(
    "filename,fourcc",
    [
        ("sample_mp4.mp4", "mp4v"),
        ("sample_avi.avi", "MJPG"),
    ],
)
def test_extract_video_frames_various_containers(tmp_path: Path, filename: str, fourcc: str) -> None:
    _clean_output_dir()
    try:
        video_path = create_synthetic_video(tmp_path / filename, frame_count=20, fourcc=fourcc)
    except RuntimeError as exc:
        pytest.skip(f"Codec not available in this environment: {exc}")

    options = ExtractionOptions(interval=2, image_format=ImageFormat.JPG, jpg_quality=90)
    result = extract_video_frames(video_path, options)
    assert result.success is True
    assert result.saved_count == 10
    assert len(list(result.output_dir.glob("*.jpg"))) == 10


def test_extract_video_frames_uses_fallback_when_cv2_cannot_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "fallback.mp4", frame_count=12, fourcc="mp4v")
    options = ExtractionOptions(interval=3, image_format=ImageFormat.PNG)

    real_video_capture = frame_extractor.cv2.VideoCapture

    class FakeCapture:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", FakeCapture)
    try:
        result = extract_video_frames(video_path, options)
    finally:
        monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", real_video_capture)

    assert result.success is True
    assert result.saved_count == 4
    assert "fallback" in result.message.lower()


def test_fallback_reader_uses_tolerant_ffmpeg_input_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "fallback_tolerant.mp4", frame_count=8, fourcc="mp4v")
    options = ExtractionOptions(interval=2, image_format=ImageFormat.PNG)

    real_video_capture = frame_extractor.cv2.VideoCapture
    real_import_module = frame_extractor.importlib.import_module
    captured: dict[str, object] = {}

    class FakeCapture:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    class FakeImageioFfmpeg:
        @staticmethod
        def read_frames(path: str, input_params: list[str]):
            captured["path"] = path
            captured["input_params"] = input_params
            yield {"size": (90, 60), "fps": 30}
            for _ in range(3):
                frame = np.zeros((60, 90, 3), dtype=np.uint8)
                frame[:, :, 0] = 120
                yield frame.tobytes()

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    try:
        result = extract_video_frames(video_path, options)
    finally:
        monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", real_video_capture)
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)

    assert result.success is True
    assert result.saved_count == 2
    assert captured["path"] == str(video_path)
    assert captured["input_params"] == frame_extractor.FFMPEG_TOLERANT_INPUT_PARAMS


def test_fallback_unavailable_does_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "fallback_missing.mp4", frame_count=8, fourcc="mp4v")
    options = ExtractionOptions(interval=2, image_format=ImageFormat.PNG)

    real_video_capture = frame_extractor.cv2.VideoCapture
    real_import_module = frame_extractor.importlib.import_module

    class FakeCapture:
        def __init__(self, *_args, **_kwargs):
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            return None

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            raise importlib.metadata.PackageNotFoundError("imageio_ffmpeg")
        return real_import_module(name)

    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)

    try:
        result = extract_video_frames(video_path, options)
    finally:
        monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", real_video_capture)
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)

    # ffmpeg pipe fallback import failure should not crash extraction flow.
    # If recovery ffmpeg backend is available, extraction can still succeed.
    assert isinstance(result.success, bool)
    if result.success:
        assert "recovery" in result.message.lower()
    else:
        assert "fallback backend is unavailable" in result.message.lower()


def test_extract_video_tries_alternate_cv2_backend_before_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "backend_try.mp4", frame_count=6, fourcc="mp4v")
    options = ExtractionOptions(interval=2, image_format=ImageFormat.PNG)

    real_video_capture = frame_extractor.cv2.VideoCapture
    real_import_module = frame_extractor.importlib.import_module

    class FakeCapture:
        def __init__(self, *_args, **_kwargs):
            self.use_backend = len(_args) >= 2
            self.frames_left = 6 if self.use_backend else 0

        def isOpened(self) -> bool:
            return self.use_backend

        def read(self):
            if self.frames_left <= 0:
                return False, None
            self.frames_left -= 1
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:, :, 1] = 180
            return True, frame

        def release(self) -> None:
            return None

    def fail_if_import_imageio(name: str):
        if name == "imageio_ffmpeg":
            raise AssertionError("ffmpeg fallback should not be used when alternate cv2 backend works")
        return real_import_module(name)

    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(frame_extractor.importlib, "import_module", fail_if_import_imageio)

    try:
        result = extract_video_frames(video_path, options)
    finally:
        monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", real_video_capture)
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)

    assert result.success is True
    assert result.saved_count == 3
    assert "opencv backend" in result.message.lower()


def test_ffmpeg_recovery_extracts_frames_when_fallback_decoder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "broken_like.avi"
    video_path.write_bytes(b"dummy")

    real_import_module = frame_extractor.importlib.import_module
    real_subprocess_run = frame_extractor.subprocess.run

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    def fake_run(cmd, capture_output, text, encoding, errors):
        pattern = Path(cmd[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)
        for idx in range(3):
            img = np.zeros((80, 120, 3), dtype=np.uint8)
            img[:, :, 2] = idx * 40
            ok, encoded = cv2.imencode(".png", img)
            assert ok
            encoded.tofile(str(pattern.parent / f"frame_{idx:06d}.png"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    try:
        result = _extract_with_ffmpeg_recovery(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert result.success is True
    assert result.saved_count == 3
    files = sorted(out_dir.glob("*.png"))
    assert len(files) == 3
    assert files[0].name == "broken_like_000000.png"


def test_cv2_backend_order_prioritizes_windows_legacy_decoders() -> None:
    assert CV2_BACKEND_ORDER[0] == "default"
    assert "CAP_DSHOW" in CV2_BACKEND_ORDER
    assert "CAP_MSMF" in CV2_BACKEND_ORDER
    assert CV2_BACKEND_ORDER.index("CAP_DSHOW") < CV2_BACKEND_ORDER.index("CAP_FFMPEG")
    assert CV2_BACKEND_ORDER.index("CAP_MSMF") < CV2_BACKEND_ORDER.index("CAP_FFMPEG")


def test_collect_decode_diagnostics_reports_repeat_runs(tmp_path: Path) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "diag.mp4", frame_count=10, fourcc="mp4v")
    options = ExtractionOptions(interval=2, image_format=ImageFormat.PNG)

    report = collect_decode_diagnostics(video_path=video_path, options=options, repeat=2)

    assert "environment" in report
    assert "probe" in report
    runs = report["extraction_runs"]
    assert len(runs) == 2
    assert all("progress_log" in run for run in runs)
    assert all("success" in run for run in runs)


def test_ffmpeg_recovery_tries_forced_input_formats_when_auto_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out_forced"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "legacy_like.avi"
    video_path.write_bytes(b"dummy")

    real_import_module = frame_extractor.importlib.import_module
    real_subprocess_run = frame_extractor.subprocess.run

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    def fake_run(cmd, capture_output, text, encoding, errors):
        pattern = Path(cmd[-1])
        use_forced_asf = "-f" in cmd and "asf" in cmd
        if use_forced_asf:
            pattern.parent.mkdir(parents=True, exist_ok=True)
            img = np.zeros((80, 120, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", img)
            assert ok
            encoded.tofile(str(pattern.parent / "frame_000000.png"))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "invalid data")

    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    try:
        result = _extract_with_ffmpeg_recovery(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert "asf" in FFMPEG_FORCED_INPUT_FORMATS
    assert result.success is True
    assert result.saved_count == 1
    assert "recovery ffmpeg decoder (forced_asf)" in result.message.lower()


def test_ffmpeg_recovery_uses_aggressive_profile_when_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out_aggressive"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "corrupt_like.avi"
    video_path.write_bytes(b"dummy")

    real_import_module = frame_extractor.importlib.import_module
    real_subprocess_run = frame_extractor.subprocess.run

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    def fake_run(cmd, capture_output, text, encoding, errors):
        pattern = Path(cmd[-1])
        uses_aggressive = "-max_error_rate" in cmd and "1" in cmd and "-f" in cmd and "h264" in cmd
        if uses_aggressive:
            pattern.parent.mkdir(parents=True, exist_ok=True)
            img = np.zeros((80, 120, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", img)
            assert ok
            encoded.tofile(str(pattern.parent / "frame_000000.png"))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "invalid data")

    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    try:
        result = _extract_with_ffmpeg_recovery(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert "-fflags" in FFMPEG_RELAXED_INPUT_PARAMS
    assert result.success is True
    assert result.saved_count == 1
    assert "aggressive_h264" in result.message.lower()


def test_h264_salvage_extracts_frames_from_annexb_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out_salvage"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "wrapped_corrupt.avi"
    video_path.write_bytes(b"\x11\x22\x33\x44\x00\x00\x00\x01\x67\x64\x00\x1f\x00\x00\x01\x68\xee\x3c\x80")

    real_import_module = frame_extractor.importlib.import_module
    real_subprocess_run = frame_extractor.subprocess.run

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    def fake_run(cmd, capture_output, text, encoding, errors):
        pattern = Path(cmd[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        assert ok
        encoded.tofile(str(pattern.parent / "salvage_000000.png"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    try:
        result = _extract_with_h264_salvage(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert result.success is True
    assert result.saved_count == 1
    assert "raw h264 salvage" in result.message.lower()


def test_h264_salvage_tries_multiple_candidate_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out_salvage_multi"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "wrapped_multi.avi"
    # First NAL is non-SPS (likely bad start), second is SPS.
    video_path.write_bytes(
        b"\x00\x00\x00\x01\x61\x11\x22\x33"
        b"\x00\x00\x00\x01\x67\x64\x00\x1f\x00\x00\x01\x68\xee\x3c\x80"
    )

    real_import_module = frame_extractor.importlib.import_module
    real_subprocess_run = frame_extractor.subprocess.run

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    seen_prefixes: list[int] = []
    run_count = {"n": 0}

    def fake_run(cmd, capture_output, text, encoding, errors):
        input_path = Path(cmd[cmd.index("-i") + 1])
        payload = input_path.read_bytes()
        # Record first NAL type for assertion.
        first_nal_type = int(payload[4] & 0x1F) if len(payload) > 4 and payload[:4] == b"\x00\x00\x00\x01" else -1
        seen_prefixes.append(first_nal_type)

        pattern = Path(cmd[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)
        run_count["n"] += 1
        # Force one retry, then succeed on the next candidate.
        if run_count["n"] == 1:
            return subprocess.CompletedProcess(cmd, 1, "", "invalid data")
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        assert ok
        encoded.tofile(str(pattern.parent / "salvage_000000.png"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    try:
        result = frame_extractor._extract_with_h264_salvage(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert result.success is True
    assert result.saved_count == 1
    # Ensure at least two candidates were tried.
    assert len(seen_prefixes) >= 2


def test_h264_salvage_uses_sps_pps_reconstruction_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out_salvage_rebuild"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "wrapped_rebuild.avi"
    # Include SPS/PPS in payload so reconstruction route can prepend/inject.
    video_path.write_bytes(
        b"\x00\x00\x00\x01\x61\x11\x22\x33"
        b"\x00\x00\x00\x01\x67\x64\x00\x1f"
        b"\x00\x00\x00\x01\x68\xee\x3c\x80"
        b"\x00\x00\x00\x01\x65\x88\x99\xaa"
    )

    real_import_module = frame_extractor.importlib.import_module
    real_subprocess_run = frame_extractor.subprocess.run

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    attempts = {"raw": 0, "rebuild": 0}

    def fake_run(cmd, capture_output, text, encoding, errors):
        input_path = Path(cmd[cmd.index("-i") + 1])
        pattern = Path(cmd[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)

        if "_salvage_rebuild_" in input_path.name:
            attempts["rebuild"] += 1
            img = np.zeros((80, 120, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", img)
            assert ok
            encoded.tofile(str(pattern.parent / "salvage_000000.png"))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        attempts["raw"] += 1
        return subprocess.CompletedProcess(cmd, 1, "", "non-existing PPS")

    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    try:
        result = frame_extractor._extract_with_h264_salvage(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert result.success is True
    assert result.saved_count == 1
    assert "rebuild_" in result.message
    assert attempts["raw"] >= 1
    assert attempts["rebuild"] >= 1


def test_extract_video_frames_applies_roi_crop(tmp_path: Path) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "roi_source.mp4", frame_count=8, width=200, height=120, fourcc="mp4v")
    options = ExtractionOptions(
        interval=2,
        image_format=ImageFormat.PNG,
        roi=RoiBox(x=20, y=10, width=80, height=50),
    )
    result = extract_video_frames(video_path, options)
    assert result.success is True
    files = sorted(result.output_dir.glob("*.png"))
    assert files
    img = _read_image_unicode_safe(files[0])
    assert img is not None
    assert img.shape[1] == 80
    assert img.shape[0] == 50


def test_extract_video_frames_saves_motion_segment(tmp_path: Path) -> None:
    _clean_output_dir()
    video_path = create_synthetic_video(tmp_path / "motion_source.mp4", frame_count=36, width=240, height=140, fps=12, fourcc="mp4v")
    options = ExtractionOptions(
        interval=3,
        image_format=ImageFormat.JPG,
        motion_sampling=MotionSamplingOptions(enabled=True, expected_duration_sec=1.5),
    )
    result = extract_video_frames(video_path, options)
    assert result.success is True
    motion_dir = result.output_dir / "motion_segment"
    files = sorted(motion_dir.glob("*.jpg"))
    assert files
    assert len(files) >= 1


def test_extract_video_frames_uses_repair_pipeline_when_recovery_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_output_dir()
    video_path = tmp_path / "broken_input.avi"
    video_path.write_bytes(b"broken")
    options = ExtractionOptions(interval=2, image_format=ImageFormat.PNG)

    real_cv2_extract = frame_extractor._extract_with_cv2
    real_imageio_extract = frame_extractor._extract_with_imageio_fallback
    real_recovery_extract = frame_extractor._extract_with_ffmpeg_recovery
    real_repair = frame_extractor._repair_video_with_ffmpeg
    real_legacy_bridge = frame_extractor._extract_with_legacy_codec_bridge
    real_salvage = frame_extractor._extract_with_h264_salvage

    state = {"cv2_calls": 0}

    def fake_cv2(video_path, output_dir, interval, image_format, jpg_quality, roi, motion_sampling, progress_cb):
        state["cv2_calls"] += 1
        if state["cv2_calls"] == 1:
            return None
        return VideoExtractionResult(
            video_path=Path(video_path),
            output_dir=output_dir,
            saved_count=2,
            total_frames_seen=20,
            success=True,
            message="Repaired decode success.",
        )

    def fake_imageio(*_args, **_kwargs):
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=Path.cwd() / "output" / "broken_input",
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="fallback failed",
        )

    def fake_recovery(*_args, **_kwargs):
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=Path.cwd() / "output" / "broken_input",
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="recovery failed",
        )

    def fake_repair(video_path, output_dir, progress_cb):
        repaired_path = output_dir / "__repaired_video_tmp" / "broken_input__repaired.mp4"
        repaired_path.parent.mkdir(parents=True, exist_ok=True)
        repaired_path.write_bytes(b"fake repaired")
        return repaired_path, "Repaired video created with profile repair_aggressive_auto."

    def fake_salvage(*_args, **_kwargs):
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=Path.cwd() / "output" / "broken_input",
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="salvage failed",
        )

    def fake_legacy_bridge(*_args, **_kwargs):
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=Path.cwd() / "output" / "broken_input",
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message="legacy failed",
        )

    monkeypatch.setattr(frame_extractor, "_extract_with_cv2", fake_cv2)
    monkeypatch.setattr(frame_extractor, "_extract_with_imageio_fallback", fake_imageio)
    monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", fake_recovery)
    monkeypatch.setattr(frame_extractor, "_repair_video_with_ffmpeg", fake_repair)
    monkeypatch.setattr(frame_extractor, "_extract_with_legacy_codec_bridge", fake_legacy_bridge)
    monkeypatch.setattr(frame_extractor, "_extract_with_h264_salvage", fake_salvage)
    try:
        result = frame_extractor.extract_video_frames(video_path, options)
    finally:
        monkeypatch.setattr(frame_extractor, "_extract_with_cv2", real_cv2_extract)
        monkeypatch.setattr(frame_extractor, "_extract_with_imageio_fallback", real_imageio_extract)
        monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", real_recovery_extract)
        monkeypatch.setattr(frame_extractor, "_repair_video_with_ffmpeg", real_repair)
        monkeypatch.setattr(frame_extractor, "_extract_with_legacy_codec_bridge", real_legacy_bridge)
        monkeypatch.setattr(frame_extractor, "_extract_with_h264_salvage", real_salvage)

    assert result.success is True
    assert result.saved_count == 2
    assert "repaired video created" in result.message.lower()


def test_legacy_codec_bridge_extracts_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "legacy_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "legacy_input.avi"
    video_path.write_bytes(b"dummy")

    real_import_module = frame_extractor.importlib.import_module
    real_subprocess_run = frame_extractor.subprocess.run

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe() -> str:
            return "ffmpeg"

    def fake_import_module(name: str):
        if name == "imageio_ffmpeg":
            return FakeImageioFfmpeg
        return real_import_module(name)

    captured_script = {"text": ""}

    def fake_run(cmd, capture_output, text, encoding, errors):
        script_path = Path(cmd[cmd.index("-i") + 1])
        captured_script["text"] = script_path.read_text(encoding="utf-8")
        pattern = Path(cmd[-1])
        pattern.parent.mkdir(parents=True, exist_ok=True)
        img = np.zeros((72, 128, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        assert ok
        encoded.tofile(str(pattern.parent / "legacy_000000.png"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    try:
        result = frame_extractor._extract_with_legacy_codec_bridge(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert result.success is True
    assert result.saved_count == 1
    assert "legacy codec bridge" in result.message.lower()
    assert 'AviSource("' in captured_script["text"]
    assert 'AviSource(r"' not in captured_script["text"]
    assert "\\" not in captured_script["text"]


def test_extract_video_frames_writes_compact_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_output_dir()
    video_path = tmp_path / "always_fail.avi"
    video_path.write_bytes(b"broken")
    options = ExtractionOptions(interval=2, image_format=ImageFormat.PNG)

    real_cv2_extract = frame_extractor._extract_with_cv2
    real_imageio_extract = frame_extractor._extract_with_imageio_fallback
    real_recovery_extract = frame_extractor._extract_with_ffmpeg_recovery
    real_repair = frame_extractor._repair_video_with_ffmpeg
    real_legacy_bridge = frame_extractor._extract_with_legacy_codec_bridge
    real_salvage = frame_extractor._extract_with_h264_salvage

    def fake_cv2(*_args, **_kwargs):
        return None

    def fake_fail_message(label: str):
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=Path.cwd() / "output" / "always_fail",
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"{label} failed: sample detailed message",
        )

    monkeypatch.setattr(frame_extractor, "_extract_with_cv2", fake_cv2)
    monkeypatch.setattr(frame_extractor, "_extract_with_imageio_fallback", lambda *_a, **_k: fake_fail_message("fallback"))
    monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", lambda *_a, **_k: fake_fail_message("recovery"))
    monkeypatch.setattr(frame_extractor, "_repair_video_with_ffmpeg", lambda *_a, **_k: (None, "Repair failed: sample detailed message"))
    monkeypatch.setattr(frame_extractor, "_extract_with_legacy_codec_bridge", lambda *_a, **_k: fake_fail_message("legacy"))
    monkeypatch.setattr(frame_extractor, "_extract_with_h264_salvage", lambda *_a, **_k: fake_fail_message("salvage"))

    try:
        result = frame_extractor.extract_video_frames(video_path, options)
    finally:
        monkeypatch.setattr(frame_extractor, "_extract_with_cv2", real_cv2_extract)
        monkeypatch.setattr(frame_extractor, "_extract_with_imageio_fallback", real_imageio_extract)
        monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", real_recovery_extract)
        monkeypatch.setattr(frame_extractor, "_repair_video_with_ffmpeg", real_repair)
        monkeypatch.setattr(frame_extractor, "_extract_with_legacy_codec_bridge", real_legacy_bridge)
        monkeypatch.setattr(frame_extractor, "_extract_with_h264_salvage", real_salvage)

    assert result.success is False
    assert "All decode pipelines failed" in result.message
    assert "Detailed report:" in result.message
    assert "Structured report:" in result.message
    report_path = result.output_dir / "decode_failure_report.txt"
    json_report_path = result.output_dir / "decode_failure_report.json"
    assert report_path.exists()
    assert json_report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "stage_summary:" in report_text
    assert "[recovery]" in report_text
    json_payload = json.loads(json_report_path.read_text(encoding="utf-8"))
    assert "stage_details" in json_payload
    assert "hints" in json_payload


def test_extract_video_frames_emits_full_failure_diagnostics_to_progress_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_output_dir()
    video_path = tmp_path / "always_fail_progress.avi"
    video_path.write_bytes(b"broken")
    options = ExtractionOptions(interval=2, image_format=ImageFormat.PNG)
    progress_log: list[str] = []

    real_cv2_extract = frame_extractor._extract_with_cv2
    real_imageio_extract = frame_extractor._extract_with_imageio_fallback
    real_recovery_extract = frame_extractor._extract_with_ffmpeg_recovery
    real_repair = frame_extractor._repair_video_with_ffmpeg
    real_legacy_bridge = frame_extractor._extract_with_legacy_codec_bridge
    real_salvage = frame_extractor._extract_with_h264_salvage

    def fake_cv2(*_args, **_kwargs):
        return None

    def fake_fail_message(label: str):
        return VideoExtractionResult(
            video_path=video_path,
            output_dir=Path.cwd() / "output" / "always_fail_progress",
            saved_count=0,
            total_frames_seen=0,
            success=False,
            message=f"{label} failed: sample detailed message",
        )

    monkeypatch.setattr(frame_extractor, "_extract_with_cv2", fake_cv2)
    monkeypatch.setattr(frame_extractor, "_extract_with_imageio_fallback", lambda *_a, **_k: fake_fail_message("fallback"))
    monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", lambda *_a, **_k: fake_fail_message("recovery"))
    monkeypatch.setattr(frame_extractor, "_repair_video_with_ffmpeg", lambda *_a, **_k: (None, "Repair failed: sample detailed message"))
    monkeypatch.setattr(frame_extractor, "_extract_with_legacy_codec_bridge", lambda *_a, **_k: fake_fail_message("legacy"))
    monkeypatch.setattr(frame_extractor, "_extract_with_h264_salvage", lambda *_a, **_k: fake_fail_message("salvage"))

    try:
        _ = frame_extractor.extract_video_frames(video_path, options, progress_cb=progress_log.append)
    finally:
        monkeypatch.setattr(frame_extractor, "_extract_with_cv2", real_cv2_extract)
        monkeypatch.setattr(frame_extractor, "_extract_with_imageio_fallback", real_imageio_extract)
        monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", real_recovery_extract)
        monkeypatch.setattr(frame_extractor, "_repair_video_with_ffmpeg", real_repair)
        monkeypatch.setattr(frame_extractor, "_extract_with_legacy_codec_bridge", real_legacy_bridge)
        monkeypatch.setattr(frame_extractor, "_extract_with_h264_salvage", real_salvage)

    assert any("[error][summary] fallback:" in line for line in progress_log)
    assert any("[error][detail:recovery] begin" in line for line in progress_log)
    assert any("[error] ----- decode failure diagnostics end -----" in line for line in progress_log)


def test_scan_container_offsets_detects_embedded_signatures() -> None:
    raw = (
        b"JUNK" * 8
        + b"\x30\x26\xB2\x75\x8E\x66\xCF\x11\xA6\xD9\x00\xAA\x00\x62\xCE\x6C"
        + b"xxxx"
        + b"RIFF" + b"\x20\x00\x00\x00" + b"AVI "
    )
    found = frame_extractor._scan_container_offsets(raw)
    labels = [item[1] for item in found]
    assert "asf_guid" in labels
    assert "riff_avi" in labels


def test_extract_with_container_carving_recovers_embedded_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "carve_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "wrapped_media.bin"
    # leading garbage + RIFF AVI signature
    video_path.write_bytes(b"X" * 64 + b"RIFF" + b"\x20\x00\x00\x00" + b"AVI " + b"Y" * 128)

    real_recovery = frame_extractor._extract_with_ffmpeg_recovery
    calls = {"count": 0}

    def fake_recovery(video_path, output_dir, interval, image_format, jpg_quality, roi, progress_cb):
        calls["count"] += 1
        out = output_dir / f"{Path(video_path).stem}_000000.png"
        img = np.zeros((50, 60, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        assert ok
        encoded.tofile(str(out))
        return VideoExtractionResult(
            video_path=Path(video_path),
            output_dir=output_dir,
            saved_count=1,
            total_frames_seen=-1,
            success=True,
            message="carved success",
        )

    monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", fake_recovery)
    try:
        result = frame_extractor._extract_with_container_carving(
            video_path=video_path,
            output_dir=out_dir,
            interval=2,
            image_format=ImageFormat.PNG,
            jpg_quality=95,
            roi=None,
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor, "_extract_with_ffmpeg_recovery", real_recovery)

    assert calls["count"] >= 1
    assert result.success is True
    assert result.saved_count == 1
    assert (out_dir / "wrapped_media_000000.png").exists()
