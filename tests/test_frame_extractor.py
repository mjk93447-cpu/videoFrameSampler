from __future__ import annotations

import importlib.metadata
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
    _extract_with_ffmpeg_recovery,
    collect_decode_diagnostics,
    extract_video_frames,
    make_unique_output_dir,
    suggest_fast_mode_interval,
)
import core.frame_extractor as frame_extractor  # noqa: E402
from core.models import ExtractionOptions, ImageFormat  # noqa: E402
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

    class FakeReader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            for _ in range(3):
                frame = np.zeros((60, 90, 3), dtype=np.uint8)
                frame[:, :, 0] = 120
                yield frame

    class FakeImageio:
        @staticmethod
        def get_reader(path: str, format: str, input_params: list[str]):
            captured["path"] = path
            captured["format"] = format
            captured["input_params"] = input_params
            return FakeReader()

    def fake_import_module(name: str):
        if name == "imageio":
            return FakeImageio
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
    assert captured["format"] == "ffmpeg"
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
        if name == "imageio":
            raise importlib.metadata.PackageNotFoundError("imageio")
        return real_import_module(name)

    monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(frame_extractor.importlib, "import_module", fake_import_module)

    try:
        result = extract_video_frames(video_path, options)
    finally:
        monkeypatch.setattr(frame_extractor.cv2, "VideoCapture", real_video_capture)
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)

    # imageio import failure should not crash extraction flow.
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
        if name == "imageio":
            raise AssertionError("imageio fallback should not be used when alternate cv2 backend works")
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
            progress_cb=None,
        )
    finally:
        monkeypatch.setattr(frame_extractor.importlib, "import_module", real_import_module)
        monkeypatch.setattr(frame_extractor.subprocess, "run", real_subprocess_run)

    assert "asf" in FFMPEG_FORCED_INPUT_FORMATS
    assert result.success is True
    assert result.saved_count == 1
    assert "recovery ffmpeg decoder (asf)" in result.message.lower()
