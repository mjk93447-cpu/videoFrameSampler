from __future__ import annotations

import importlib.metadata
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.frame_extractor import (  # noqa: E402
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

    assert result.success is False
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
