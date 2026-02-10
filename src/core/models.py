from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ImageFormat(str, Enum):
    PNG = "png"
    JPG = "jpg"


@dataclass(slots=True)
class ExtractionOptions:
    interval: int = 1
    image_format: ImageFormat = ImageFormat.PNG
    jpg_quality: int = 95
    fast_mode: bool = False

    def normalized_interval(self) -> int:
        return max(1, int(self.interval))

    def normalized_jpg_quality(self) -> int:
        return min(100, max(1, int(self.jpg_quality)))


@dataclass(slots=True)
class VideoExtractionResult:
    video_path: Path
    output_dir: Path
    saved_count: int
    total_frames_seen: int
    success: bool
    message: str
