from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_synthetic_video(
    output_path: Path,
    frame_count: int = 24,
    width: int = 320,
    height: int = 180,
    fps: int = 12,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (idx * 9) % 255
        frame[:, :, 1] = (idx * 13) % 255
        frame[:, :, 2] = (idx * 17) % 255
        cv2.putText(
            frame,
            f"F{idx:03d}",
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    return output_path


if __name__ == "__main__":
    create_synthetic_video(Path("tests_artifacts/smoke.mp4"))
