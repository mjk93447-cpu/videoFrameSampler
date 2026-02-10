from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class EvalResult:
    frame_count: int
    elapsed_seconds: float
    throughput_fps: float
    psnr: float | None
    ssim_like: float | None


def calc_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return 99.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calc_ssim_like(img_a: np.ndarray, img_b: np.ndarray) -> float:
    # Lightweight SSIM approximation for quick regression checks.
    a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = ((a - mu_a) ** 2).mean()
    sigma_b = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def evaluate_output_dir(output_dir: Path, reference_dir: Path | None = None) -> EvalResult:
    started = time.perf_counter()
    frames = sorted(output_dir.glob("*.png")) + sorted(output_dir.glob("*.jpg"))
    elapsed = max(1e-9, time.perf_counter() - started)
    count = len(frames)
    throughput = count / elapsed

    if not frames or reference_dir is None:
        return EvalResult(count, elapsed, throughput, None, None)

    references = sorted(reference_dir.glob("*.png")) + sorted(reference_dir.glob("*.jpg"))
    if not references:
        return EvalResult(count, elapsed, throughput, None, None)

    pairs = min(len(frames), len(references))
    psnr_scores: list[float] = []
    ssim_scores: list[float] = []

    for idx in range(pairs):
        img_a = cv2.imread(str(frames[idx]))
        img_b = cv2.imread(str(references[idx]))
        if img_a is None or img_b is None:
            continue
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        psnr_scores.append(calc_psnr(img_a, img_b))
        ssim_scores.append(calc_ssim_like(img_a, img_b))

    return EvalResult(
        frame_count=count,
        elapsed_seconds=elapsed,
        throughput_fps=throughput,
        psnr=sum(psnr_scores) / len(psnr_scores) if psnr_scores else None,
        ssim_like=sum(ssim_scores) / len(ssim_scores) if ssim_scores else None,
    )


if __name__ == "__main__":
    target = Path("output")
    result = evaluate_output_dir(target)
    print(result)
