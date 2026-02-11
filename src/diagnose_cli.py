from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.frame_extractor import collect_decode_diagnostics
from core.models import ExtractionOptions, ImageFormat


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precision decode diagnostics for version/backend issues.",
    )
    parser.add_argument("--video", required=True, help="Target video file path")
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval")
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output image format for extraction test",
    )
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPG quality (1-100)")
    parser.add_argument("--repeat", type=int, default=1, help="Number of extraction repeats")
    parser.add_argument(
        "--report-json",
        default="decode_diagnostic_report.json",
        help="Output report path (json)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    image_format = ImageFormat.JPG if args.format == "jpg" else ImageFormat.PNG
    options = ExtractionOptions(
        interval=args.interval,
        image_format=image_format,
        jpg_quality=args.jpg_quality,
    )
    report = collect_decode_diagnostics(
        video_path=Path(args.video),
        options=options,
        repeat=args.repeat,
    )

    report_path = Path(args.report_json)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    failures = [run for run in report.get("extraction_runs", []) if not run.get("success")]
    print(f"[diagnostic] report: {report_path.resolve()}")
    print(f"[diagnostic] runs: {len(report.get('extraction_runs', []))}, failures: {len(failures)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
