# videoFrameSampler

Windows desktop application for offline video frame extraction with ROI and motion-focused sampling.

## Main Features

- Batch processing up to 500 videos.
- ROI extraction:
  - Load first-frame preview from the selected video.
  - Draw an ROI box in the GUI.
  - Save only cropped ROI frames for performance and storage efficiency.
- Motion segment sampling:
  - Detect frame-to-frame motion intensity.
  - Auto-select one major continuous segment near the requested duration.
  - User-adjustable tolerance range (percent) for segment length.
  - Save contiguous frames into `output/<video_name>/motion_segment`.
- Standard interval extraction:
  - Custom interval (`N`), output format (`PNG`/`JPG`), JPG quality.
- Fast mode:
  - Auto-select JPG and suggest interval from first video metadata.
- Robust decode fallback pipeline:
  - OpenCV backends -> imageio ffmpeg -> forced format profiles -> aggressive profiles -> repair transcode -> legacy codec bridge (AviSynth/DirectShow) -> raw H264 salvage.

## Tech Stack

- GUI: PySide6
- Decoding and processing: OpenCV, imageio/imageio-ffmpeg
- Packaging: PyInstaller
- Tests: pytest

## Run Locally

```bash
python -m pip install -r requirements.txt
python src/app.py
```

## Test

```bash
python -m pytest -q
```

Generate synthetic test video:

```bash
python tests/generate_test_video.py
```

Validate extraction output:

```bash
python tests/validate_extraction_outputs.py --output-dir output/<video_name>
```

## Decode Diagnostics CLI

```bash
python src/diagnose_cli.py --video "D:\Non_Documents\SDV_NG_01.avi" --interval 1 --format jpg --jpg-quality 95 --repeat 3 --report-json decode_diagnostic_report.json
```

- `probe.cv2`: open/read result per OpenCV backend.
- `probe.imageio`: fallback ffmpeg metadata and first-frame decode result.
- `probe.recovery_ffmpeg`: profile-by-profile ffmpeg recovery attempts.
- `extraction_runs`: repeatable extraction runs for environment/version comparison.

## Build EXE

```bash
python -m PyInstaller --noconfirm --onefile --windowed --name videoFrameSampler --hidden-import imageio --hidden-import imageio_ffmpeg --copy-metadata imageio --copy-metadata imageio-ffmpeg --exclude-module matplotlib --exclude-module tkinter --exclude-module PIL.ImageTk --exclude-module IPython --exclude-module jupyter src/app.py
```

Output:

- `dist/videoFrameSampler.exe`
- Optional release batch: `build_release.bat`

## Documents

- Spec: `docs/specification.md`
- Dev cycle notes: `docs/dev_test_cycles.md`
- Release routine: `docs/release_routine.md`
- CI workflow: `.github/workflows/build-exe.yml`
