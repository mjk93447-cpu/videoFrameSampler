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
  - OpenCV backends -> ffmpeg pipe fallback (`imageio_ffmpeg`) -> forced/aggressive profiles -> embedded container carving -> repair transcode -> legacy codec bridge (AviSynth/DirectShow) -> raw H264 salvage.
- Enhanced failure logging:
  - On full decode failure, a compact summary is shown in GUI logs.
  - Full stage-by-stage diagnostics are saved to `output/<video_name>/decode_failure_report.txt`.
  - Structured machine-readable diagnostics are saved to `output/<video_name>/decode_failure_report.json`.

## Tech Stack

- GUI: PySide6
- Decoding and processing: OpenCV, imageio-ffmpeg
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
- `probe.imageio`: ffmpeg pipe fallback metadata and first-frame decode result.
- `probe.recovery_ffmpeg`: profile-by-profile ffmpeg recovery attempts.
- `extraction_runs`: repeatable extraction runs for environment/version comparison.

For offline issue sharing, attach these files after one failed run:

- `output/<video_name>/decode_failure_report.txt`
- `output/<video_name>/decode_failure_report.json`
- `decode_diagnostic_report.json` (from CLI)

## Build EXE

```bash
python -m PyInstaller --noconfirm --onefile --windowed --name videoFrameSampler --hidden-import imageio_ffmpeg --copy-metadata imageio-ffmpeg --exclude-module matplotlib --exclude-module tkinter --exclude-module PIL.ImageTk --exclude-module IPython --exclude-module jupyter --exclude-module PySide6.Qt3DAnimation --exclude-module PySide6.Qt3DCore --exclude-module PySide6.Qt3DExtras --exclude-module PySide6.Qt3DInput --exclude-module PySide6.Qt3DLogic --exclude-module PySide6.Qt3DRender --exclude-module PySide6.QtBluetooth --exclude-module PySide6.QtCharts --exclude-module PySide6.QtDataVisualization --exclude-module PySide6.QtDesigner --exclude-module PySide6.QtHelp --exclude-module PySide6.QtMultimedia --exclude-module PySide6.QtMultimediaWidgets --exclude-module PySide6.QtNfc --exclude-module PySide6.QtOpenGL --exclude-module PySide6.QtOpenGLWidgets --exclude-module PySide6.QtPdf --exclude-module PySide6.QtPdfWidgets --exclude-module PySide6.QtPositioning --exclude-module PySide6.QtQml --exclude-module PySide6.QtQuick --exclude-module PySide6.QtQuick3D --exclude-module PySide6.QtQuickControls2 --exclude-module PySide6.QtQuickWidgets --exclude-module PySide6.QtRemoteObjects --exclude-module PySide6.QtScxml --exclude-module PySide6.QtSensors --exclude-module PySide6.QtSerialBus --exclude-module PySide6.QtSerialPort --exclude-module PySide6.QtSpatialAudio --exclude-module PySide6.QtSql --exclude-module PySide6.QtStateMachine --exclude-module PySide6.QtSvg --exclude-module PySide6.QtSvgWidgets --exclude-module PySide6.QtTest --exclude-module PySide6.QtTextToSpeech --exclude-module PySide6.QtUiTools --exclude-module PySide6.QtWebChannel --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineQuick --exclude-module PySide6.QtWebEngineWidgets --exclude-module PySide6.QtWebSockets --exclude-module PySide6.QtWebView --exclude-module PySide6.QtXml --exclude-module PySide6.QtXmlPatterns src/app.py
```

Output:

- `dist/videoFrameSampler.exe`
- Optional release batch: `build_release.bat`

## Documents

- Spec: `docs/specification.md`
- Dev cycle notes: `docs/dev_test_cycles.md`
- Release routine: `docs/release_routine.md`
- Feature matrix and staged verification: `docs/feature_validation_matrix.md`
- CI workflow: `.github/workflows/build-exe.yml`
