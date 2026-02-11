# Feature Validation Matrix for Size Optimization

Last updated: 2026-02-11

## Purpose

This document defines a strict, staged process to minimize final EXE size while preserving all functional behavior in `videoFrameSampler`.

## Full Feature Inventory

### A. Input and Job Control

- Select one or many video files from GUI.
- Process videos sequentially with per-file progress logs.
- Batch handling target: up to 500 videos.
- Global extraction controls:
  - Interval (`N` frame sampling)
  - Output format (`PNG` / `JPG`)
  - JPG quality

### B. ROI (Region of Interest)

- First-frame preview load for the selected video.
- ROI selection dialog with rectangle drawing.
- ROI clear/reset behavior.
- ROI checkbox on/off behavior.
- ROI applied to all extraction paths:
  - OpenCV
  - ffmpeg pipe fallback
  - ffmpeg recovery profiles
  - container carving recovery
  - repair-transcoded stream
  - legacy codec bridge
  - raw H264 salvage

### C. Motion Sampling

- Motion detection score per frame.
- One contiguous major segment detection.
- User-defined target duration (seconds).
- User-defined tolerance ratio (percent).
- No frame gaps in saved motion segment.
- Motion segment output path:
  - `output/<video_stem>/motion_segment`

### D. Decode Robustness Pipeline

- OpenCV multi-backend probing (`default`, `CAP_DSHOW`, `CAP_MSMF`, `CAP_FFMPEG`).
- ffmpeg pipe fallback via `imageio_ffmpeg`.
- ffmpeg recovery attempts:
  - tolerant profile
  - forced container formats
  - aggressive decode profiles
- Embedded container carving:
  - signature scanning
  - non-zero offset stream carving
  - frame name normalization
- Repair transcode fallback.
- Legacy codec bridge (AviSynth + DirectShow/VfW).
- Raw H264 Annex-B salvage fallback.

### E. Diagnostics and Reporting

- Decode diagnostics CLI (`src/diagnose_cli.py`).
- Environment + backend probe output JSON.
- Repeatable extraction runs with result logs.

### F. UI/UX and Globalization

- Dark theme layout tuned for 1920x1080 class screens.
- All GUI messages and logs in English.
- Fast mode convenience behavior.

## Staged EXE Size Reduction Strategy

### Stage 0 - Baseline

- Build baseline EXE and record:
  - timestamp
  - commit hash
  - EXE size (bytes)
  - test pass/fail

Command examples:

```powershell
python -m pytest -q
python -m PyInstaller --noconfirm --onefile --windowed --name videoFrameSampler src/app.py
(Get-Item .\dist\videoFrameSampler.exe).Length
```

### Stage 1 - Dependency Pruning

- Remove optional runtime libraries that are not required for user-visible features.
- Replace heavier fallback dependency paths with lighter equivalent implementations.
- Rebuild and compare EXE size delta against Stage 0.

### Stage 2 - Packaging Trim

- Keep onefile mode.
- Keep only required hidden imports and metadata.
- Exclude unused Qt modules and unrelated packages.
- Rebuild and compare EXE size delta against Stage 1.

### Stage 3 - Behavioral Parity Validation

- Run full automated tests.
- Run synthetic extraction validation.
- Run GUI smoke checks.
- Run diagnostics CLI on representative files.
- Confirm no feature regression in the matrix above.

### Stage 4 - Release Gate

Release is allowed only when all are true:

- Tests pass.
- No new lint errors.
- Feature matrix check complete.
- Final EXE size is less than or equal to prior release artifact.
- Build and runtime instructions are updated in `README.md`.

## Verification Checklist (Per Change Batch)

- [ ] `python -m pytest -q`
- [ ] `python tests/generate_test_video.py`
- [ ] `python tests/validate_extraction_outputs.py --output-dir output/<video_name>`
- [ ] ROI extraction manually verified
- [ ] Motion segment extraction manually verified
- [ ] Broken media fallback path manually verified
- [ ] `python src/diagnose_cli.py --video "<path>" --repeat 3`
- [ ] `python build_release.bat`
- [ ] Record EXE size and compare with previous artifact

## Artifact Size Log Template

| Date | Commit | Stage | EXE Size (bytes) | Delta vs Previous | Notes |
| --- | --- | --- | ---: | ---: | --- |
| 2026-02-11 | working-tree | Baseline (before imageio removal) | 141988446 | 0 | onefile + current excludes |
| 2026-02-11 | working-tree | Stage 1 (imageio removed) | 131224333 | -10764113 | fallback migrated to `imageio_ffmpeg` pipe |

