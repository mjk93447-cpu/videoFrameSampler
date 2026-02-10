# videoFrameSampler Development Specification (ver1 target)

## Purpose

Build an offline Windows desktop application that extracts frames from up to 500 input videos and saves them in deterministic order for downstream analysis.

## Functional Requirements

1. GUI allows selecting up to 500 video files in one batch.
2. Videos are processed sequentially in selected order.
3. Frame extraction runs from start to end of each video.
4. Sampling interval is user-configurable:
   - Default: 1 (every frame)
   - Custom: N (save every Nth frame)
5. Output folders are created per video name under application directory:
   - `output/<video_name>/`
6. Saved filename format:
   - `<video_name>_<frame_index_6digits>.<ext>`
7. Image format options:
   - PNG (default, quality-first)
   - JPG (quality configurable)
8. Fast mode is available from GUI:
   - Suggests a larger interval based on source FPS/resolution
   - Uses JPG by default for faster throughput
9. Application works fully offline.

## Non-Functional Requirements

- Stable with large batch inputs (up to 500 files).
- Deterministic naming and ordering for reproducible analysis.
- Clear status feedback in GUI:
  - current video
  - saved frame count
  - completion/failure status
- Error handling for unreadable/corrupt files and write permission failures.

## Data/Path Rules

- Root output directory: `./output` relative to executable location.
- If same video name appears repeatedly, use suffix to avoid collisions:
  - `<video_name>`, `<video_name>__2`, `<video_name>__3`, ...
- File extension:
  - `.png` for PNG
  - `.jpg` for JPG

## UI Requirements

- Main controls:
  - file picker (multi-select)
  - interval input
  - format selector (PNG/JPG)
  - JPG quality slider
  - fast mode toggle + "suggest interval" action
  - start button
- Result panel:
  - log messages
  - progress bar
  - summary text

## Verification Requirements

- Unit tests for frame extraction decisions and output naming.
- Synthetic test video generation for reproducible integration tests.
- A quality/performance evaluation script producing:
  - output frame count and order check
  - timing metrics
  - optional image quality metrics for JPG (PSNR/SSIM approximation)

## Release Routine Requirement

Every completed result version must run this routine:

1. run tests
2. update docs and version label
3. commit and push
4. trigger GitHub Actions Windows EXE build
5. confirm artifact and record in release notes
