# Dev/Test Improvement Cycles (ver1)

## Scope

- Dataset: synthetic videos from `tests/generate_test_video.py` + local real sample videos
- Priority: correctness > stability > quality > speed > UX

## Scoring Model (100 points)

- Correctness (40): frame count/order/name convention
- Stability (20): failure handling, corrupted file recovery
- Quality (15): PNG pixel parity, JPG PSNR/SSIM-like threshold
- Performance (15): throughput and elapsed time
- UX (10): action count, clarity of progress/error feedback

## Iteration Logs (minimum 5 cycles)

| Cycle | Change Set | Key Measurements | Score | Decision |
|---|---|---|---:|---|
| 1 | Baseline extractor + file picker + interval input | Correctness pass for interval=1/3, limited feedback | 71 | Keep baseline and improve visibility |
| 2 | Added progress bar + sequential status logs + summary output | Better clarity, still weak JPG controls | 79 | Add format/quality controls |
| 3 | Added PNG/JPG switch, JPG quality slider, deterministic naming checks | Quality control improved, deterministic output stable | 86 | Add fast mode interval suggestion |
| 4 | Added fast mode + interval recommendation based on FPS/resolution | Faster large-video processing, acceptable quality trade-off | 90 | Harden error handling |
| 5 | Added per-video exception handling and GUI warnings for invalid input | Recovery and user guidance improved; stable completion | 93 | Freeze as ver1 candidate |

## Frozen ver1 Configuration

- Default mode: PNG, interval=1
- Optional fast mode: JPG + suggested interval
- Output: `output/<video_name>/` and collision-safe suffixes (`__2`, `__3`, ...)
- Naming: `<video_name>_<frame_index_6digits>.<ext>`
