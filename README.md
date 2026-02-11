# videoFrameSampler (ver2)

오프라인 환경에서 동작하는 Windows GUI 기반 비디오 프레임 샘플러입니다.

## 핵심 기능

- 최대 500개 비디오 파일을 한 번에 선택하고 순차 처리
- 샘플링 간격 설정 (기본 1, 사용자 지정 N)
- 저장 포맷 선택:
  - PNG (기본, 품질 우선)
  - JPG (품질 슬라이더 제공)
- Fast mode:
  - JPG 기본 선택
  - 첫 번째 비디오 FPS/해상도 기반 간격 제안
- 출력 경로/규칙:
  - `output/<video_name>/`
  - 이름 충돌 시 `__2`, `__3` 접미사
  - 파일명: `<video_name>_<frame_index_6digits>.<ext>`

## 기술 스택

- GUI: PySide6
- 프레임 추출: OpenCV
- 패키징: PyInstaller
- 테스트: pytest

## 로컬 실행

```bash
python -m pip install -r requirements.txt
python src/app.py
```

## 테스트

```bash
python -m pytest -q
```

테스트 비디오 생성 유틸:

```bash
python tests/generate_test_video.py
```

추출 결과 자동 검증:

```bash
python tests/validate_extraction_outputs.py --output-dir output/<video_name>
```

버전/코덱 정밀 진단 CLI (Windows legacy decoder 포함):

```bash
python src/diagnose_cli.py --video "D:\Non_Documents\SDV_NG_01.avi" --interval 1 --format jpg --jpg-quality 95 --repeat 3 --report-json decode_diagnostic_report.json
```

- `probe.cv2`: OpenCV backend(`default`, `dshow`, `msmf`, `ffmpeg`)별 `open/read` 결과
- `probe.imageio`: fallback ffmpeg 메타/첫 프레임 디코딩 결과
- `probe.recovery_ffmpeg`: tolerant recovery ffmpeg 1프레임 추출 성공 여부
- 복구 단계는 자동 입력 감지 실패 시 `forced_*`/`aggressive_*` 프로필(강제 포맷 + 완화 디코드 옵션)을 순차 시도
- 위 단계가 모두 실패하면 파일 내부 Annex-B 시그니처를 탐색해 `raw H264 salvage`를 마지막으로 시도
- `extraction_runs`: 반복 실행별 로그/성공 여부 (버전/환경 편차 재현용)

## EXE 빌드 (로컬)

```bash
python -m PyInstaller --noconfirm --onefile --windowed --name videoFrameSampler --hidden-import imageio --hidden-import imageio_ffmpeg --copy-metadata imageio --copy-metadata imageio-ffmpeg src/app.py
```

결과물:

- `dist/videoFrameSampler.exe`

## 문서

- 요구사항/명세: `docs/specification.md`
- 개선 사이클 기록: `docs/dev_test_cycles.md`
- 릴리스 루틴 템플릿: `docs/release_routine.md`
- CI 빌드: `.github/workflows/build-exe.yml`
