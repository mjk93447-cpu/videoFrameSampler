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

## EXE 빌드 (로컬)

```bash
pyinstaller --noconfirm --onefile --windowed --name videoFrameSampler --hidden-import imageio --hidden-import imageio_ffmpeg --copy-metadata imageio --copy-metadata imageio-ffmpeg src/app.py
```

결과물:

- `dist/videoFrameSampler.exe`

## 문서

- 요구사항/명세: `docs/specification.md`
- 개선 사이클 기록: `docs/dev_test_cycles.md`
- 릴리스 루틴 템플릿: `docs/release_routine.md`
- CI 빌드: `.github/workflows/build-exe.yml`
