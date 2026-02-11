@echo off
setlocal

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q

python -m PyInstaller ^
  --noconfirm ^
  --onefile ^
  --windowed ^
  --name videoFrameSampler ^
  --hidden-import imageio ^
  --hidden-import imageio_ffmpeg ^
  --copy-metadata imageio ^
  --copy-metadata imageio-ffmpeg ^
  --exclude-module matplotlib ^
  --exclude-module tkinter ^
  --exclude-module PIL.ImageTk ^
  --exclude-module IPython ^
  --exclude-module jupyter ^
  src/app.py

echo Build complete. Output: dist\videoFrameSampler.exe
endlocal
