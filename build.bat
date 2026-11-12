@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 3.12 not found in PATH.
  exit /b 1
)

echo [1/3] Installing dependencies...
python -m pip install --disable-pip-version-check -r requirements.txt || exit /b 1

echo [2/3] Downloading and verifying ONNX models...
python scripts\fetch_models.py || exit /b 1

echo [3/3] Building single-file exe...
python -m PyInstaller --noconfirm --clean --onefile --windowed --noupx --name stem3 ^
  --distpath dist --workpath build --specpath . ^
  --add-data "models\UVR_MDXNET_Main.onnx;models" ^
  --add-data "models\UVR_MDXNET_KARA_2.onnx;models" ^
  --collect-all sherpa_onnx ^
  --collect-all soundfile ^
  --exclude-module torch ^
  --exclude-module onnxruntime ^
  --exclude-module scipy ^
  --exclude-module librosa ^
  --exclude-module matplotlib ^
  --exclude-module pandas ^
  --exclude-module PIL ^
  --exclude-module pytest ^
  --exclude-module PySide6.QtWebEngineCore ^
  --exclude-module PySide6.QtWebEngineWidgets ^
  --exclude-module PySide6.QtPdf ^
  --exclude-module PySide6.Qt3DCore ^
  app\stem3.py || exit /b 1

echo.
echo [OK] Built: dist\stem3.exe
endlocal
