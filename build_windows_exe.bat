@echo off
chcp 65001>nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo 请先运行 setup_windows.bat
 pause
 exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install pyinstaller || goto fail
python -m PyInstaller --noconfirm --clean --onedir --windowed --name TEM_WaterRisk_V1 --collect-all matplotlib --collect-all scipy run_gui.py || goto fail
echo 完成：dist\TEM_WaterRisk_V1\TEM_WaterRisk_V1.exe
pause
exit /b 0
:fail
echo 打包失败，请检查依赖和日志。
pause
exit /b 1
