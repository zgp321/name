@echo off
chcp 65001>nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 首次安装: 请先安装 Python 3.12，然后运行 setup_windows.bat
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m mine_tem.gui
if errorlevel 1 pause
