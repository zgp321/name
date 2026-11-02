@echo off
chcp 65001>nul
cd /d "%~dp0"
py -3.12 -m venv .venv || goto fail
call ".venv\Scripts\activate.bat"
python -m pip install -e . || goto fail
python -m pytest -q || echo 可选: pip install pytest 后运行单元测试
pause
exit /b 0
:fail
echo 安装失败。请检查 Python 3.12 和网络连接。
pause
exit /b 1
