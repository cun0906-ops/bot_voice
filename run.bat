@echo off
chcp 65001 >nul
title Voice Studio Server
echo ========================================================
echo        VOICE STUDIO - VIETNAMESE TTS ENGINE
echo ========================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Không tìm thấy môi trường ảo .venv!
    pause
    exit /b 1
)

echo Đang tự động kiểm tra cổng và khởi chạy Voice Studio...
echo Nhấn Ctrl+C để dừng server.
echo.

.\.venv\Scripts\python.exe app.py %1
pause


