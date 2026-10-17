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

:: Kiểm tra và cài đặt edge-tts nếu chưa có trong môi trường ảo
if not exist ".venv\Lib\site-packages\edge_tts" (
    echo [INFO] Đang tự động cài đặt thư viện edge-tts vào môi trường ảo...
    .\.venv\Scripts\python.exe -m pip install edge-tts
    if errorlevel 1 (
        echo [CẢNH BÁO] Không thể tự động cài đặt edge-tts.
    ) else (
        echo [OK] Đã cài đặt edge-tts thành công!
    )
    echo.
)

echo Đang tự động kiểm tra cổng và khởi chạy Voice Studio...
echo Nhấn Ctrl+C để dừng server.
echo.

.\.venv\Scripts\python.exe app.py %1
pause


