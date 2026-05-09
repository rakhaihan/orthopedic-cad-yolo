@echo off
setlocal

REM Pindah ke root project berdasarkan lokasi file .bat ini
cd /d "%~dp0"

REM Gunakan interpreter virtual environment langsung (lebih aman dari path activate lama)
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Python virtual environment tidak ditemukan.
  echo Buat dulu dengan: python -m venv .venv
  pause
  exit /b 1
)

REM Buka browser beberapa detik setelah server mulai (non-blocking)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8501

REM Jalankan Streamlit lewat python -m agar tidak tergantung launcher streamlit.exe
".venv\Scripts\python.exe" -m streamlit run "src\app_streamlit.py"
if errorlevel 1 (
  echo [ERROR] Gagal menjalankan Streamlit.
  echo Coba install dependency: pip install -r requirements.txt
  pause
  exit /b 1
)
