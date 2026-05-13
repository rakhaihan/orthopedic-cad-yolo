@echo off
setlocal

REM Pindah ke root project berdasarkan lokasi file .bat ini
cd /d "%~dp0"

REM Cek apakah Python terinstall di sistem
python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python tidak ditemukan. Pastikan Python sudah terinstall dan ditambahkan ke PATH.
  pause
  exit /b 1
)

REM Cek apakah virtual environment sudah ada, jika belum buat otomatis
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Virtual environment tidak ditemukan. Membuat virtual environment baru...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Gagal membuat virtual environment.
    pause
    exit /b 1
  )
  
  echo [INFO] Menginstal dependensi dari requirements.txt...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Gagal menginstal dependensi. Pastikan koneksi internet aktif.
    pause
    exit /b 1
  )
  echo [INFO] Persiapan awal selesai!
)

REM Buka browser beberapa detik setelah server mulai (non-blocking)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8501"

REM Jalankan Streamlit lewat python -m agar tidak tergantung launcher streamlit.exe
echo [INFO] Menjalankan aplikasi Streamlit...
".venv\Scripts\python.exe" -m streamlit run "src\app_streamlit.py"
if errorlevel 1 (
  echo [ERROR] Gagal menjalankan Streamlit.
  pause
  exit /b 1
)
