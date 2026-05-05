@echo off
REM Aktifkan virtual environment
call "D:\Education\University\telkom\mulai kuliah\SMT 6\Tatulim\project\cad-ortopedi\.venv\Scripts\activate"

REM Jalankan Streamlit dengan file app_streamlit.py
streamlit run "D:\Education\University\telkom\mulai kuliah\SMT 6\Tatulim\project\cad-ortopedi\src\app_streamlit.py"

REM (Opsional) otomatis buka browser ke localhost:8501
start http://localhost:8501

REM Supaya jendela tidak langsung tertutup
pause
