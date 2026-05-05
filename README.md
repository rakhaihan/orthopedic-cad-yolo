<h1 align="center">🦴 CAD-Ortopedi (CAD Fracture Detection)</h1>

<p align="center">
  <em>Sistem CAD fraktur muskuloskeletal berbasis AI medis dengan alur SDLC 5 fase.</em>
</p>

---

## 🚀 Fase SDLC yang didukung

### 📊 Fase 1 - Data Collection & Preprocessing
- **Dataset target**: FracAtlas dan MURA (disimpan di `data/raw/`).
- **Preprocessing otomatis**: CLAHE, Gaussian denoising, resize `640x640` via `src/preprocessing.py`.
- **Split dataset**: `70/15/15` secara stratified berbasis `anatomy_region` dan label (lihat `scripts/create_splits.py`).
- **Augmentasi untuk YOLOv8**: rotasi `+-15`, horizontal flip, brightness jitter (`hsv_v`), mosaic (diatur pada `config.yaml`).

### 🧠 Fase 2 - Arsitektur Sistem
- **Detector 1**: YOLOv8-m (`src/train_detection.py`).
- **Backbone 2**: EfficientDet-D3 transfer learning (`src/train_efficientdet.py`).
- **Komponen modular**: loader (`src/dataset.py`), preprocessor (`src/preprocessing.py`), detector (`src/model.py`), visualizer/XAI (`src/explainability.py`), evaluator (`src/evaluate.py`).

### 💻 Fase 3 - Antarmuka & Explainability
- **Web UI Streamlit**: `src/app_streamlit.py`.
- **Fitur**: upload X-ray ➡️ prediksi bounding box ➡️ heatmap (Grad-CAM / EigenCAM) ➡️ ringkasan.
- **Containerized** dengan Docker (`Dockerfile`).

### 📈 Fase 4 - Evaluasi & Validasi
- **Kerangka metrik** tersedia pada `src/evaluate.py` (teknis).
- **Validasi klinis/manual reader study** disiapkan di `docs/protocol_reader_study.md`.
- *Fase ini belum dieksekusi penuh.*

### 🌍 Fase 5 - Publikasi Open Source
- Repo memuat kode, notebook demo, Dockerfile, config, dan dokumentasi.
- **Checklist rilis**: upload bobot model `.pt` ke `runs/` atau release artifacts.

---

## 📁 Struktur Proyek
```text
cad-ortopedi/
├── data/             # raw, processed, splits, yolo yaml
├── src/              # pipeline AI (train, infer, explainability, UI)
├── scripts/          # utilitas data preparation
├── notebooks/        # EDA & demo
├── tests/            # unit tests
├── docs/             # dokumentasi penelitian & protokol
├── config.yaml
├── requirements.txt
└── Dockerfile
```

---

## ⚙️ Instalasi

```bash
python -m venv .venv
```

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 💻 Panduan Training di PC Lain (Clone dari GitHub)

Karena folder data mentah (`data/raw/`), data hasil proses (`data/processed/`), dan file bobot model (`*.pt`) **tidak ikut di-push ke GitHub** (diabaikan oleh `.gitignore`), Anda harus melakukan langkah tambahan berikut jika melakukan *clone* di PC baru:

1. **Pindahkan Dataset Manual**: Salin file dataset asli (FracAtlas/MURA) dari sumber asli atau PC lama Anda.
   - Letakkan citra X-ray di `data/raw/images/`.
   - Letakkan file anotasi di `data/raw/annotations/`.
2. **Jalankan Ulang Preprocessing**: Ikuti **Langkah 1 hingga 4** pada panduan di bawah untuk memproses ulang citra dan membangun struktur dataset YOLO.
3. **Konfigurasi Hardware**: Buka file `config.yaml`.
   - Jika PC baru memiliki GPU NVIDIA, pastikan tertulis `device: cuda`.
   - Jika PC baru tanpa GPU, ubah menjadi `device: cpu` (proses training akan berjalan lambat).
4. *(Otomatis)* Pustaka YOLO akan men-download file bobot awal (`yolov8m.pt`) secara otomatis dari internet saat skrip *training* pertama kali dijalankan.

---

## 🏃‍♂️ Cara Jalankan End-to-End (Fase 1-3,5)

**1️⃣ Split stratified**
```bash
python scripts/create_splits.py --csv data/raw/annotations/dataset.csv --out data/splits --image-prefix data/processed/images_640
```

**2️⃣ Preprocessing citra**
```bash
python -m src.preprocessing --input data/raw/images --output data/processed/images_640 --size 640 640 --gaussian-kernel 5
```

**3️⃣ COCO ke YOLO labels**
```bash
python scripts/coco2yolo.py --coco data/raw/annotations/COCO_fracture_masks.json --images data/processed/images_640 --out data/processed/labels_yolo
```

**4️⃣ Susun dataset YOLO standar (wajib sebelum training)**
```bash
python scripts/prepare_yolo_dataset.py --splits-dir data/splits --labels-src data/processed/labels_yolo --out-root data/yolo
```

**5️⃣ Training YOLOv8-m**
```bash
python -m src.train_detection --config config.yaml
```

> ⚠️ **Catatan GPU:**
> - Pastikan `config.yaml` berisi `device: cuda`.
> - Cek CUDA terdeteksi:
>   ```bash
>   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
>   ```

**6️⃣ Training EfficientDet-D3**
```bash
python -m src.train_efficientdet --config config.yaml
```

**7️⃣ UI Streamlit**
```bash
streamlit run src/app_streamlit.py
```

---

## 🐳 Docker
```bash
docker build -t cad-ortopedi .
docker run --rm -p 8501:8501 cad-ortopedi
```

---

## 🔄 Reproducibility
- Simpan `config.yaml`, `requirements.txt`, dan `data/splits/splits.json`.
- Dokumentasikan hash dataset dan versi model weights.

---

## 📄 Lisensi
[MIT](LICENSE)