"""Interactive CAD fracture demo UI using Streamlit."""

from __future__ import annotations

import copy
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from model import YOLOWrapper, ResNetClassifier
from explainability import bbox_guided_classifier_cam, cam_for_model
from classifier_preprocess import preprocess_rgb_tensor
from classification_labels import LABEL_MAP_FILENAME, load_label_map


st.set_page_config(
    page_title="CAD Ortopedi",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 50
DEFAULT_IMAGE_SIDE = 1280
CLS_IMG_SIZE = 224


@st.cache_resource
def load_models(yolo_path: str | None, cls_path: str | None):
    yolo = YOLOWrapper(model_path=yolo_path)
    classifier = None
    if cls_path and Path(cls_path).exists():
        classifier = ResNetClassifier(backbone="resnet50", num_classes=2, pretrained=False)
        classifier.load_state_dict(torch.load(cls_path, map_location="cpu"))
        classifier.eval()
    return yolo, classifier


def _discover_default_yolo_path() -> str:
    candidates = sorted(Path(".").glob("runs/**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return ""


def _discover_default_cls_path() -> str:
    patterns = [
        "runs/classification_resnet50.pt",
        "runs/**/*classification*.pt",
        "runs/**/*.pt",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))
    existing = [p for p in candidates if p.is_file()]
    existing = sorted(set(existing), key=lambda p: p.stat().st_mtime, reverse=True)
    if existing:
        return str(existing[0])
    return ""


def predict_summary(has_box: bool) -> str:
    if has_box:
        return "Model mendeteksi area yang berpotensi fraktur. Tinjau bbox dan heatmap sebagai pendukung keputusan klinis."
    return "Belum ada area fraktur terdeteksi di threshold saat ini. Pertimbangkan menurunkan threshold atau meninjau ulang kualitas citra."


def resolve_target_layer(classifier: ResNetClassifier):
    if hasattr(classifier.net, "layer4"):
        return classifier.net.layer4[-1]
    return list(classifier.net.children())[-2]


def render_style(theme_mode: str):
    is_dark = theme_mode == "Dark"
    palette = {
        "app_bg": "#0f172a" if is_dark else "#f6f8fb",
        "panel_bg": "#172033" if is_dark else "#ffffff",
        "panel_soft": "#1f2a44" if is_dark else "#eef4ff",
        "text": "#e5edf7" if is_dark else "#172033",
        "muted": "#a7b4c7" if is_dark else "#526173",
        "border": "#2b3b58" if is_dark else "#dbe5f0",
        "accent": "#38bdf8" if is_dark else "#2563eb",
        "success": "#34d399" if is_dark else "#059669",
        "warning": "#fbbf24" if is_dark else "#d97706",
    }
    st.markdown(
        f"""
        <style>
            :root {{
                --app-bg: {palette["app_bg"]};
                --panel-bg: {palette["panel_bg"]};
                --panel-soft: {palette["panel_soft"]};
                --text-main: {palette["text"]};
                --text-muted: {palette["muted"]};
                --border-soft: {palette["border"]};
                --accent: {palette["accent"]};
                --success: {palette["success"]};
                --warning: {palette["warning"]};
            }}
            .stApp {{
                background: var(--app-bg);
                color: var(--text-main);
            }}
            section[data-testid="stSidebar"] {{
                background: var(--panel-bg);
                border-right: 1px solid var(--border-soft);
            }}
            .block-container {{
                padding-top: 2rem;
                padding-bottom: 3rem;
            }}
            .main-title {
                color: var(--text-main);
                font-size: 2.25rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .subtitle {
                color: var(--text-muted);
                margin-bottom: 1.25rem;
            }
            .hero-panel {
                background: var(--panel-bg);
                border: 1px solid var(--border-soft);
                border-radius: 8px;
                padding: 1.2rem 1.35rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 14px 36px rgba(15, 23, 42, 0.10);
            }
            .upload-shell {
                background: var(--panel-soft);
                border: 1px dashed var(--accent);
                border-radius: 8px;
                padding: 0.8rem 1rem 0.3rem;
                margin-bottom: 1rem;
            }
            div[data-testid="stMetric"] {
                background: var(--panel-bg);
                border: 1px solid var(--border-soft);
                border-radius: 8px;
                padding: 0.85rem 1rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            .stTabs [data-baseweb="tab"] {
                background: var(--panel-bg);
                border: 1px solid var(--border-soft);
                border-radius: 8px 8px 0 0;
                color: var(--text-main);
            }
            .stButton > button {
                min-height: 3rem;
                border-radius: 8px;
                font-weight: 700;
            }
            .stFileUploader {
                background: var(--panel-soft);
                border: 1px dashed var(--accent);
                border-radius: 8px;
                padding: 0.75rem 0.9rem 0.15rem;
            }
            .status-ok {color: var(--success); font-weight: 600;}
            .status-warn {color: var(--warning); font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_detection_stats(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0.0, 0.0
    confs = boxes.conf.detach().cpu().numpy()
    return len(confs), float(np.max(confs)), float(np.mean(confs))


def blend_cam(rgb_image: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb_u8 = np.uint8(np.clip(rgb_image, 0, 255))
    mask = np.uint8(np.clip(cam_mask, 0.0, 1.0) * 255.0)
    mask = cv2.resize(mask, (rgb_u8.shape[1], rgb_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
    color_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(rgb_u8, 1 - alpha, color_map, alpha, 0)
    return blended


def _make_detection_heatmap(result, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    heat = np.zeros((h, w), dtype=np.float32)
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return heat

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(conf))

    max_val = float(np.max(heat))
    if max_val > 0:
        heat = heat / max_val
    return heat


def _plot_detection_overlay(result, rgb_image: np.ndarray) -> np.ndarray:
    plotted = rgb_image.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return plotted

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

    for box, conf, cls_id in zip(xyxy, confs, cls_ids):
        x1, y1, x2, y2 = box.astype(int)
        _ = cls_id  # class index intentionally ignored to avoid misleading class labels
        label = f"Detected Region {conf:.2f}"

        cv2.rectangle(plotted, (x1, y1), (x2, y2), (56, 189, 248), 2)
        text_pos = (x1, max(18, y1 - 8))
        cv2.putText(
            plotted,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,  # smaller text to avoid covering X-ray details
            (56, 189, 248),
            1,
            cv2.LINE_AA,
        )
    return plotted


def _decode_upload_to_rgb(image_bytes: bytes, max_side: int) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV gagal membaca gambar (format tidak dikenali atau file corrupt).")

    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _display_width(rgb_image: np.ndarray, display_scale: int) -> int:
    width = int(rgb_image.shape[1])
    return min(1100, max(260, int(width * (display_scale / 100.0))))


def main():
    st.markdown('<div class="main-title">🩻 CAD Ortopedi Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Deteksi fraktur berbasis YOLOv8 + Explainability (Grad-CAM / EigenCAM)</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pengaturan")
        theme_mode = st.radio("Mode tampilan", ("Light", "Dark"), horizontal=True)
        render_style(theme_mode)
        default_yolo_path = _discover_default_yolo_path()
        default_cls_path = _discover_default_cls_path()
        yolo_path = st.text_input(
            "Path model YOLO",
            default_yolo_path if default_yolo_path else "",
            help="Kosongkan untuk fallback ke model default Ultralytics (yolov8m.pt).",
        )
        cls_path = st.text_input(
            "Path model klasifikasi (opsional)",
            default_cls_path if default_cls_path else "",
            help="Jika kosong/tidak ada, app pakai heatmap fallback dari area deteksi.",
        )
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        cam_method = st.selectbox(
            "Metode heatmap klasifikasi",
            ["gradcam++", "gradcam", "eigencam", "layercam"],
            index=0,
            help="Grad-CAM++ / LayerCAM sering menghasilkan fokus spatial lebih konsisten daripada Grad-CAM vanilla.",
        )
        cam_alpha = st.slider("Intensitas heatmap", 0.10, 0.90, 0.45, 0.05)
        image_max_side = st.slider(
            "Batas dimensi X-Y gambar",
            512,
            2048,
            DEFAULT_IMAGE_SIDE,
            128,
            help="Sisi terpanjang gambar upload akan diperkecil ke batas ini sebelum preview dan inferensi.",
        )
        display_scale = st.slider(
            "Skala tampilan gambar",
            35,
            100,
            70,
            5,
            help="Mengatur ukuran preview di layar tanpa mengubah file asli.",
        )
        cam_scope = st.radio(
            "Cakupan heatmap (klasifikasi)",
            ("Seluruh citra", "Ikuti bbox deteksi (disarankan)"),
            index=1,
            help="Seluruh citra: klasifikasi level gambar dapat menekankan area tanpa konteks spatial. Ikuti bbox: CAM dari crop ROI detektor lalu digabung — lebih konsisten dengan region curiga.",
        )
        if default_yolo_path:
            st.caption(f"Model YOLO otomatis terdeteksi: `{default_yolo_path}`")
        else:
            st.caption("Belum ada `runs/**/weights/best.pt`. App akan fallback ke `yolov8m.pt`.")
        if default_cls_path:
            st.caption(f"Model klasifikasi otomatis terdeteksi: `{default_cls_path}`")
        else:
            st.caption("Belum ada model klasifikasi `.pt` di folder `runs`.")
        st.caption("Tip: gunakan threshold lebih rendah jika bbox tidak muncul.")

    upload_panel = st.container()
    with upload_panel:
        upload_col, action_col = st.columns([4.2, 1.25])
        with upload_col:
            uploaded = st.file_uploader("Upload citra X-ray", type=["jpg", "jpeg", "png"], key="xray_upload")
        with action_col:
            st.write("")
            run_btn = st.button("Jalankan Analisis", type="primary", use_container_width=True)

    if not uploaded:
        st.info("Silakan upload citra terlebih dahulu untuk memulai analisis.")
        return

    if getattr(uploaded, "size", None) is not None and uploaded.size > (MAX_UPLOAD_MB * 1024 * 1024):
        st.error(f"Ukuran file terlalu besar ({uploaded.size / (1024 * 1024):.1f} MB). Maks {MAX_UPLOAD_MB} MB.")
        return

    upload_signature = (uploaded.name, getattr(uploaded, "size", None), image_max_side)
    cached_signature = st.session_state.get("upload_signature")
    if cached_signature != upload_signature:
        with st.spinner("Mengunggah & memproses gambar..."):
            try:
                image_bytes = uploaded.getvalue()
                rgb = _decode_upload_to_rgb(image_bytes, image_max_side)
            except Exception as exc:
                st.error("Gagal memproses file upload. Coba gambar lain.")
                st.exception(exc)
                return
            st.session_state["upload_signature"] = upload_signature
            st.session_state["uploaded_bytes"] = image_bytes
            st.session_state["uploaded_rgb"] = rgb

    rgb = st.session_state.get("uploaded_rgb")
    if rgb is None:
        st.error("Upload tidak terbaca. Silakan upload ulang.")
        return

    with upload_panel:
        st.caption(f"Dimensi aktif: {rgb.shape[1]} x {rgb.shape[0]} px")
        st.image(rgb, caption=f"Preview: {uploaded.name}", width=_display_width(rgb, display_scale))
    if not run_btn:
        st.warning("Klik **Jalankan Analisis** untuk memproses citra.")
        return

    image_bytes = st.session_state.get("uploaded_bytes")
    if not image_bytes:
        st.error("Gagal membaca bytes upload. Silakan upload ulang.")
        return

    with st.spinner("Memuat model dan menjalankan inferensi..."):
        yolo_model_path: str | None = None
        if yolo_path.strip():
            yolo_path_obj = Path(yolo_path)
            if not yolo_path_obj.exists():
                st.error(f"Model YOLO tidak ditemukan di path: `{yolo_path}`")
                st.stop()
            yolo_model_path = str(yolo_path_obj)
        else:
            st.info("Path YOLO kosong: menggunakan model default `yolov8m.pt`.")

        cls_path_obj = Path(cls_path) if cls_path else None
        if cls_path and cls_path_obj and not cls_path_obj.exists():
            st.warning(f"Model klasifikasi tidak ditemukan di path: `{cls_path}`. Heatmap akan dinonaktifkan.")
            cls_path = None
            cls_path_obj = None

        try:
            yolo, classifier = load_models(yolo_model_path, str(cls_path_obj) if cls_path_obj else None)
        except Exception as exc:
            st.error("Gagal memuat model. Periksa path model dan dependency (ultralytics/timm/torch).")
            st.exception(exc)
            st.stop()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        try:
            results = yolo.predict(temp_path, conf=conf)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not results:
        st.error("Inferensi gagal: tidak ada output dari model YOLO.")
        st.stop()

    result = results[0]
    plotted_rgb = _plot_detection_overlay(result, rgb)
    det_count, max_conf, mean_conf = get_detection_stats(result)
    has_box = det_count > 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Deteksi", det_count)
    m2.metric("Max confidence", f"{max_conf:.2f}")
    m3.metric("Mean confidence", f"{mean_conf:.2f}")
    m4.markdown(
        f'Status: <span class="{"status-ok" if has_box else "status-warn"}">{"Suspect Fracture" if has_box else "No Box Detected"}</span>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Perbandingan", "Deteksi", "Explainability"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.image(rgb, caption="Input Original", use_container_width=True)
        c2.image(plotted_rgb, caption="Output Bounding Box", use_container_width=True)
        st.write(predict_summary(has_box))

    with tab2:
        st.image(plotted_rgb, caption="Hasil deteksi YOLOv8", use_container_width=True)
        with st.expander("Detail Output"):
            st.text(f"File: {uploaded.name}")
            st.text(f"Threshold: {conf:.2f}")
            st.text(f"Jumlah box: {det_count}")

    with tab3:
        if classifier is None:
            det_cam_mask = _make_detection_heatmap(result, rgb.shape[:2])
            det_heatmap = blend_cam(rgb, det_cam_mask, alpha=cam_alpha)
            st.info("Model klasifikasi belum tersedia. Menampilkan heatmap fallback dari confidence area deteksi.")
            hc1, hc2 = st.columns(2)
            hc1.image(rgb, caption="Input", use_container_width=True)
            hc2.image(det_heatmap, caption="Heatmap fallback (deteksi)", use_container_width=True)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            clf_infer = classifier
            if device.type == "cuda":
                clf_infer = copy.deepcopy(classifier).to(device).eval()
            else:
                clf_infer = classifier.cpu().eval()

            label_yaml = Path(cls_path_obj).expanduser().resolve().parent / LABEL_MAP_FILENAME if cls_path_obj else None
            loaded_labels = load_label_map(label_yaml) if label_yaml else None
            if loaded_labels:
                class_to_idx, fracture_idx = loaded_labels
                st.caption(
                    "Pemetaan kelas (dari pelatihan): "
                    + ", ".join(f"{k}→{class_to_idx[k]}" for k in sorted(class_to_idx, key=lambda x: class_to_idx[x]))
                )
                st.caption(f"CAM diarahkan ke logit kelas fraktur: indeks `{fracture_idx}`.")
            else:
                fracture_idx = 0
                st.warning(
                    f"Tidak ditemukan `{LABEL_MAP_FILENAME}` di folder model. CAM memakai indeks kelas default `0`. "
                    "Jalankan ulang pelatihan klasifikasi agar pemetaan label tersimpan otomatis."
                )

            try:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

                targets = [ClassifierOutputTarget(fracture_idx)]
            except Exception:
                targets = None
                st.warning("Tidak dapat membuat target CAM eksplisit; heatmap menggunakan kelas dengan skor logits tertinggi.")

            rgb_u8 = np.clip(rgb.astype(np.uint8), 0, 255)

            tensor_g = preprocess_rgb_tensor(rgb_u8, CLS_IMG_SIZE, device=device)
            with torch.no_grad():
                probs = torch.softmax(clf_infer(tensor_g)[0], dim=0).detach().cpu().numpy()

            cols_p = st.columns(len(probs))
            for i, p in enumerate(probs):
                cols_p[i].metric(f"logit-{i}", f"{float(p)*100:.1f}%")

            target_layer = resolve_target_layer(clf_infer)

            cam_mask: np.ndarray | None = None
            bbox_guided_ok = False
            if cam_scope.startswith("Ikuti bbox") and has_box:
                guided = bbox_guided_classifier_cam(
                    clf_infer,
                    rgb_u8,
                    result,
                    target_layer,
                    preprocess_rgb_tensor,
                    img_size=CLS_IMG_SIZE,
                    device=device,
                    method=cam_method,
                    fracture_class_idx=fracture_idx,
                )
                if guided is not None:
                    cam_mask = guided.astype(np.float32)
                    bbox_guided_ok = True

            if cam_mask is None:
                cam_mask = cam_for_model(
                    clf_infer,
                    tensor_g,
                    target_layer,
                    method=cam_method,
                    targets=targets,
                )

            heatmap = blend_cam(rgb_u8, cam_mask.astype(np.float32), alpha=cam_alpha)
            hc1, hc2 = st.columns(2)
            hc1.image(rgb_u8, caption="Input", use_container_width=True)
            cap = cam_method.upper()
            if cam_scope.startswith("Ikuti bbox") and bbox_guided_ok:
                cap += " (guided bbox)"
            elif cam_scope.startswith("Ikuti bbox") and has_box and not bbox_guided_ok:
                cap += " (fallback penuh — CAM crop gagal)"
            elif cam_scope.startswith("Ikuti bbox") and not has_box:
                cap += " (penuh — tiada bbox)"
            hc2.image(heatmap.astype(np.uint8), caption=f"Heatmap klasifikasi: {cap}", use_container_width=True)

            with st.expander("Catatan klinis / interpretasi"):
                st.markdown(
                    "Heatmap klasifikasi menunjukkan **kontributor spasial bagi skor kelas tertentu** "
                    "(bukan lokasi anatomis diagnosis). Kombinasikan dengan bbox deteksi dan validasi ahli radiologi."
                )


if __name__ == "__main__":
    main()
