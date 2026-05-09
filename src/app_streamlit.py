"""Interactive CAD fracture demo UI using Streamlit."""

from pathlib import Path
import tempfile
import os
import cv2
import numpy as np
import streamlit as st
import torch

from model import YOLOWrapper, ResNetClassifier
from explainability import cam_for_model


st.set_page_config(
    page_title="CAD Ortopedi",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 50
MAX_IMAGE_SIDE = 2048


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


def predict_summary(has_box: bool) -> str:
    if has_box:
        return "Model mendeteksi area yang berpotensi fraktur. Tinjau bbox dan heatmap sebagai pendukung keputusan klinis."
    return "Belum ada area fraktur terdeteksi di threshold saat ini. Pertimbangkan menurunkan threshold atau meninjau ulang kualitas citra."


def resolve_target_layer(classifier: ResNetClassifier):
    if hasattr(classifier.net, "layer4"):
        return classifier.net.layer4[-1]
    return list(classifier.net.children())[-2]


def render_style():
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.0rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .subtitle {
                color: #9AA0A6;
                margin-bottom: 1rem;
            }
            .status-ok {color: #19c37d; font-weight: 600;}
            .status-warn {color: #f59e0b; font-weight: 600;}
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
    mask = np.uint8(np.clip(cam_mask, 0.0, 1.0) * 255.0)
    mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    color_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(rgb_image, 1 - alpha, color_map, alpha, 0)
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


def _decode_upload_to_rgb(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV gagal membaca gambar (format tidak dikenali atau file corrupt).")

    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    render_style()
    st.markdown('<div class="main-title">🩻 CAD Ortopedi Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Deteksi fraktur berbasis YOLOv8 + Explainability (Grad-CAM / EigenCAM)</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pengaturan")
        default_yolo_path = _discover_default_yolo_path()
        yolo_path = st.text_input(
            "Path model YOLO",
            default_yolo_path if default_yolo_path else "",
            help="Kosongkan untuk fallback ke model default Ultralytics (yolov8m.pt).",
        )
        cls_path = st.text_input("Path model klasifikasi (opsional)", "runs/classification_resnet50.pt")
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        cam_method = st.selectbox("Metode heatmap", ["gradcam", "eigencam"])
        cam_alpha = st.slider("Intensitas heatmap", 0.10, 0.90, 0.45, 0.05)
        run_btn = st.button("Jalankan Analisis", type="primary", use_container_width=True)

        if default_yolo_path:
            st.caption(f"Model YOLO otomatis terdeteksi: `{default_yolo_path}`")
        else:
            st.caption("Belum ada `runs/**/weights/best.pt`. App akan fallback ke `yolov8m.pt`.")
        st.caption("Tip: gunakan threshold lebih rendah jika bbox tidak muncul.")

    uploaded = st.file_uploader("Upload citra X-ray", type=["jpg", "jpeg", "png"], key="xray_upload")
    if not uploaded:
        st.info("Silakan upload citra terlebih dahulu untuk memulai analisis.")
        return

    if getattr(uploaded, "size", None) is not None and uploaded.size > (MAX_UPLOAD_MB * 1024 * 1024):
        st.error(f"Ukuran file terlalu besar ({uploaded.size / (1024 * 1024):.1f} MB). Maks {MAX_UPLOAD_MB} MB.")
        return

    upload_signature = (uploaded.name, getattr(uploaded, "size", None))
    cached_signature = st.session_state.get("upload_signature")
    if cached_signature != upload_signature:
        with st.spinner("Mengunggah & memproses gambar..."):
            try:
                image_bytes = uploaded.getvalue()
                rgb = _decode_upload_to_rgb(image_bytes)
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

    st.image(rgb, caption=f"Preview: {uploaded.name}", use_container_width=True)
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
        if cls_path and not cls_path_obj.exists():
            st.warning(f"Model klasifikasi tidak ditemukan di path: `{cls_path}`. Heatmap akan dinonaktifkan.")
            cls_path = None

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
            tensor = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
            tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
            target_layer = resolve_target_layer(classifier)
            cam_mask = cam_for_model(classifier, tensor, target_layer, method=cam_method)
            heatmap = blend_cam(rgb, cam_mask, alpha=cam_alpha)
            hc1, hc2 = st.columns(2)
            hc1.image(rgb, caption="Input", use_container_width=True)
            hc2.image(heatmap, caption=f"Heatmap {cam_method}", use_container_width=True)


if __name__ == "__main__":
    main()
