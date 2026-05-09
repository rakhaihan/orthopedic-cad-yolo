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


@st.cache_resource
def load_models(yolo_path: str, cls_path: str | None):
    yolo = YOLOWrapper(model_path=yolo_path)
    classifier = None
    if cls_path and Path(cls_path).exists():
        classifier = ResNetClassifier(backbone="resnet50", num_classes=2, pretrained=False)
        classifier.load_state_dict(torch.load(cls_path, map_location="cpu"))
        classifier.eval()
    return yolo, classifier


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


def main():
    render_style()
    st.markdown('<div class="main-title">🩻 CAD Ortopedi Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Deteksi fraktur berbasis YOLOv8 + Explainability (Grad-CAM / EigenCAM)</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pengaturan")
        yolo_path = st.text_input("Path model YOLO", "runs/detect/train/weights/best.pt")
        cls_path = st.text_input("Path model klasifikasi (opsional)", "runs/classification_resnet50.pt")
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        cam_method = st.selectbox("Metode heatmap", ["gradcam", "eigencam"])
        cam_alpha = st.slider("Intensitas heatmap", 0.10, 0.90, 0.45, 0.05)
        run_btn = st.button("Jalankan Analisis", type="primary", use_container_width=True)

        st.caption("Tip: gunakan threshold lebih rendah jika bbox tidak muncul.")

    uploaded = st.file_uploader("Upload citra X-ray", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Silakan upload citra terlebih dahulu untuk memulai analisis.")
        return
    if not run_btn:
        st.warning("Klik **Jalankan Analisis** untuk memproses citra.")
        return

    image_bytes = uploaded.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Gagal membaca gambar. Coba upload file JPG/PNG lain (file ini mungkin corrupt).")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with st.spinner("Memuat model dan menjalankan inferensi..."):
        yolo_path_obj = Path(yolo_path)
        if not yolo_path_obj.exists():
            st.error(f"Model YOLO tidak ditemukan di path: `{yolo_path}`")
            st.stop()

        cls_path_obj = Path(cls_path) if cls_path else None
        if cls_path and not cls_path_obj.exists():
            st.warning(f"Model klasifikasi tidak ditemukan di path: `{cls_path}`. Heatmap akan dinonaktifkan.")
            cls_path = None

        try:
            yolo, classifier = load_models(str(yolo_path_obj), str(cls_path_obj) if cls_path_obj else None)
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
    plotted = result.plot()
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
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
            st.info("Model klasifikasi belum tersedia, heatmap tidak ditampilkan.")
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
