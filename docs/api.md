# API Modules

Dokumentasi ringkas modul inti pipeline CAD-Ortopedi.

## `src.preprocessing`
- `apply_clahe_gray(img)` -> CLAHE grayscale.
- `process_image(in_path, out_path, size, gaussian_kernel)` -> preprocess 1 citra.
- `batch_process(input_dir, output_dir, size, gaussian_kernel)` -> preprocess batch.

## `src.model`
- `YOLOWrapper` -> training/inference YOLOv8.
- `EfficientDetWrapper` -> backbone EfficientDet-D3.
- `ResNetClassifier` -> classifier biner fraktur/non-fraktur.

## `src.train_detection`
- `train(cfg_path)` -> train YOLOv8 memakai `config.yaml`.

## `src.train_efficientdet`
- `train(cfg_path)` -> fine-tuning EfficientDet-D3 dari pretrained weights.

## `src.train_classification`
- `train(cfg_path)` -> train classifier folder-based.

## `src.explainability`
- `cam_for_model(model, input_tensor, target_layer, method)` -> Grad-CAM / EigenCAM.
- `overlay_cam(img_rgb, cam_mask)` -> visualisasi heatmap di atas citra.

## `src.app_streamlit`
- Web UI upload -> deteksi -> heatmap -> summary.
