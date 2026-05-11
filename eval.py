import pandas as pd
from pathlib import Path
import yaml
import os
from ultralytics import YOLO

def get_image_specificity(model, image_paths, df, conf=0.25):
    """
    Menghitung Image-Level Specificity dan Sensitivity.
    Gambar dianggap positif (fractured) jika model mendeteksi minimal 1 bounding box kelas 'fractured' (kelas 1).
    """
    TN, FP, TP, FN = 0, 0, 0, 0
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run prediction pada tiap gambar satu per satu (menghindari CUDA OOM)
    for path in image_paths:
        res = model.predict(path, conf=conf, verbose=False)[0]
        stem = Path(res.path).stem
        # Ambil ground truth dari dataframe
        row = df[df['file_name_stem'] == stem]
        if len(row) == 0:
            continue
        is_true_fractured = int(row['fractured'].values[0])
        
        # Cek apakah model mendeteksi tulang patah (kelas 1)
        is_pred_fractured = 0
        if res.boxes is not None and len(res.boxes.cls) > 0:
            classes = res.boxes.cls.cpu().numpy()
            if 1 in classes:  # Asumsi kelas 1 adalah fractured
                is_pred_fractured = 1
                
        # Hitung metrik confusion matrix
        if is_true_fractured == 1:
            if is_pred_fractured == 1: TP += 1
            else: FN += 1
        else:
            if is_pred_fractured == 1: FP += 1
            else: TN += 1
            
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    return specificity, sensitivity, TN, FP, TP, FN

def main():
    print("Memuat dataset dan model...")
    # Load dataset.csv untuk mapping regio dan label image-level
    df = pd.read_csv('data/raw/annotations/dataset.csv')
    df['file_name_stem'] = df['image_id'].apply(lambda x: Path(x).stem)
    
    # Ambil list file test dari direktori YOLO
    test_dir = Path('data/yolo/images/test')
    if not test_dir.exists():
        print(f"Error: Direktori test tidak ditemukan di {test_dir}")
        return
        
    test_files = list(test_dir.glob('*.jpg'))
    
    # Load model
    model = YOLO('runs/detect/train-23/weights/best.pt')
    
    # ==========================================
    # 1. EVALUASI KESELURUHAN
    # ==========================================
    print("\n" + "="*40)
    print("=== METRIK KESELURUHAN ===")
    print("="*40)
    
    metrics_all = model.val(data='data/yolo_dataset.yaml', split='test', conf=0.25, iou=0.5, verbose=False)
    
    print(f"mAP@0.5:      {metrics_all.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics_all.box.map:.4f}")
    print(f"Box Precision:{metrics_all.box.mp:.4f}")
    print(f"Box Recall:   {metrics_all.box.mr:.4f}")
    
    # Hitung Image-Level Specificity Keseluruhan
    paths_all = [str(p) for p in test_files]
    spec_all, sens_all, tn, fp, tp, fn = get_image_specificity(model, paths_all, df, conf=0.25)
    print(f"\n[Image-Level] Specificity: {spec_all:.4f} (TN:{tn}, FP:{fp})")
    print(f"[Image-Level] Sensitivity: {sens_all:.4f} (TP:{tp}, FN:{fn})")
    
    # ==========================================
    # 2. EVALUASI PER REGIO
    # ==========================================
    regions = ['hand', 'leg', 'hip', 'shoulder', 'mixed']
    os.makedirs('data/temp_regions', exist_ok=True)
    
    # Baca yaml base untuk dicopy
    with open('data/yolo_dataset.yaml', 'r') as f:
        base_yaml = yaml.safe_load(f)
        
    print("\n" + "="*40)
    print("=== METRIK PER REGIO ANATOMI ===")
    print("="*40)
    
    for region in regions:
        # Cari file test yang masuk dalam regio ini
        region_stems = df[df[region] == 1]['file_name_stem'].tolist()
        region_files = [f for f in test_files if f.stem in region_stems]
        
        if len(region_files) == 0:
            print(f"\n--- Regio {region.upper()} ---")
            print("Tidak ada data di test set.")
            continue
            
        print(f"\n--- Regio {region.upper()} ({len(region_files)} gambar) ---")
        
        # Buat file txt berisi path absolut gambar test untuk regio ini
        txt_path = os.path.abspath(f'data/temp_regions/test_{region}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for rf in region_files:
                f.write(str(rf.absolute()) + '\n')
                
        # Buat file konfigurasi yaml sementara untuk regio ini
        yaml_content = base_yaml.copy()
        yaml_content['test'] = txt_path
        yaml_path = f'data/temp_regions/{region}.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f)
            
        # Evaluasi YOLO Bounding Box (mAP)
        try:
            metrics = model.val(data=yaml_path, split='test', conf=0.25, iou=0.5, verbose=False)
            print(f"mAP@0.5:      {metrics.box.map50:.4f}")
            print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        except Exception as e:
            print(f"Gagal mengevaluasi bbox: {e}")
            
        # Evaluasi Image-Level (Specificity & Sensitivity)
        paths_region = [str(p) for p in region_files]
        spec, sens, tn, fp, tp, fn = get_image_specificity(model, paths_region, df, conf=0.25)
        print(f"[Image-Level] Specificity: {spec:.4f} (TN:{tn}, FP:{fp})")
        print(f"[Image-Level] Sensitivity: {sens:.4f} (TP:{tp}, FN:{fn})")

if __name__ == '__main__':
    main()
