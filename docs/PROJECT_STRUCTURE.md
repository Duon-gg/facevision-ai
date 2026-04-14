# 📁 Cấu Trúc Project — FaceVision AI

```
facevision-ai/
│
├── 📱 app.py                          ← CHƯƠNG TRÌNH CHÍNH (Streamlit UI)
├── 📄 .streamlit/config.toml          ← Cấu hình giao diện Streamlit
├── 📄 requirements.txt                ← Danh sách thư viện Python
├── 📄 .gitignore                      ← Exclude data/raw, __pycache__, .zip
├── 📄 .editorconfig                   ← Quy tắc formatting
├── 📄 LICENSE                         ← MIT License
│
├── 📂 src/                            ← MÃ NGUỒN CHÍNH (6 modules)
│   ├── 📂 models/                     ← Kiến trúc mô hình
│   │   └── face_attribute_model.py    ← ResNet50 + SE: dự đoán Tuổi, Giới tính, Sắc tộc
│   │
│   ├── 📂 data/                       ← Xử lý dữ liệu
│   │   ├── dataset.py                 ← UTKFaceDataset + Mixup augmentation
│   │   └── loader.py                  ← Chia train/val/test, augmentation, WeightedRandomSampler
│   │
│   ├── 📂 losses/                     ← Hàm mất mát
│   │   └── multi_task_loss.py         ← WingLoss + FocalLoss + Uncertainty Weighting
│   │
│   ├── 📂 detection/                  ← Phát hiện khuôn mặt
│   │   └── face_detector.py           ← MTCNN (primary) + Haar Cascade (fallback)
│   │
│   ├── 📂 inference/                  ← Suy luận (chạy model)
│   │   └── analyzer.py                ← FaceAnalyzer: TTA + batch predict pipeline
│   │
│   ├── 📂 preprocessing/              ← Tiền xử lý ảnh (DIP)
│   │   └── image_processing.py        ← CLAHE, Gaussian Blur, Resize, RGB convert
│   │
│   └── 📂 utils/                      ← Tiện ích chung
│       └── constants.py               ← Hằng số: IMG_SIZE, RACE_LABELS, DEFAULT_CONFIG
│
├── 📂 scripts/                        ← SCRIPTS TRAINING & EVALUATION
│   ├── train_v3.py                    ← Training chính (Freeze→Unfreeze, Mixup, Progressive)
│   ├── eval.py                        ← Đánh giá model trên test set
│   ├── export_charts.py               ← Xuất biểu đồ training metrics
│   └── organize_data.py               ← Tổ chức lại thư mục data
│
├── 📂 checkpoints/                    ← MODEL ĐÃ TRAIN
│   └── best_model.pth                 ← Model tốt nhất (~100MB)
│
├── 📂 data/                           ← DỮ LIỆU (raw + processed excluded from Git)
│   ├── labels.csv                     ← Labels gốc (23,705 rows)
│   ├── train.csv                      ← Labels train split (16,592 rows)
│   ├── val.csv                        ← Labels val split (3,556 rows)
│   ├── test.csv                       ← Labels test split (3,556 rows)
│   └── README.md                      ← Mô tả dataset
│
├── 📂 notebooks/
│   └── train_colab.ipynb              ← Training trên Google Colab
│
├── 📂 chart/                          ← BIỂU ĐỒ ĐÃ XUẤT
│   ├── 01-13_*.png                    ← Biểu đồ training metrics
│   └── 📂 data/01-08_*.png           ← Biểu đồ phân bố dataset
│
├── 📂 logs/                           ← METRICS
│   └── metrics.json                   ← Kết quả training & test
│
└── 📂 docs/                           ← TÀI LIỆU
    ├── PROJECT_STRUCTURE.md           ← File này
    └── 📂 screenshots/               ← Ảnh chụp cho README
```

---

## 🔍 Giải thích chi tiết

### 1. `app.py` — Giao diện chính
Ứng dụng web Streamlit 3 chế độ:
- **Ảnh tĩnh**: Upload / Webcam / Clipboard → detect + predict
- **Camera trực tiếp**: Real-time ~15-25 FPS
- **Biểu đồ đánh giá**: Training curves, Confusion Matrix, Radar

### 2. `src/` — Mã nguồn chính (6 modules)

| Module | File | Công dụng |
|---|---|---|
| **models** | `face_attribute_model.py` | ResNet50 backbone + SE attention + 3 heads |
| **data** | `dataset.py`, `loader.py` | UTKFaceDataset + Mixup + WeightedSampler |
| **losses** | `multi_task_loss.py` | WingLoss (age) + FocalLoss (race) + Uncertainty Weighting |
| **detection** | `face_detector.py` | MTCNN (chính xác) + Haar Cascade (nhanh) |
| **inference** | `analyzer.py` | Load model → TTA predict → batch processing |
| **preprocessing** | `image_processing.py` | CLAHE histogram equalization + Gaussian blur |
| **utils** | `constants.py` | IMG_SIZE, labels, training defaults |

### 3. Inference Pipeline

```
Ảnh đầu vào
    → face_detector.py (MTCNN detect + crop mặt)
    → image_processing.py (CLAHE + resize 224×224)
    → analyzer.py (TTA: 5 augmentations → average)
    → Kết quả: Tuổi 25, Nữ, Asian (95%)
```

### 4. Training Pipeline

```
data/raw/UTKFace/ (23,707 ảnh)
    → labels.csv (tuổi, giới, sắc tộc từ filename)
    → loader.py (split 70/15/15 + WeightedSampler)
    → dataset.py (Mixup augmentation)
    → face_attribute_model.py (forward pass)
    → multi_task_loss.py (WingLoss + FocalLoss + Uncertainty)
    → Optimizer (AdamW + OneCycleLR)
    → Lưu best model → checkpoints/best_model.pth
```
