# CHƯƠNG 3: PHÂN TÍCH YÊU CẦU VÀ THIẾT KẾ HỆ THỐNG

---

## 3.1 Phân tích yêu cầu chức năng

### UC01 — Dự đoán thuộc tính từ ảnh tĩnh

| Mục | Mô tả |
|-----|-------|
| **Tác nhân** | Người dùng (User) |
| **Mô tả** | Người dùng upload ảnh chân dung, hệ thống tự động phát hiện khuôn mặt, tiền xử lý và dự đoán Tuổi, Giới tính, Sắc tộc |
| **Đầu vào** | Ảnh JPG/PNG (≤ 10MB), chụp hoặc clipboard, hoặc webcam snapshot |
| **Xử lý** | (1) MTCNN detect faces → (2) Crop & CLAHE preprocessing → (3) Forward qua model với TTA → (4) Hiển thị kết quả |
| **Đầu ra** | Bounding box trên ảnh gốc + Tuổi (năm), Giới tính (Male/Female), Sắc tộc (White/Black/Asian/Others) kèm confidence % |
| **Luồng thay thế** | Nếu MTCNN detect thất bại → Haar Cascade fallback → Nếu vẫn không detect → thông báo lỗi |

### UC02 — Dự đoán từ camera trực tiếp

| Mục | Mô tả |
|-----|-------|
| **Tác nhân** | Người dùng |
| **Mô tả** | Hệ thống capture webcam liên tục, phát hiện và dự đoán thuộc tính khuôn mặt theo thời gian thực |
| **Đầu vào** | Video stream từ webcam (cv2.VideoCapture) |
| **Xử lý** | Mỗi frame: Haar Cascade detect → CLAHE → Model predict (không TTA — tốc độ ưu tiên) → Vẽ bounding box + label |
| **Đầu ra** | Video stream với annotation (tuổi, giới tính, sắc tộc) overlay, FPS hiển thị, nút Start/Stop |
| **Yêu cầu** | ≥ 15 FPS, latency ≤ 100ms/frame |

### UC03 — Xem biểu đồ đánh giá mô hình

| Mục | Mô tả |
|-----|-------|
| **Tác nhân** | Người dùng / Reviewer (giảng viên) |
| **Mô tả** | Hiển thị 13+ biểu đồ tương tác đánh giá hiệu năng mô hình |
| **Đầu vào** | File `logs/metrics.json` chứa training history và test metrics |
| **Đầu ra** | Plotly charts (Training Curves, Confusion Matrix, P/R/F1, Radar, Age Error by Group, Dataset Distribution, Score Card) |

### UC04 — Huấn luyện mô hình (Developer)

| Mục | Mô tả |
|-----|-------|
| **Tác nhân** | Developer (thông qua CLI) |
| **Mô tả** | Chạy script `scripts/train_v3.py` để huấn luyện mô hình trên dataset UTKFace |
| **Tham số CLI** | `--resume`, `--extra-epochs N`, `--no-mixup`, `--no-progressive` |
| **Đầu ra** | File `checkpoints/best_model_v3.pth` + `logs/metrics_v3.json` |

### UC05 — Đánh giá mô hình trên test set (Developer)

| Mục | Mô tả |
|-----|-------|
| **Tác nhân** | Developer |
| **Mô tả** | Chạy `scripts/eval.py` để đánh giá chi tiết mô hình trên test set |
| **Đầu ra** | Console output: Age MAE, Gender/Race Accuracy, Confusion Matrix, Classification Report, Xếp hạng |

---

## 3.2 Phân tích yêu cầu phi chức năng

| ID | Yêu cầu | Chỉ tiêu |
|----|---------|---------|
| NFR01 | **Hiệu năng** | Inference ≤ 500ms/ảnh (CPU), ≤ 100ms (GPU) |
| NFR02 | **Thời gian thực** | Camera live ≥ 15 FPS |
| NFR03 | **Độ chính xác** | Age MAE ≤ 5 năm, Gender ≥ 90%, Race ≥ 80% |
| NFR04 | **Khả năng mở rộng** | Kiến trúc modular, dễ thêm task mới |
| NFR05 | **Tương thích** | Chạy trên Windows, macOS, Linux; CPU và CUDA GPU |
| NFR06 | **Giao diện** | UI responsive, hỗ trợ upload, camera, clipboard |
| NFR07 | **Bảo trì** | Code clean, comment đầy đủ, docstring tiếng Việt |
| NFR08 | **Dung lượng model** | ≤ 200MB (checkpoint `.pth`) |

---

## 3.3 Biểu đồ Use Case

```
                    ┌────────────────────────────────────────────┐
                    │            FaceVision AI System            │
                    │                                            │
  ┌────────┐        │  ┌──────────────────────────────────┐     │
  │        │        │  │  UC01: Dự đoán từ ảnh tĩnh      │     │
  │        │───────→│  │   (Upload / Webcam / Clipboard)   │     │
  │        │        │  └──────────────────────────────────┘     │
  │  User  │        │                                            │
  │        │        │  ┌──────────────────────────────────┐     │
  │        │───────→│  │  UC02: Dự đoán từ camera live    │     │
  │        │        │  │   (Real-time ~15-25 FPS)          │     │
  │        │        │  └──────────────────────────────────┘     │
  │        │        │                                            │
  │        │        │  ┌──────────────────────────────────┐     │
  │        │───────→│  │  UC03: Xem biểu đồ đánh giá     │     │
  └────────┘        │  │   (13 Plotly charts)              │     │
                    │  └──────────────────────────────────┘     │
                    │                                            │
  ┌────────┐        │  ┌──────────────────────────────────┐     │
  │        │        │  │  UC04: Huấn luyện mô hình        │     │
  │Developer│──────→│  │   (train_v3.py CLI)               │     │
  │        │        │  └──────────────────────────────────┘     │
  │        │        │                                            │
  │        │        │  ┌──────────────────────────────────┐     │
  │        │───────→│  │  UC05: Đánh giá test set         │     │
  └────────┘        │  │   (eval.py CLI)                   │     │
                    │  └──────────────────────────────────┘     │
                    └────────────────────────────────────────────┘
```

---

## 3.4 Biểu đồ hoạt động (Activity Diagram)

### 3.4.1 Activity Diagram — Dự đoán ảnh tĩnh (UC01)

```
[Bắt đầu]
    │
    ▼
┌────────────────────────┐
│  Người dùng upload ảnh │
│  (hoặc chụp webcam)    │
└────────────┬───────────┘
             │
             ▼
┌────────────────────────┐
│  Chuyển đổi sang RGB   │
│  (ensure_rgb)           │
└────────────┬───────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Phát hiện khuôn mặt (MTCNN)   │
│  → Trả về bounding boxes       │
└────────────┬────────────────────┘
             │
     ┌───────┴──────┐
     │ Detect = 0?  │
     └───────┬──────┘
        Yes  │  No
        │    │
        ▼    ▼
  ┌──────────┐  ┌──────────────────────┐
  │ Fallback │  │ Duyệt từng khuôn mặt│
  │  Haar    │  └──────────┬───────────┘
  │ Cascade  │             │
  └────┬─────┘             ▼
       │          ┌────────────────────┐
       │          │ Crop khuôn mặt     │
       │          │ (từ bounding box)   │
       │          └────────┬───────────┘
       │                   │
       ▼                   ▼
  ┌──────────┐  ┌────────────────────┐
  │ Detect   │  │ Tiền xử lý CLAHE   │
  │  = 0?    │  │ (LAB → CLAHE → RGB) │
  └────┬─────┘  └────────┬───────────┘
   Yes │                  │
   │   │                  ▼
   ▼   │       ┌────────────────────┐
  ┌────┴───┐   │ Resize 224×224      │
  │Thông   │   │ + Normalize          │
  │báo lỗi│   └────────┬───────────┘
  └────────┘            │
                        ▼
             ┌────────────────────┐
             │ TTA: 5 augmented   │
             │ versions → average │
             └────────┬───────────┘
                      │
                      ▼
             ┌────────────────────┐
             │ Kết quả:           │
             │  Tuổi: 25 năm     │
             │  Giới tính: Female │
             │  Sắc tộc: Asian   │
             │  Confidence: 95%  │
             └────────┬───────────┘
                      │
                      ▼
             ┌────────────────────┐
             │ Hiển thị trên UI   │
             │ (bounding box +    │
             │  label overlay)    │
             └────────────────────┘
                      │
                      ▼
                 [Kết thúc]
```

### 3.4.2 Activity Diagram — Huấn luyện mô hình (UC04)

```
[Bắt đầu Training]
    │
    ▼
┌──────────────────────────────┐
│ Load config (DEFAULT_CONFIG) │
│ Parse arguments CLI          │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Load data (train/val/test)    │
│ WeightedRandomSampler applied │
│ Progressive Resize → 160px   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ PHASE 1: Freeze backbone     │
│ (epoch 1→5)                   │
│ Train heads only, LR=1e-3    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ PHASE 2: Unfreeze backbone   │
│ (epoch 6→50)                  │
│ Differential LR:             │
│  backbone=1e-5, heads=1e-4   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Mỗi epoch:                    │
│  (1) Train with Mixup + AMP  │
│  (2) Gradient Accumulation    │
│  (3) Validate                 │
│  (4) Save best model          │
│  (5) Early Stopping check     │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Test Set Evaluation           │
│ Reload best model @ 224px     │
│ → Age MAE, Gender/Race Acc   │
│ → Confusion matrix, Report   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Export metrics.json            │
│ Save checkpoint .pth          │
└──────────────────────────────┘
           │
           ▼
      [Kết thúc]
```

---

## 3.5 Kiến trúc tổng thể hệ thống

### Sơ đồ kiến trúc 4 tầng

```
╔═══════════════════════════════════════════════════════════════╗
║  TẦNG 1: INPUT LAYER                                         ║
║    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   ║
║    │  Upload  │  │  Webcam  │  │Clipboard │  │  Camera  │   ║
║    │   Image  │  │ Snapshot │  │   Paste  │  │   Live   │   ║
║    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   ║
╚═════════╪═════════════╪════════════╪═════════════╪═══════════╝
          └──────────────┴────────────┴─────────────┘
                              │
╔═════════════════════════════╪═════════════════════════════════╗
║  TẦNG 2: DETECTION LAYER    │                                 ║
║    ┌──────────────────────────────────────────┐               ║
║    │         Face Detector                     │               ║
║    │  ┌──────────┐       ┌──────────────────┐ │               ║
║    │  │  MTCNN   │──OR──→│  Haar Cascade    │ │               ║
║    │  │(primary) │       │   (fallback)      │ │               ║
║    │  └──────────┘       └──────────────────┘ │               ║
║    │  Output: Bounding boxes + Confidence     │               ║
║    └──────────────────────────────────────────┘               ║
╚═════════════════════════════╪═════════════════════════════════╝
                              │
╔═════════════════════════════╪═════════════════════════════════╗
║  TẦNG 3: PREPROCESSING LAYER│                                 ║
║    ┌──────────────────────────────────────────┐               ║
║    │         Image Processor                   │               ║
║    │  RGB Convert → CLAHE (LAB) → Gaussian    │               ║
║    │  → Resize 224×224 → Normalize (ImageNet) │               ║
║    └──────────────────────────────────────────┘               ║
╚═════════════════════════════╪═════════════════════════════════╝
                              │
╔═════════════════════════════╪═════════════════════════════════╗
║  TẦNG 4: PREDICTION LAYER   │                                 ║
║    ┌──────────────────────────────────────────┐               ║
║    │         FaceAttributeModel                │               ║
║    │  ResNet50 backbone → SE Attention         │               ║
║    │    ├── Age Head (regression → MAE)        │               ║
║    │    ├── Gender Head (classification → 2)   │               ║
║    │    └── Race Head (classification → 4)     │               ║
║    │                                            │               ║
║    │  TTA: 5 augmented predictions → average   │               ║
║    └──────────────────────────────────────────┘               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 3.6 Thiết kế cấu trúc thư mục (Project Structure)

```
facevision-ai/
│
├── app.py                          ← Ứng dụng chính Streamlit (1,397 dòng)
├── .streamlit/config.toml          ← Cấu hình theme + server
├── requirements.txt                ← Dependencies (10 packages)
├── .gitignore                      ← Exclude data/raw, __pycache__
├── .editorconfig                   ← Code formatting rules
├── LICENSE                         ← MIT License
│
├── src/                            ← MÃ NGUỒN CHÍNH (7 modules)
│   ├── models/
│   │   └── face_attribute_model.py ← ResNet50 + SE + 3 heads (139 dòng)
│   ├── data/
│   │   ├── dataset.py              ← UTKFaceDataset + Mixup (192 dòng)
│   │   └── loader.py               ← Split + Augment + Sampler (175 dòng)
│   ├── losses/
│   │   └── multi_task_loss.py      ← Wing + Focal + Uncertainty (137 dòng)
│   ├── detection/
│   │   └── face_detector.py        ← MTCNN + Haar fallback (149 dòng)
│   ├── inference/
│   │   └── analyzer.py             ← FaceAnalyzer + TTA (466 dòng)
│   ├── preprocessing/
│   │   └── image_processing.py     ← CLAHE + Blur + Resize (126 dòng)
│   └── utils/
│       └── constants.py            ← Labels + Config (81 dòng)
│
├── scripts/                        ← Scripts huấn luyện & đánh giá
│   ├── train_v3.py                 ← Training chính v3 (639 dòng)
│   ├── eval.py                     ← Đánh giá test set (206 dòng)
│   ├── export_charts.py            ← Xuất 13 biểu đồ PNG (639 dòng)
│   └── organize_data.py            ← Tổ chức data thành splits (211 dòng)
│
├── checkpoints/
│   └── best_model.pth              ← Model tốt nhất (~100 MB)
│
├── data/
│   ├── raw/UTKFace/                ← 23,704 ảnh gốc
│   ├── processed/                  ← Ảnh chia theo split + class
│   ├── labels.csv                  ← Nhãn gốc (23,704 rows)
│   ├── train.csv, val.csv, test.csv
│   └── README.md                   ← Mô tả dataset
│
├── notebooks/
│   └── train_colab.ipynb           ← Notebook Colab training
│
├── chart/                          ← 13 biểu đồ PNG (2x resolution)
│   └── data/                       ← Biểu đồ phân bố dataset
│
├── logs/
│   ├── metrics.json                ← Kết quả v2 (914 dòng)
│   └── metrics_v3.json             ← Kết quả v3 (nếu có)
│
└── docs/
    ├── PROJECT_STRUCTURE.md
    └── screenshots/                ← Ảnh chụp UI
```

---

## 3.7 Thiết kế dữ liệu

### 3.7.1 Bộ dữ liệu UTKFace

**UTKFace** (Large Scale Face Dataset) là bộ dữ liệu khuôn mặt công khai do University of Tennessee, Knoxville phát triển.

| Thuộc tính | Giá trị |
|------------|---------|
| Tổng số ảnh | 23.704 ảnh (sau lọc) |
| Kích thước gốc | 200×200 pixels, JPEG RGB |
| Phạm vi tuổi | 0–116 tuổi |
| Giới tính | 0 = Male, 1 = Female |
| Sắc tộc (gốc) | 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others |
| Sắc tộc (remap) | 0=White, 1=Black, 2=Asian, 3=Others (gộp Indian+Others) |

**Quy tắc đặt tên file**: `age_gender_race_timestamp.jpg`

Ví dụ: `25_0_2_20170116172557225.jpg` → Tuổi 25, Male (0), Asian (2)

### 3.7.2 Cấu trúc file CSV nhãn

File `data/labels.csv`:

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `image` | string | Tên file ảnh |
| `age` | int | Tuổi (0–116) |
| `gender` | int | 0=Male, 1=Female |
| `race` | int | 0–4 (5 lớp gốc) |

### 3.7.3 Phân chia Train/Val/Test

| Split | Số mẫu | Tỷ lệ | White | Black | Asian | Others |
|-------|--------|--------|-------|-------|-------|--------|
| **Train** | 16.592 | 70% | 7.051 | 3.141 | 2.425 | 3.975 |
| **Val** | 3.556 | 15% | 1.508 | 722 | 482 | 844 |
| **Test** | 3.556 | 15% | 1.518 | 663 | 527 | 848 |
| **Tổng** | **23.704** | **100%** | **10.077** | **4.526** | **3.434** | **5.667** |

Phân chia thực hiện bằng `sklearn.model_selection.train_test_split` với `random_state=42` và `stratify=gender` để đảm bảo phân bố giới tính đồng đều giữa các split.

### 3.7.4 Ánh xạ nhãn (Label Mapping)

| Gốc | Tên gốc | Remap | Tên mới |
|-----|---------|-------|---------|
| 0 | White | 0 | White |
| 1 | Black | 1 | Black |
| 2 | Asian | 2 | Asian |
| 3 | Indian | **3** | **Others** |
| 4 | Others | **3** | **Others** |

Lý do gộp: Lớp Indian (3) và Others (4) có số mẫu ít, khó phân biệt → gộp lại thành "Others" để cân bằng hơn.

Ánh xạ được định nghĩa trong `src/utils/constants.py`:

```python
RACE_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
RACE_LABELS = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Others'}
NUM_RACE_CLASSES = 4
```

---

## 3.8 Thiết kế kiến trúc mô hình

### Sơ đồ kiến trúc FaceAttributeModel

```
Input Tensor [B, 3, 224, 224]
          │
          ▼
┌──────────────────────────────────┐
│     ResNet50 Backbone            │
│  (pretrained ImageNet, fc=Identity)  │
│  → Output: [B, 2048]            │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│     SE Block (channel attention) │
│  squeeze: AdaptiveAvgPool1d(1)   │
│  excite: Linear(2048→128→2048)   │
│           + Sigmoid              │
│  → Output: [B, 2048]            │
└──────────────┬───────────────────┘
               │
       ┌───────┼───────┐
       │       │       │
       ▼       ▼       ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Age Head │ │ Gender   │ │ Race     │
│          │ │ Head     │ │ Head     │
│ FC 2048  │ │ FC 2048  │ │ FC 2048  │
│ BN+ReLU  │ │ BN+ReLU  │ │ BN+ReLU  │
│ Dropout  │ │ Dropout  │ │ Dropout  │
│ FC  512  │ │ FC  256  │ │ FC  256  │
│ BN+ReLU  │ │ Drop0.3  │ │ Drop0.3  │
│ Dropout  │ │ FC   2   │ │ FC   4   │
│ FC  128  │ └──────────┘ └──────────┘
│ BN+ReLU  │
│ FC    1  │ ← ReLU clamp [0, 120]
└──────────┘

Output: (age_pred [B,1], gender_logits [B,2], race_logits [B,4])
```

### Bảng tóm tắt số tham số

| Thành phần | Số tham số | Tỷ lệ |
|-----------|-----------|--------|
| ResNet50 backbone | ~23.5M | 87.5% |
| SE Block | ~0.5M | 1.9% |
| Age Head | ~1.2M | 4.5% |
| Gender Head | ~0.5M | 1.9% |
| Race Head | ~0.5M | 1.9% |
| Criterion (learnable weights) | 3 | ~0% |
| **Tổng** | **~26.2M** | **100%** |

---

## 3.9 Thiết kế pipeline Huấn luyện

```
┌─────────────────────────────────────────┐
│           TRAINING PIPELINE              │
│                                          │
│  Data Flow:                              │
│  labels.csv → train_test_split (70/15/15)│
│  → UTKFaceDataset (transforms + Mixup)   │
│  → WeightedRandomSampler (Race + Age)    │
│  → DataLoader (batch=32, workers=4)      │
│                                          │
│  Training Config:                        │
│  ┌────────────────────────────────────┐  │
│  │ Phase 1: Freeze backbone (ep 1-5) │  │
│  │  LR = 1e-3 (heads only)           │  │
│  │  Image size: 160px                 │  │
│  ├────────────────────────────────────┤  │
│  │ Phase 2: Unfreeze (ep 6-50)        │  │
│  │  DLR: backbone=1e-5, heads=1e-4   │  │
│  │  Progressive: 160→192→224px       │  │
│  ├────────────────────────────────────┤  │
│  │ Loss: WingLoss + FocalLoss +       │  │
│  │       Adaptive Uncertainty         │  │
│  ├────────────────────────────────────┤  │
│  │ Optimizer: AdamW (weight_decay=1e-4)│ │
│  │ AMP: float16 on CUDA               │  │
│  │ Grad Accum: 4 steps (eff.128)      │  │
│  │ Grad Clip: max_norm = 1.0          │  │
│  │ Early Stopping: patience = 10      │  │
│  └────────────────────────────────────┘  │
│                                          │
│  Output:                                 │
│  → checkpoints/best_model.pth            │
│  → logs/metrics.json                     │
└─────────────────────────────────────────┘
```

---

## 3.10 Thiết kế pipeline Suy luận (Inference)

```
┌─────────────────────────────────────────────┐
│            INFERENCE PIPELINE                │
│                                              │
│  Input Image                                 │
│      │                                       │
│      ▼                                       │
│  ┌─────────────────────────────────┐         │
│  │ Face Detection                   │         │
│  │  Primary: MTCNN                  │         │
│  │  Fallback: Haar Cascade          │         │
│  │  Output: List[BBox]              │         │
│  └─────────────┬───────────────────┘         │
│                │                              │
│      ┌─────────┴─────────┐                   │
│      │ For each face      │                   │
│      │                    │                   │
│      ▼                    │                   │
│  ┌─────────────────┐     │                   │
│  │ Crop face region │     │                   │
│  │ (from bbox + pad)│     │                   │
│  └────────┬────────┘     │                   │
│           │               │                   │
│           ▼               │                   │
│  ┌─────────────────┐     │                   │
│  │ Preprocessing    │     │                   │
│  │ CLAHE (LAB)      │     │                   │
│  │ Resize 224×224   │     │                   │
│  │ Normalize        │     │                   │
│  └────────┬────────┘     │                   │
│           │               │                   │
│           ▼               │                   │
│  ┌──────────────────┐    │                   │
│  │ TTA (5 augs)      │    │                   │
│  │  → Model forward  │    │                   │
│  │  → Average preds  │    │                   │
│  └────────┬─────────┘    │                   │
│           │               │                   │
│           ▼               │                   │
│  ┌──────────────────┐    │                   │
│  │ Result per face   │    │                   │
│  │  age, gender, race│    │                   │
│  │  + confidence     │    │                   │
│  └──────────────────┘    │                   │
│      └───────────────────┘                   │
│                                              │
│  Output: List[{age, gender, race, bbox, conf}]│
└─────────────────────────────────────────────┘
```

---

## 3.11 Thiết kế giao diện người dùng (UI/UX)

### 3.11.1 Bố cục tổng thể

Ứng dụng Streamlit sử dụng **3 tab chính** trong sidebar navigation:

| Tab | Tên | Mô tả |
|-----|-----|-------|
| 📸 | **Dự đoán ảnh** | Upload / Webcam / Clipboard → Dự đoán + hiển thị kết quả |
| 📹 | **Camera trực tiếp** | Real-time detection + prediction overlay |
| 📊 | **Biểu đồ đánh giá** | Dashboard phân tích hiệu năng mô hình |

### 3.11.2 Thiết kế Tab 1: Dự đoán ảnh

```
┌──────────────────────────────────────────────┐
│  📸 FaceVision AI — Dự đoán thuộc tính       │
│──────────────────────────────────────────────│
│                                              │
│  ┌────────┐ ┌────────┐ ┌──────────┐         │
│  │Upload  │ │Webcam  │ │Clipboard │         │
│  │ Image  │ │Capture │ │  Paste   │         │
│  └────────┘ └────────┘ └──────────┘         │
│                                              │
│  ┌──────────────────┐ ┌──────────────────┐  │
│  │   Ảnh gốc        │ │   Ảnh kết quả    │  │
│  │   (col 1)        │ │   (col 2)        │  │
│  │                  │ │  [BBox + Label]  │  │
│  │                  │ │                  │  │
│  └──────────────────┘ └──────────────────┘  │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  Kết quả (metric cards):             │    │
│  │  🎂 Tuổi: 25  │ 👤 Nữ  │ 🌍 Asian  │    │
│  │     ±3 năm    │  95%   │    92%     │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

### 3.11.3 Thiết kế Tab 3: Biểu đồ đánh giá

Hiển thị 13 biểu đồ tương tác Plotly, được render trực tiếp từ `logs/metrics.json`:

1. 📈 Training Curves (Loss + MAE + Gender + Race) — 4-in-1
2. 📉 Loss Curve chi tiết
3. 🎂 Age MAE Curve
4. 📊 Gender + Race Accuracy Curves
5. 👤 Gender Confusion Matrix (heatmap)
6. 🌍 Race Confusion Matrix (heatmap)
7. 👤 Gender Precision/Recall/F1 (grouped bar)
8. 🌍 Race Precision/Recall/F1 (grouped bar)
9. 🎂 Age Error by Group (bar + line)
10. 🎯 Radar hiệu năng tổng quan (polar)
11. 📊 Accuracy Overview (target line)
12. 📦 Dataset Distribution (3 donuts)
13. 🏆 Score Card (final grade)

### 3.11.4 Thiết kế CSS

Giao diện sử dụng custom CSS inject qua `st.markdown()`, bao gồm:
- Theme màu Indigo (#4F46E5) làm primary
- Font sans-serif (Inter, Segoe UI)
- Metric cards với gradient border
- Responsive layout 2 cột
- Smooth transitions và hover effects
