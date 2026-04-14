# TÀI LIỆU THAM KHẢO

---

## Bài báo khoa học

[1] **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778. DOI: 10.1109/CVPR.2016.90.
> *Vai trò trong đồ án*: Kiến trúc ResNet50 backbone — 50 lớp với residual connections, pretrained trên ImageNet.

[2] **Hu, J., Shen, L., & Sun, G.** (2018). "Squeeze-and-Excitation Networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7132–7141. DOI: 10.1109/CVPR.2018.00745.
> *Vai trò trong đồ án*: SE Block (channel attention) — nén 2048→128→2048 với Sigmoid gating.

[3] **Feng, Z.-H., Kittler, J., Awais, M., Huber, P., & Wu, X.-J.** (2018). "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2235–2245. DOI: 10.1109/CVPR.2018.00238.
> *Vai trò trong đồ án*: Wing Loss cho age regression — nhạy cảm với sai số nhỏ, w=10, ε=2.

[4] **Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). "Focal Loss for Dense Object Detection." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2999–3007. DOI: 10.1109/ICCV.2017.324.
> *Vai trò trong đồ án*: Focal Loss cho race classification — giải quyết class imbalance, γ=2.

[5] **Kendall, A., Gal, Y., & Cipolla, R.** (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7482–7491. DOI: 10.1109/CVPR.2018.00781.
> *Vai trò trong đồ án*: Adaptive Uncertainty Weighting — tự học trọng số log(σ²) cho 3 tasks.

[6] **Zhang, K., Zhang, Z., Li, Z., & Qiao, Y.** (2016). "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." *IEEE Signal Processing Letters*, 23(10), pp. 1499–1503. DOI: 10.1109/LSP.2016.2603342.
> *Vai trò trong đồ án*: MTCNN face detection — 3-stage cascade (P-Net → R-Net → O-Net).

[7] **Zhang, Z., Song, Y., & Qi, H.** (2017). "Age Progression/Regression by Conditional Adversarial Autoencoder." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5810–5818. DOI: 10.1109/CVPR.2017.463.
> *Vai trò trong đồ án*: Bộ dữ liệu UTKFace — 23.704 ảnh khuôn mặt với nhãn tuổi, giới tính, sắc tộc.

[8] **Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D.** (2018). "mixup: Beyond Empirical Risk Minimization." *International Conference on Learning Representations (ICLR)*.
> *Vai trò trong đồ án*: Mixup augmentation — trộn ảnh và nhãn, α=0.2.

[9] **Viola, P., & Jones, M.** (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features." *Proceedings of the IEEE Conference on CVPR*, pp. 511–518. DOI: 10.1109/CVPR.2001.990517.
> *Vai trò trong đồ án*: Haar Cascade face detection — fast fallback method.

## Tài liệu kỹ thuật và thư viện

[10] **PyTorch Documentation.** https://pytorch.org/docs/stable/
> Framework deep learning chính, hỗ trợ GPU acceleration và autograd.

[11] **torchvision Documentation.** https://pytorch.org/vision/stable/
> Pretrained models (ResNet50_Weights.IMAGENET1K_V2), data transforms (RandomCrop, ColorJitter, RandomErasing).

[12] **Streamlit Documentation.** https://docs.streamlit.io/
> Framework xây dựng giao diện web tương tác cho ứng dụng ML.

[13] **OpenCV Documentation.** https://docs.opencv.org/
> Thư viện xử lý ảnh: CLAHE, Gaussian Blur, Haar Cascade, color space conversion.

[14] **facenet-pytorch (Tim Esler).** https://github.com/timesler/facenet-pytorch
> Thư viện Python cung cấp MTCNN face detection cho PyTorch.

[15] **Plotly Documentation.** https://plotly.com/python/
> Thư viện biểu đồ tương tác: line, bar, heatmap, radar, pie charts.

[16] **scikit-learn Documentation.** https://scikit-learn.org/stable/
> train_test_split, confusion_matrix, classification_report, WeightedRandomSampler support.

---

# PHỤ LỤC

---

## Phụ lục A: Hướng dẫn cài đặt và chạy hệ thống

### A.1 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|-----------|-------------------|------------|
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.9+ | 3.10+ |
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc (CPU-only) | NVIDIA GTX 1650+ (CUDA 11.8+) |
| Disk | 5 GB | 10 GB (bao gồm dataset) |

### A.2 Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/[username]/facevision-ai.git
cd facevision-ai

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# hoặc: .\venv\Scripts\activate  # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. (Tùy chọn) Cài đặt PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### A.3 Chuẩn bị dữ liệu

```bash
# 1. Tải UTKFace dataset
# Đặt ảnh vào: data/UTKFace/

# 2. Tổ chức data
python scripts/organize_data.py
```

### A.4 Huấn luyện mô hình

```bash
# Training từ đầu (50 epochs)
python scripts/train_v3.py

# Resume training thêm 30 epochs
python scripts/train_v3.py --resume --extra-epochs 30

# Disable Mixup
python scripts/train_v3.py --no-mixup

# Disable Progressive Resizing (luôn 224px)
python scripts/train_v3.py --no-progressive
```

### A.5 Đánh giá mô hình

```bash
# Đánh giá trên test set
python scripts/eval.py

# Xuất biểu đồ
python scripts/export_charts.py
```

### A.6 Chạy ứng dụng

```bash
# Khởi động Streamlit
streamlit run app.py

# Mở trình duyệt tại: http://localhost:8501
```

### A.7 Chạy trên Google Colab

1. Upload `notebooks/train_colab.ipynb` lên Google Colab
2. Kết nối Runtime GPU (T4 hoặc A100)
3. Mount Google Drive
4. Chạy lần lượt các cell

---

## Phụ lục B: Mã nguồn chính

### B.1 Danh sách file mã nguồn

| # | File | Số dòng | Mô tả |
|---|------|---------|-------|
| 1 | `app.py` | 1.397 | Ứng dụng Streamlit chính |
| 2 | `src/models/face_attribute_model.py` | 139 | Kiến trúc CNN (ResNet50 + SE + 3 heads) |
| 3 | `src/data/dataset.py` | 192 | UTKFaceDataset + Mixup |
| 4 | `src/data/loader.py` | 175 | Data splitting + augmentation + sampler |
| 5 | `src/losses/multi_task_loss.py` | 137 | WingLoss + FocalLoss + Uncertainty Weighting |
| 6 | `src/detection/face_detector.py` | 149 | MTCNN + Haar Cascade |
| 7 | `src/inference/analyzer.py` | 466 | FaceAnalyzer: TTA + batch predict |
| 8 | `src/preprocessing/image_processing.py` | 126 | CLAHE + Gaussian Blur pipeline |
| 9 | `src/utils/constants.py` | 81 | Labels + config + age groups |
| 10 | `scripts/train_v3.py` | 639 | Training script v3 (full pipeline) |
| 11 | `scripts/eval.py` | 206 | Test set evaluation |
| 12 | `scripts/export_charts.py` | 639 | Xuất 13 biểu đồ PNG |
| 13 | `scripts/organize_data.py` | 211 | Data organization (split + class folders) |
| **Tổng** | **13 files** | **~4.557 dòng** | |

### B.2 Import Graph

```
app.py
  ├── src.inference.analyzer      → FaceAnalyzer
  │     ├── src.models.face_attribute_model → FaceAttributeModel
  │     ├── src.detection.face_detector     → FaceDetector
  │     ├── src.preprocessing.image_processing → ImagePreprocessor
  │     └── src.utils.constants              → LABELS, CONFIG
  └── plotly (charts)

scripts/train_v3.py
  ├── src.models.face_attribute_model → FaceAttributeModel
  ├── src.losses.multi_task_loss      → MultiTaskLoss
  ├── src.data.loader                 → load_data
  ├── src.data.dataset                → mixup_data, mixup_criterion
  └── src.utils.constants             → DEFAULT_CONFIG, LABELS

scripts/eval.py
  ├── src.models.face_attribute_model → FaceAttributeModel
  ├── src.data.loader                 → load_data
  └── src.utils.constants             → AGE_GROUPS, LABELS
```

---

## Phụ lục C: Cấu trúc dữ liệu

### C.1 Cấu trúc file checkpoint (.pth)

```python
{
    'epoch': 50,
    'model_state_dict': OrderedDict(...),   # ~26.2M params
    'optimizer_state_dict': {...},
    'criterion_state_dict': {
        'log_var_age': tensor([...]),         # Learnable weight
        'log_var_gender': tensor([...]),
        'log_var_race': tensor([...]),
    },
    'val_metrics': {
        'loss': 2.555,
        'age_mae': 11.63,
        'gender_acc': 89.96,
        'race_acc': 80.37,
    },
    'age_mode': 'regression',
    'num_race_classes': 4,
    'model_version': 'resnet50_se_v3',
    'img_size': 224,
}
```

### C.2 Cấu trúc file metrics.json

```python
{
    'best_epoch': 50,
    'best_val_loss': 2.555,
    'test_metrics': {
        'age_mae': 5.878,
        'gender_accuracy': 91.37,
        'race_accuracy': 78.82,
        'race_macro_f1': 78.14,
        'gender_confusion_matrix': [[1764, 95], [212, 1485]],
        'race_confusion_matrix': [[1230, 70, 69, 149], ...],
        'gender_classification_report': {...},
        'race_classification_report': {...},
    },
    'history': [
        {
            'epoch': 1,
            'time': 176.0,
            'train': {'loss': 12.12, 'age_mae': 24.11, 'gender_acc': 66.79, 'race_acc': 46.44},
            'val': {'loss': 8.64, 'age_mae': 18.10, 'gender_acc': 77.78, 'race_acc': 61.08},
        },
        # ... 50 epochs total
    ]
}
```

### C.3 Cấu trúc file labels.csv

```
image,age,gender,race
1_0_0_20170109150557335.jpg,1,0,0
25_1_2_20170116172557225.jpg,25,1,2
60_0_3_20170117142105283.jpg,60,0,3
...
```

Tổng: 23.704 dòng (dòng đầu là header).

---

## Phụ lục D: Danh sách biểu đồ đã xuất

| # | File | Kích thước | Mô tả |
|---|------|-----------|-------|
| 1 | `chart/01_training_curves.png` | 2400×1400px | 4-in-1: Loss, MAE, Gender Acc, Race Acc |
| 2 | `chart/02_loss_curve.png` | 2400×1000px | Loss chi tiết + best epoch marker |
| 3 | `chart/03_age_mae_curve.png` | 2400×1000px | Age MAE + target line ≤4.2 |
| 4 | `chart/04_accuracy_curves.png` | 2400×1000px | Gender + Race accuracy 2-panel |
| 5 | `chart/05_gender_confusion_matrix.png` | 2400×1100px | Heatmap gender (2×2) |
| 6 | `chart/06_race_confusion_matrix.png` | 2400×1100px | Heatmap race (4×4) |
| 7 | `chart/07_gender_precision_recall_f1.png` | 2400×1000px | P/R/F1 grouped bar |
| 8 | `chart/08_race_precision_recall_f1.png` | 2400×1000px | P/R/F1 grouped bar (4 classes) |
| 9 | `chart/09_age_error_by_group.png` | 2400×1100px | MAE + count per age group |
| 10 | `chart/10_radar_performance.png` | 2400×1200px | Radar 5 axes + target |
| 11 | `chart/11_accuracy_overview.png` | 2400×1000px | Overview bar + 90% target line |
| 12 | `chart/12_dataset_distribution.png` | 2400×900px | 3 donut charts (gender, race, age) |
| 13 | `chart/13_score_card.png` | 2400×1000px | Final score + grade |

Tất cả biểu đồ được render ở **2x resolution** (SCALE=2, tương đương ~300 DPI) trên nền **light theme** (background #FFFFFF, text #0F172A).

---

## Phụ lục E: Giấy phép

```
MIT License

Copyright (c) 2025 Nguyễn Văn Tùng Dương

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
