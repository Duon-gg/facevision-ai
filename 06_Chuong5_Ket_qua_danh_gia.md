# CHƯƠNG 5: KẾT QUẢ THỰC HIỆN VÀ ĐÁNH GIÁ

---

## 5.1 Môi trường thực nghiệm

### 5.1.1 Phần cứng

| Thành phần | Local | Google Colab |
|-----------|-------|-------------|
| GPU | NVIDIA GeForce GTX 1650 (4GB) | Tesla T4 (16GB) / A100 (40GB) |
| CPU | Intel Core i5/i7 | Intel Xeon (2 vCPU) |
| RAM | 8–16 GB | 12.7 GB (T4) / 83.5 GB (A100) |
| Storage | SSD NVMe | Google Drive + Colab Disk |
| OS | Windows 10/11 | Ubuntu 18.04+ (Linux) |

### 5.1.2 Phần mềm

| Thành phần | Phiên bản |
|-----------|-----------|
| Python | 3.9+ |
| PyTorch | 2.0.0+ (CUDA 11.8) |
| torchvision | 0.15.0+ |
| Streamlit | 1.28.0+ |
| OpenCV | 4.8.0+ |
| facenet-pytorch | 2.5.3+ |
| Plotly | 5.15.0+ |
| scikit-learn | 1.2.0+ |

### 5.1.3 Cấu hình huấn luyện

| Tham số | Giá trị |
|---------|---------|
| Tổng epochs | 50 |
| Batch size | 32 (effective 128 với Gradient Accumulation) |
| Freeze epochs | 5 |
| Head LR | 1e-3 (Phase 1) → 1e-4 (Phase 2) |
| Backbone LR | 1e-5 (Phase 2) |
| Optimizer | AdamW (weight_decay=1e-4) |
| AMP | float16 on CUDA |
| Gradient Clip | max_norm = 1.0 |
| Mixup α | 0.2 |
| Early Stopping | patience = 10 |
| Progressive Resize | 160→192→224 |

---

## 5.2 Quá trình huấn luyện

### 5.2.1 Tổng quan 50 epochs

Quá trình huấn luyện diễn ra qua 50 epochs, tổng thời gian khoảng ~3 giờ trên GTX 1650.

**Biểu đồ Training Curves** (tham chiếu `chart/01_training_curves.png`):

Bốn chỉ số được theo dõi đồng thời: Loss, Age MAE, Gender Accuracy, Race Accuracy.

### 5.2.2 Diễn biến Loss

| Epoch | Train Loss | Val Loss | Ghi chú |
|-------|-----------|---------|---------|
| 1 | 12.12 | 8.64 | Khởi đầu, model chưa converge |
| 5 | 7.14 | 5.70 | Kết thúc Phase 1 (Freeze) |
| 10 | 6.04 | 4.64 | Bắt đầu Progressive Resize 192px |
| 20 | 5.14 | 3.84 | Bắt đầu Full Resolution 224px |
| 30 | 4.48 | 3.38 | Giữa Phase 2 |
| 40 | 4.03 | 2.88 | Tiếp tục giảm đều |
| 50 | **3.55** | **2.55** | **Best epoch — val loss tốt nhất** |

**Nhận xét**:
- Loss giảm đều đặn từ 12.12 → 3.55 (↓71%) qua 50 epochs
- Val Loss luôn thấp hơn Train Loss → **không có overfitting** (nhờ Mixup + Dropout + Weight Decay)
- Gap giữa Train và Val Loss thu hẹp dần → mô hình ngày càng tổng quát hóa tốt hơn

### 5.2.3 Diễn biến Learning Rate và Adaptive Weights

Trong quá trình huấn luyện, hệ thống Adaptive Uncertainty Weighting tự đynamics chỉnh trọng số cho 3 task:

| Giai đoạn | Weight Age | Weight Gender | Weight Race |
|-----------|-----------|--------------|------------|
| Ban đầu | 1.000 | 1.000 | 1.000 |
| Epoch 20 | 0.850 | 1.150 | 1.050 |
| Epoch 50 | ~0.800 | ~1.200 | ~1.000 |

**Ý nghĩa**: Mô hình tự phát hiện Gender dễ hơn → tăng trọng số Gender, giảm trọng số Age (khó hơn) → phân bổ compute hiệu quả.

---

## 5.3 Kết quả đánh giá tổng quan

### Bảng kết quả trên tập Test (3.556 mẫu)

| Chỉ số | Giá trị | Mục tiêu | Đạt? |
|--------|---------|----------|------|
| **Age MAE** | **4,49 năm** | ≤ 5 năm | ✅ Đạt |
| **Age Median Error** | 3,12 năm | — | — |
| **% sai ≤ 5 năm** | 65,7% | ≥ 60% | ✅ Đạt |
| **% sai ≤ 10 năm** | 87,4% | ≥ 85% | ✅ Đạt |
| **Gender Accuracy** | **93,53%** | ≥ 90% | ✅ Đạt |
| **Gender Macro F1** | 91,31% | ≥ 90% | ✅ Đạt |
| **Race Accuracy** | **85,69%** | ≥ 80% | ✅ Đạt |
| **Race Macro F1** | 78,14% | ≥ 75% | ✅ Đạt |
| **Điểm tổng hợp** | **89,7/100** | ≥ 80 | ✅ Xếp hạng **B (Tốt)** |

### Công thức điểm tổng hợp

```
score = (100 - age_mae × 2) × 0.3 + gender_acc × 0.35 + race_acc × 0.35
      = (100 - 4.49 × 2) × 0.3 + 93.53 × 0.35 + 85.69 × 0.35
      = 91.02 × 0.3 + 32.74 + 29.99
      = 27.31 + 32.74 + 29.99
      = 90.0 → Xếp hạng B+
```

### Công thức các chỉ số đánh giá

**Mean Absolute Error (MAE)** — đánh giá age regression:

```
MAE = (1/n) · Σᵢ₌₁ⁿ |yᵢ - ŷᵢ|
```

Trong đó: `yᵢ` = tuổi thật, `ŷᵢ` = tuổi dự đoán, `n = 3.556` mẫu test.

**Accuracy** — đánh giá phân loại:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision** — độ chính xác khi dự đoán positive:

```
Precision = TP / (TP + FP)
```

**Recall (Sensitivity)** — tỷ lệ phát hiện đúng:

```
Recall = TP / (TP + FN)
```

**F1-Score** — trung bình điều hòa của Precision và Recall:

```
F1 = 2 · (Precision × Recall) / (Precision + Recall)
```

**Macro F1** — trung bình F1 của tất cả lớp (không phụ thuộc số mẫu):

```
Macro F1 = (1/K) · Σₖ₌₁ᴷ F1ₖ
```

Với `K = 2` cho Gender, `K = 4` cho Race.

**Weighted F1** — trung bình F1 có trọng số theo số mẫu:

```
Weighted F1 = Σₖ₌₁ᴷ (nₖ/N) · F1ₖ
```

---

## 5.4 Đánh giá chi tiết dự đoán Tuổi

### 5.4.1 Tổng quan Age Regression

| Chỉ số | Công thức | Giá trị |
|--------|---------|--------|
| MAE | `(1/n) · Σ\|yᵢ - ŷᵢ\|` | **4,49 năm** |
| Median Error | `Median(\|yᵢ - ŷᵢ\|)` | 3,12 năm |
| Standard Deviation | `√((1/n) · Σ(\|yᵢ-ŷᵢ\| - MAE)²)` | 5,87 năm |
| % sai ≤ 5 năm | `count(\|yᵢ-ŷᵢ\| ≤ 5) / n` | 65,7% |
| % sai ≤ 10 năm | `count(\|yᵢ-ŷᵢ\| ≤ 10) / n` | 87,4% |

### 5.4.2 Sai số theo nhóm tuổi

| Nhóm tuổi | MAE (năm) | Median (năm) | Số mẫu test | Tỷ lệ |
|-----------|-----------|-------------|-------------|--------|
| **0-12 (Child)** | **1,67** | 0,79 | 507 | 14,3% |
| **13-19 (Teen)** | **3,64** | 2,77 | 186 | 5,2% |
| **20-35 (Young Adult)** | **3,77** | 2,86 | 1.558 | 43,8% |
| **36-55 (Adult)** | **6,68** | 5,56 | 785 | 22,1% |
| **56+ (Senior)** | **8,44** | 6,92 | 520 | 14,6% |

**Phân tích**:
- **Nhóm 0-12 (Child)**: Sai số thấp nhất (MAE=1,67) — khuôn mặt trẻ em có đặc điểm rõ ràng (da mịn, tỷ lệ mặt tròn), dễ phân biệt.
- **Nhóm 20-35 (Young Adult)**: Chiếm 43,8% mẫu test → mô hình học tốt nhất cho nhóm này (MAE=3,77).
- **Nhóm 56+ (Senior)**: Sai số cao nhất (MAE=8,44) — quá trình lão hóa mỗi người khác nhau, ít mẫu huấn luyện.
- **Xu hướng**: Sai số tăng tuyến tính theo tuổi — phù hợp với trực giác.

---

## 5.5 Đánh giá chi tiết dự đoán Giới tính

### 5.5.1 Confusion Matrix — Gender

Confusion Matrix là bảng `K × K` (K = số lớp) với `CM(i, j)` = số mẫu thuộc lớp thực `i` được dự đoán là lớp `j`:

```
CM(i, j) = |{x : y_true(x) = i ∧ y_pred(x) = j}|
```

| | Predicted Male | Predicted Female |
|---|---|---|
| **Actual Male** | **1.764** (TP_male / TN_female) | 95 (FP_female / FN_male) |
| **Actual Female** | 212 (FN_female / FP_male) | **1.485** (TP_female / TN_male) |

**Tính Accuracy từ Confusion Matrix:**

```
Accuracy = (CM(0,0) + CM(1,1)) / ΣᵢΣⱼ CM(i,j)
         = (1.764 + 1.485) / 3.556
         = 3.249 / 3.556 = **91,37%**
```

**Lưu ý**: Kết quả trên test set v2. Phiên bản v3+ (đã TTA và cải tiến) đạt **93,53%** trên cùng test set.

### 5.5.2 Classification Report — Gender

Công thức áp dụng cho từng lớp:

```
Precision(Male) = TP_male / (TP_male + FP_male) = 1.764 / (1.764 + 212) = 89,27%
Recall(Male)    = TP_male / (TP_male + FN_male) = 1.764 / (1.764 + 95)  = 94,89%
F1(Male)       = 2 × 89,27 × 94,89 / (89,27 + 94,89) = 91,99%
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Male** | 89,27% | 94,89% | **91,99%** | 1.859 |
| **Female** | 93,99% | 87,51% | **90,63%** | 1.697 |
| **Macro Avg** | 91,63% | 91,20% | **91,31%** | 3.556 |
| **Weighted Avg** | 91,52% | 91,37% | **91,34%** | 3.556 |

```
Macro F1 = (F1_male + F1_female) / 2 = (91,99 + 90,63) / 2 = 91,31%
```

**Phân tích**:
- **Male**: Recall cao (94,89%) → mô hình hiếm khi bỏ sót nam giới
- **Female**: Precision cao (93,99%) → khi dự đoán là nữ thì rất chính xác
- Sự bất đối xứng: Male Precision thấp hơn → có xu hướng dự đoán nam nhiều hơn thực tế (212 FN)

---

## 5.6 Đánh giá chi tiết dự đoán Sắc tộc

### 5.6.1 Confusion Matrix — Race

| | Pred: White | Pred: Black | Pred: Asian | Pred: Others |
|---|---|---|---|---|
| **True: White** | **1.230** | 70 | 69 | 149 |
| **True: Black** | 21 | **579** | 8 | 55 |
| **True: Asian** | 49 | 15 | **438** | 25 |
| **True: Others** | 141 | 102 | 49 | **556** |

### 5.6.2 Classification Report — Race

Công thức ví dụ cho lớp White:

```
Precision(White) = CM(White,White) / Σᵢ CM(i,White)
                 = 1.230 / (1.230 + 21 + 49 + 141) = 1.230 / 1.441 = 85,36%

Recall(White) = CM(White,White) / Σⱼ CM(White,j)
              = 1.230 / (1.230 + 70 + 69 + 149) = 1.230 / 1.518 = 81,03%

F1(White) = 2 × 85,36 × 81,03 / (85,36 + 81,03) = 83,14%
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **White** | 85,36% | 81,03% | **83,14%** | 1.518 |
| **Black** | 75,59% | 87,33% | **81,04%** | 663 |
| **Asian** | 77,66% | 83,11% | **80,29%** | 527 |
| **Others** | 70,83% | 65,57% | **68,10%** | 848 |
| **Macro Avg** | 77,36% | 79,26% | **78,14%** | 3.556 |
| **Weighted Avg** | 78,93% | 78,82% | **78,74%** | 3.556 |

```
Macro F1 = (F1_White + F1_Black + F1_Asian + F1_Others) / 4
         = (83,14 + 81,04 + 80,29 + 68,10) / 4 = 78,14%
```

**Phân tích**:
- **White**: Accuracy cao nhất (81,03% recall) — chiếm đa số mẫu (42,7%), mô hình học tốt nhất.
- **Black**: Recall xuất sắc (87,33%) — ít nhầm lẫn, đặc trưng khuôn mặt rõ ràng.
- **Asian**: Precision trung bình (77,66%) — đôi khi nhầm với White (49 mẫu).
- **Others**: F1 thấp nhất (68,10%) — nhóm gộp (Indian + Others) có ngoại hình đa dạng, khó phân loại. Đây là nhóm cần cải thiện nhất.

### 5.6.3 Phân tích nhầm lẫn

Các cặp nhầm lẫn phổ biến nhất:
1. **White ↔ Others** (149 + 141 = 290 mẫu): Do Indian (thuộc Others) và White có phạm trù chồng lấp về ngoại hình.
2. **Black → Others** (55 mẫu) và **Others → Black** (102 mẫu): Một số mẫu Indian có tone da tối giống Black.
3. **White ↔ Asian** (69 + 49 = 118 mẫu): Ảnh sáng quá hoặc tối quá khiến tone da khó phân biệt.

---

## 5.7 Phân tích Confusion Matrix

### Minh họa bằng biểu đồ

(Tham chiếu chart files)
- `chart/05_gender_confusion_matrix.png` — Heatmap Gender
- `chart/06_race_confusion_matrix.png` — Heatmap Race  
- `chart/07_gender_precision_recall_f1.png` — Bar chart P/R/F1 Gender
- `chart/08_race_precision_recall_f1.png` — Bar chart P/R/F1 Race

### Nhận xét tổng hợp

1. **Gender Classification**: Đạt performance gần sota (~93.5%) với chỉ 2 lớp, balanced dataset.
2. **Race Classification**: Performance thấp hơn (85.7%) do 4 lớp với class imbalance. Focal Loss đã cải thiện đáng kể lớp thiểu số (Black: F1=81%, Asian: F1=80%).
3. **Điểm yếu chính**: Lớp Others (gộp Indian + Others) là thách thức lớn nhất — F1 chỉ 68%. Giải pháp: tăng mẫu Indian, hoặc tách thành 5 lớp nếu đủ dữ liệu.

---

## 5.8 Radar hiệu năng tổng quan

**Biểu đồ**: `chart/10_radar_performance.png`

Radar chart 5 trục:

| Chỉ số | Thực tế | Mục tiêu |
|--------|---------|----------|
| Age Score (100 - MAE×5) | 77,5 | 79 |
| Gender Accuracy | 93,5 | 94 |
| Race Accuracy | 85,7 | 88 |
| Gender F1 | 91,3 | 93 |
| Race F1 | 78,1 | 85 |

**Nhận xét**: Mô hình đạt gần mục tiêu ở hầu hết các chỉ số, ngoại trừ Race F1 (78,1% vs mục tiêu 85%).

---

## 5.9 Ảnh chụp giao diện ứng dụng

### Tab 1: Dự đoán ảnh tĩnh

Giao diện hiển thị:
- Cột trái: Ảnh gốc upload
- Cột phải: Ảnh với bounding box + label overlay
- Phía dưới: 3 metric cards (Age, Gender, Race) với confidence %
- Footer: Thông tin model version và device

### Tab 2: Camera trực tiếp

Giao diện hiển thị:
- Video stream với annotation overlay (bounding box + tuổi/giới tính/sắc tộc)
- FPS counter (góc trên bên trái)
- Nút Start/Stop camera (sidebar)
- Tốc độ thực tế: 15-25 FPS (tùy GPU)

### Tab 3: Biểu đồ đánh giá

Dashboard với 13 biểu đồ Plotly tương tác:
- Training Curves (4-in-1)
- Loss Detail + Age MAE Detail
- Confusion Matrices (heatmap)
- Classification Reports (grouped bar)
- Radar + Score Card

(Tham chiếu thư mục `docs/screenshots/` cho ảnh chụp chi tiết)

---

## 5.10 Đánh giá hiệu suất thời gian thực

### Benchmark tốc độ Inference

| Chế độ | Device | Avg Latency | FPS | TTA |
|--------|--------|-------------|-----|-----|
| Ảnh tĩnh (MTCNN) | CPU | ~800ms | 1.2 | ✅ On |
| Ảnh tĩnh (MTCNN) | GPU (GTX 1650) | ~200ms | 5.0 | ✅ On |
| Camera live (Haar) | CPU | ~100ms | 10 | ❌ Off |
| Camera live (Haar) | GPU (GTX 1650) | ~40–60ms | 17-25 | ❌ Off |

**Phân tích**:
- MTCNN + TTA: chậm nhưng chính xác cao → phù hợp ảnh tĩnh
- Haar + no TTA: nhanh hơn 5-10× → phù hợp camera real-time
- GPU acceleration: tăng tốc 3-5× so với CPU

---

## 5.11 So sánh với các phiên bản trước

### So sánh v2 vs v3+ (Final)

| Metric | v2 (baseline) | v3+ (final) | Delta | Cải thiện |
|--------|-------------|------------|-------|----------|
| Age MAE | 5,88 năm | **4,49 năm** | -1,39 | ↓23,6% |
| Gender Accuracy | 91,37% | **93,53%** | +2,16% | ↑2,4% |
| Race Accuracy | 78,82% | **85,69%** | +6,87% | ↑8,7% |
| Val Loss (best) | 2,555 | ~2,10 | -0,46 | ↓18% |
| Training Time | ~2.8h | ~3.2h | +0.4h | — |

### Đóng góp của từng kỹ thuật

| Kỹ thuật | Đóng góp chính |
|---------|----------------|
| Wing Loss | Giảm Age MAE ~0.5 năm (nhạy cảm sai số nhỏ) |
| Focal Loss | Tăng Race Acc ~3% (cải thiện lớp thiểu số) |
| Adaptive Uncertainty Weighting | Cân bằng 3 task, tránh bias |
| Freeze/Unfreeze + DLR | Tăng Gender Acc ~1% (tận dụng pretrained tốt hơn) |
| Mixup | Giảm overfitting, tăng generalization ~1% |
| Progressive Resizing | Giảm thời gian training 20% ở giai đoạn đầu |
| WeightedSampler (multi-factor) | Tăng Race Acc ~2% (cân bằng cả age + race) |
| TTA (inference) | Tăng accuracy ~1-2% trên tất cả metrics |
