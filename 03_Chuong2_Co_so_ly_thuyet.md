# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ CÔNG NGHỆ SỬ DỤNG

---

## 2.1 Tổng quan về Thị giác Máy tính

Thị giác Máy tính (Computer Vision) là một nhánh của Trí tuệ Nhân tạo chuyên nghiên cứu cách máy tính "nhìn" và hiểu nội dung ảnh/video. Trong bối cảnh bài toán Face Attribute Prediction, Thị giác Máy tính đóng vai trò nền tảng, bao gồm các bước: phát hiện khuôn mặt (Face Detection), trích xuất đặc trưng (Feature Extraction), và phân loại/hồi quy thuộc tính (Attribute Prediction).

Các bước xử lý ảnh số (Digital Image Processing — DIP) truyền thống như cân bằng lược đồ xám (Histogram Equalization), lọc nhiễu (Noise Filtering), và chuẩn hóa kích thước (Resizing) vẫn đóng vai trò quan trọng trong pipeline tiền xử lý, cải thiện chất lượng đầu vào cho mô hình Học sâu.

---

## 2.2 Mạng nơ-ron tích chập (CNN)

### 2.2.1 Khái niệm

Mạng nơ-ron tích chập (Convolutional Neural Network — CNN) là kiến trúc mạng nơ-ron sâu chuyên biệt cho dữ liệu dạng lưới (grid-like data), đặc biệt hiệu quả với ảnh. CNN sử dụng phép tích chập (convolution) để tự động học các bộ lọc (filters/kernels) từ dữ liệu, trích xuất các đặc trưng có tính phân cấp — từ cạnh, góc ở lớp nông đến các đặc trưng ngữ nghĩa (mắt, mũi, hình dáng khuôn mặt) ở lớp sâu.

### 2.2.2 Các thành phần chính

1. **Lớp tích chập (Convolutional Layer)**: Thực hiện phép tích chập giữa ảnh đầu vào và kernel để tạo feature map. Mỗi kernel học cách phát hiện một đặc trưng cụ thể.

   **Công thức phép tích chập 2D:**

   ```
   Y(i, j) = Σₘ Σₙ X(i + m, j + n) · K(m, n) + b
   ```

   Trong đó:
   - `X`: ảnh đầu vào (input feature map)
   - `K`: kernel (bộ lọc) kích thước `m × n`
   - `b`: bias
   - `Y`: feature map đầu ra
   - `(i, j)`: vị trí pixel trên feature map đầu ra

2. **Lớp kích hoạt (Activation Layer)**: Hàm phi tuyến áp dụng sau lớp tích chập để tăng khả năng biểu diễn phi tuyến của mô hình.

   **Hàm ReLU (Rectified Linear Unit):**

   ```
   ReLU(x) = max(0, x)
   ```

   Đạo hàm:

   ```
   ReLU'(x) = { 1  nếu x > 0
              { 0  nếu x ≤ 0
   ```

   **Hàm Sigmoid** (dùng trong SE Block):

   ```
   σ(x) = 1 / (1 + e^(-x))
   ```

   Đạo hàm: `σ'(x) = σ(x) · (1 - σ(x))`

   **Hàm Softmax** (dùng trong Gender Head, Race Head):

   ```
   Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)     với j = 1, 2, ..., K
   ```

   Trong đó `zᵢ` là logit thứ `i`, `K` là số lớp (K=2 cho Gender, K=4 cho Race).

3. **Lớp gộp (Pooling Layer)**: Giảm kích thước không gian của feature map.

   **Max Pooling:**

   ```
   Y(i, j) = max { X(i·s + m, j·s + n) }     với m, n ∈ [0, k-1]
   ```

   **Average Pooling:**

   ```
   Y(i, j) = (1/k²) · Σₘ Σₙ X(i·s + m, j·s + n)
   ```

   **Global Average Pooling (GAP)** — dùng trong ResNet50:

   ```
   z_c = (1 / H × W) · Σᵢ Σⱼ X_c(i, j)
   ```

   Trong đó `z_c` là giá trị đại diện cho kênh `c`, `H × W` là kích thước spatial.

4. **Lớp chuẩn hóa batch (Batch Normalization — BN)**:

   **Công thức BatchNorm:**

   ```
   μ_B = (1/m) · Σᵢ xᵢ                          (mean của mini-batch)
   σ²_B = (1/m) · Σᵢ (xᵢ - μ_B)²                (variance của mini-batch)
   x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)               (chuẩn hóa)
   yᵢ = γ · x̂ᵢ + β                               (scale và shift)
   ```

   Trong đó:
   - `m`: kích thước mini-batch
   - `ε = 1e-5`: hằng số nhỏ tránh chia cho 0
   - `γ, β`: tham số learnable (scale, shift)

5. **Lớp kết nối đầy đủ (Fully Connected — FC)**:

   **Công thức FC Layer:**

   ```
   y = W · x + b
   ```

   Trong đó `W ∈ ℝ^(out × in)` là ma trận trọng số, `b ∈ ℝ^out` là bias.

6. **Lớp Dropout**: Ngẫu nhiên tắt nơ-ron trong quá trình huấn luyện để giảm overfitting.

   **Công thức Dropout:**

   ```
   Training:   ŷᵢ = rᵢ · yᵢ / (1 - p)     với rᵢ ~ Bernoulli(1 - p)
   Inference:  ŷᵢ = yᵢ                      (giữ nguyên, không dropout)
   ```

   Trong đó `p = 0.3` là tỷ lệ dropout. Phép chia cho `(1 - p)` đảm bảo kỳ vọng đầu ra không đổi giữa training và inference (inverted dropout).

---

## 2.3 Kiến trúc ResNet50

### 2.3.1 Giới thiệu

ResNet (Residual Network) được đề xuất bởi He et al. (2016) trong bài báo "Deep Residual Learning for Image Recognition" [1]. ResNet giải quyết vấn đề **vanishing gradient** trong mạng CNN rất sâu bằng cơ chế **skip connection** (hay residual connection).

### 2.3.2 Residual Block

Ý tưởng cốt lõi của ResNet là thay vì học ánh xạ `H(x)` trực tiếp, mạng sẽ học phần **residual** (dư) `F(x) = H(x) - x`, sau đó cộng lại với đầu vào.

**Công thức Residual Connection:**

```
y = F(x, {Wᵢ}) + x
```

Trong đó:
- `x`: đầu vào của block (identity shortcut)
- `F(x, {Wᵢ})`: hàm residual cần học (các lớp conv + BN + ReLU)
- `y`: đầu ra của block

**Bottleneck Block** (dùng trong ResNet50):

```
F(x) = W₃ · ReLU(BN(W₂ · ReLU(BN(W₁ · x))))
```

Gồm 3 lớp conv: `1×1` (giảm chiều) → `3×3` (tích chập) → `1×1` (tăng chiều).

**Đạo hàm qua skip connection:**

```
∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
```

Số hạng `+1` đảm bảo gradient luôn ≥ 1 → **không bị triệt tiêu** gradient dù mạng rất sâu (50–152 lớp).

### 2.3.3 Kiến trúc ResNet50

ResNet50 bao gồm 50 lớp, được tổ chức thành 5 giai đoạn (stages):

| Giai đoạn | Các lớp | Kích thước đầu ra | Số kênh |
|-----------|---------|-------------------|---------|
| conv1 | 7×7, stride 2 + MaxPool | 56×56 | 64 |
| conv2_x | 3 Bottleneck blocks | 56×56 | 256 |
| conv3_x | 4 Bottleneck blocks | 28×28 | 512 |
| conv4_x | 6 Bottleneck blocks | 14×14 | 1024 |
| conv5_x | 3 Bottleneck blocks | 7×7 | 2048 |
| Average Pool | Global Average Pooling | 1×1 | 2048 |

Trong đồ án, lớp FC cuối cùng (1000 lớp cho ImageNet) của ResNet50 được thay thế bằng `nn.Identity()`, giữ lại feature vector 2048 chiều để phục vụ 3 prediction heads riêng biệt (file `src/models/face_attribute_model.py`, dòng 62):

```python
self.backbone.fc = nn.Identity()
```

### 2.3.4 Transfer Learning

Trong đồ án, ResNet50 được khởi tạo với trọng số **pretrained trên ImageNet** (`ResNet50_Weights.IMAGENET1K_V2`), tận dụng kiến thức đã học từ 1,2 triệu ảnh của ImageNet. Việc này giúp mô hình hội tụ nhanh hơn và đạt kết quả tốt hơn trên bộ dữ liệu UTKFace tương đối nhỏ (~23K ảnh).

---

## 2.4 Cơ chế chú ý Squeeze-and-Excitation (SE)

### 2.4.1 Giới thiệu

Squeeze-and-Excitation Network (SE-Net) được đề xuất bởi Hu et al. (2018) trong bài báo "Squeeze-and-Excitation Networks" [2], đoạt giải nhất ILSVRC 2017. SE block thực hiện **channel-wise attention** — tức là gán trọng số khác nhau cho từng kênh (channel) của feature map, giúp mô hình tập trung vào các kênh mang thông tin quan trọng.

### 2.4.2 Cơ chế hoạt động

SE Block gồm 3 bước chính với công thức toán học:

**Bước 1 — Squeeze (Nén):** Global Average Pooling nén thông tin không gian:

```
z_c = F_sq(u_c) = (1 / H × W) · Σᵢ₌₁ᴴ Σⱼ₌₁ᵂ u_c(i, j)
```

Trong đó: `u_c` là feature map tại kênh `c` có kích thước `H × W`, `z_c` là scalar đại diện cho kênh `c`. Kết quả: vector `z ∈ ℝ^C` với `C = 2048` (đầu ra ResNet50).

**Bước 2 — Excitation (Kích hoạt):** Hai lớp FC học trọng số kênh:

```
s = F_ex(z, W) = σ(W₂ · δ(W₁ · z))
```

Trong đó:
- `W₁ ∈ ℝ^(C/r × C)` = `ℝ^(128 × 2048)` — giảm chiều (reduction)
- `δ` = ReLU — hàm kích hoạt phi tuyến
- `W₂ ∈ ℝ^(C × C/r)` = `ℝ^(2048 × 128)` — khôi phục chiều
- `σ` = Sigmoid — scale về [0, 1]
- `r = 16` — reduction ratio

**Bước 3 — Scale:** Nhân trọng số kênh với feature map ban đầu:

```
x̃_c = F_scale(u_c, s_c) = s_c · u_c
```

Trong đó `s_c ∈ [0, 1]` là trọng số kênh: kênh quan trọng → `s_c ≈ 1`, kênh không quan trọng → `s_c ≈ 0`.

### 2.4.3 Cài đặt trong đồ án

Trong file `src/models/face_attribute_model.py`, SE Block được cài đặt như sau (dòng 14–32):

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.excitation(x)  # [B, C]
        return x * scale
```

Tham số: `channels=2048` (số kênh đầu ra của ResNet50), `reduction=16` (giảm chiều 2048→128→2048).

---

## 2.5 Học đa nhiệm (Multi-Task Learning)

### 2.5.1 Khái niệm

Học đa nhiệm (Multi-Task Learning — MTL) là phương pháp huấn luyện một mô hình duy nhất trên nhiều nhiệm vụ liên quan đồng thời. Thay vì xây dựng 3 mô hình riêng biệt cho Age, Gender, Race, MTL sử dụng một shared backbone để trích xuất đặc trưng chung, sau đó phân nhánh thành các task-specific heads.

### 2.5.2 Ưu điểm của MTL

- **Chia sẻ biểu diễn (Shared Representation)**: Các nhiệm vụ liên quan (tuổi, giới tính, sắc tộc đều là thuộc tính khuôn mặt) có thể tận dụng các đặc trưng chung, cải thiện khả năng tổng quát hóa.
- **Tiết kiệm tài nguyên**: Chỉ cần một backbone duy nhất thay vì 3 backbone riêng biệt, giảm đáng kể số tham số và thời gian suy luận.
- **Regularization ẩn (Implicit Regularization)**: Việc tối ưu đồng thời nhiều nhiệm vụ hoạt động như một dạng regularization, giảm overfitting trên từng nhiệm vụ đơn lẻ.

### 2.5.3 Kiến trúc Hard Parameter Sharing

Đồ án sử dụng kiến trúc **Hard Parameter Sharing** — dạng phổ biến nhất của MTL.

**Công thức MTL Forward Pass:**

Gọi `f_shared(x; θ_s)` là backbone chung (ResNet50 + SE), `f_k(h; θ_k)` là head thứ `k`:

```
h = f_shared(x; θ_s)          h ∈ ℝ^2048       (shared features)
ŷ_age    = f_age(h; θ_age)     ŷ_age ∈ ℝ        (regression)
ŷ_gender = f_gender(h; θ_gen)  ŷ_gender ∈ ℝ²    (2-class logits)
ŷ_race   = f_race(h; θ_race)   ŷ_race ∈ ℝ⁴      (4-class logits)
```

**Sơ đồ kiến trúc:**

```
ResNet50 (shared backbone θ_s) → SE Block → 2048-dim feature h
    ├── Age Head (θ_age):    FC(2048→512→128→1)    regression
    ├── Gender Head (θ_gen): FC(2048→256→2)         classification
    └── Race Head (θ_race):  FC(2048→256→4)         classification
```

**Tổng tham số cần tối ưu:**

```
Θ = {θ_s, θ_SE, θ_age, θ_gen, θ_race, σ₁, σ₂, σ₃}
```

Trong đó `σ₁, σ₂, σ₃` là 3 tham số uncertainty learnable của Adaptive Weighting.

---

## 2.6 Hàm mất mát chuyên biệt

### 2.6.1 Wing Loss

#### Giới thiệu

Wing Loss được đề xuất bởi Feng et al. (2018) trong bài báo "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks" [3]. Wing Loss được thiết kế đặc biệt cho bài toán regression trên khuôn mặt, với đặc tính nhạy cảm hơn với sai số nhỏ so với SmoothL1 Loss.

#### Công thức

```
               ⎧ w · ln(1 + |x|/ε)   nếu |x| < w
Wing(x) =     ⎨
               ⎩ |x| - C             nếu |x| ≥ w
```

Trong đó:
- `x = ŷ - y` là sai số dự đoán (ŷ = predicted, y = ground truth)
- `w = 10.0`: ngưỡng chuyển đổi giữa logarithmic và linear
- `ε = 2.0`: tham số điều khiển độ cong của vùng logarithmic
- `C = w - w · ln(1 + w/ε)`: hằng số đảm bảo tính liên tục tại `|x| = w`

#### Đạo hàm (Gradient)

```
              ⎧ w / (ε + |x|) · sign(x)     nếu |x| < w
∂Wing/∂x =   ⎨
              ⎩ sign(x)                      nếu |x| ≥ w
```

**Ý nghĩa đạo hàm:** Khi `|x|` nhỏ (sai số nhỏ), gradient ≈ `w/ε` = 10/2 = **5** → lớn hơn nhiều so với L1 Loss (gradient = 1) → mô hình **nhạy cảm hơn** với sai số nhỏ, giúp tinh chỉnh dự đoán tuổi chính xác.

#### So sánh gradient với các loss khác

| Loss Function | Gradient khi \|x\| nhỏ | Gradient khi \|x\| lớn |
|---|---|---|
| **L1 Loss** | 1 (hằng số) | 1 (hằng số) |
| **L2 (MSE) Loss** | 2x (→ 0 khi x→0) | 2x (tăng tuyến tính) |
| **SmoothL1 Loss** | x/β (→ 0 khi x→0) | 1 (hằng số) |
| **Wing Loss** | **w/(ε + \|x\|) ≈ 5** (lớn) | 1 (hằng số) |

#### Đặc tính

- Với sai số nhỏ (`|x| < w = 10`): sử dụng dạng logarithmic → gradient lớn → nhạy cảm với sai số nhỏ → tinh chỉnh dự đoán chính xác.
- Với sai số lớn (`|x| ≥ w = 10`): sử dụng dạng tuyến tính → gradient = 1 → ổn định, tránh bùng nổ gradient.

#### Cài đặt (file `src/losses/multi_task_loss.py`, dòng 40–65)

```python
class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)

    def forward(self, pred, target):
        x = pred - target
        abs_x = torch.abs(x)
        abs_x = torch.clamp(abs_x, max=100.0)  # Prevent overflow
        log_term = self.w * torch.log1p(abs_x / self.epsilon)
        linear_term = abs_x - self.C
        loss = torch.where(abs_x < self.w, log_term, linear_term)
        return loss.mean()
```

### 2.6.2 Focal Loss

#### Giới thiệu

Focal Loss được đề xuất bởi Lin et al. (2017) trong bài báo "Focal Loss for Dense Object Detection" [4]. Focal Loss giải quyết vấn đề mất cân bằng lớp bằng cách giảm trọng số cho các mẫu dễ (easy examples) và tập trung vào các mẫu khó (hard examples).

#### Công thức

**Cross Entropy Loss chuẩn:**

```
CE(pₜ) = -log(pₜ)
```

**Focal Loss:**

```
FL(pₜ) = -αₜ · (1 - pₜ)^γ · log(pₜ)
```

Trong đó:
- `pₜ = Softmax(zₜ)` = xác suất dự đoán đúng lớp target
- `γ = 2.0` (focusing parameter): kiểm soát mức độ tập trung
- `αₜ`: trọng số lớp target (class weight)
- `(1 - pₜ)^γ`: modulating factor — giảm loss cho mẫu dễ

**Phân tích modulating factor `(1 - pₜ)^γ` với γ = 2:**

| pₜ (confidence) | (1-pₜ)² | Hiệu ứng |
|---|---|---|
| 0.9 (mẫu dễ) | 0.01 | Loss giảm **100×** → gần như bỏ qua |
| 0.5 (mẫu trung bình) | 0.25 | Loss giảm 4× |
| 0.1 (mẫu khó) | 0.81 | Loss giữ gần nguyên → tập trung học |

#### Label Smoothing

Kết hợp Label Smoothing để cải thiện generalization:

```
y_smooth = (1 - ε) · y_onehot + ε / K
```

Với `ε = 0.1`, `K = 4` (số lớp race):
- Nhãn "hard" `[1, 0, 0, 0]` → nhãn "smooth" `[0.925, 0.025, 0.025, 0.025]`
- Giúp mô hình không quá tự tin, cải thiện khả năng tổng quát hóa

#### Cài đặt (file `src/losses/multi_task_loss.py`, dòng 15–37)

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing,
            weight=self.class_weights
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

### 2.6.3 Adaptive Uncertainty Weighting

#### Giới thiệu

Phương pháp Adaptive Uncertainty Weighting được đề xuất bởi Kendall et al. (2018) trong bài báo "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" [5]. Thay vì chọn trọng số cố định cho mỗi task, phương pháp này cho mô hình **tự học** trọng số thông qua homoscedastic uncertainty.

#### Cơ sở toán học — Maximum Likelihood

Xuất phát từ multi-task likelihood:

```
p(y₁, y₂, y₃ | f(x)) = p(y₁ | f(x)) · p(y₂ | f(x)) · p(y₃ | f(x))
```

Giả sử mỗi task có phân bố Gaussian với variance σₖ²:

```
p(yₖ | f(x)) = N(fₖ(x), σₖ²)
```

Log-likelihood:

```
log p(y₁, y₂, y₃ | f(x)) = -1/(2σ₁²) · ‖y₁ - f₁(x)‖² - log(σ₁)
                            -1/(2σ₂²) · ‖y₂ - f₂(x)‖² - log(σ₂)
                            -1/(2σ₃²) · ‖y₃ - f₃(x)‖² - log(σ₃)
```

#### Công thức Loss tổng hợp

Minimize negative log-likelihood:

```
L_total = (1/2σ₁²) · L_age + (1/2σ₂²) · L_gender + (1/2σ₃²) · L_race
          + log(σ₁) + log(σ₂) + log(σ₃)
```

#### Biến đổi ổn định số học

Đặt `sₖ = log(σₖ²)` là tham số learnable → đảm bảo `σₖ² > 0` mà không cần ràng buộc:

```
precision_k = exp(-sₖ) = 1/σₖ²

L_total = 0.5 · exp(-s₁) · L_age    + 0.5 · s₁
        + 0.5 · exp(-s₂) · L_gender + 0.5 · s₂
        + 0.5 · exp(-s₃) · L_race   + 0.5 · s₃
```

**Cơ chế tự cân bằng:**
- Nếu `σₖ²` quá nhỏ → `exp(-sₖ)` lớn → loss task k bị khuếch đại → `sₖ` tăng để bù
- Nếu `σₖ²` quá lớn → `log(σₖ)` lớn → regularization term phạt → `sₖ` giảm
- Kết quả: optimizer tìm điểm cân bằng tối ưu cho mỗi task

#### Cài đặt (file `src/losses/multi_task_loss.py`, dòng 68–136)

```python
# Learnable sₖ = log(σₖ²), khởi tạo = 0 → σₖ² = 1 → weight = 0.5
self.log_var_age = nn.Parameter(torch.zeros(1))     # s₁
self.log_var_gender = nn.Parameter(torch.zeros(1))   # s₂
self.log_var_race = nn.Parameter(torch.zeros(1))     # s₃

# Trong forward():
precision_age = torch.exp(-self.log_var_age)         # 1/σ₁²
total = (0.5 * precision_age * l_age + 0.5 * self.log_var_age +
         0.5 * precision_gender * l_gender + 0.5 * self.log_var_gender +
         0.5 * precision_race * l_race + 0.5 * self.log_var_race)
```

---

## 2.7 Phát hiện khuôn mặt (Face Detection)

### 2.7.1 MTCNN (Multi-task Cascaded Convolutional Networks)

#### Giới thiệu

MTCNN được đề xuất bởi Zhang et al. (2016) [6] và là một trong những phương pháp phát hiện khuôn mặt phổ biến nhất. MTCNN sử dụng kiến trúc cascade gồm 3 mạng CNN:

1. **P-Net (Proposal Network)**: Mạng nông, quét ảnh ở nhiều scale, tạo ra các vùng ứng viên (candidate regions). Nhanh nhưng nhiều false positive.

2. **R-Net (Refine Network)**: Mạng trung bình, lọc bỏ các vùng ứng viên sai, tinh chỉnh bounding box. Giảm đáng kể false positive.

3. **O-Net (Output Network)**: Mạng sâu nhất, đưa ra kết quả cuối cùng gồm bounding box chính xác và 5 facial landmarks (2 mắt, mũi, 2 khóe miệng).

#### Ưu điểm

- Chính xác cao, hỗ trợ đa góc nhìn
- Phát hiện đồng thời nhiều khuôn mặt
- Bền vữm với ánh sáng yếu, khuôn mặt nghiêng nhẹ

#### Cài đặt

Sử dụng thư viện `facenet-pytorch` (file `src/detection/face_detector.py`, dòng 33–40):

```python
self.mtcnn = MTCNN(
    keep_all=True,             # Phát hiện tất cả khuôn mặt
    device=self.device,        # Hỗ trợ CUDA GPU
    min_face_size=30,          # Kích thước khuôn mặt tối thiểu 30px
    thresholds=[0.6, 0.7, 0.7],  # Ngưỡng cho P-Net, R-Net, O-Net
    post_process=False,
)
```

### 2.7.2 Haar Cascade (Fallback)

#### Giới thiệu

Haar Cascade là phương pháp phát hiện khuôn mặt cổ điển của Viola-Jones (2001), sử dụng đặc trưng Haar-like kết hợp bộ phân loại cascade AdaBoost. Mặc dù kém chính xác hơn MTCNN, Haar Cascade có tốc độ rất nhanh (~5ms/frame vs ~100ms cho MTCNN).

#### Vai trò trong hệ thống

Haar Cascade được sử dụng làm phương án dự phòng (fallback) trong hai trường hợp:
1. Khi MTCNN không khả dụng (thiếu thư viện `facenet-pytorch`)
2. Trong chế độ camera trực tiếp, cần tốc độ cao (~15-25 FPS)

#### Cài đặt (file `src/detection/face_detector.py`, dòng 112–124)

```python
def fast_detect_haar(self, image_np, min_size=(40, 40)):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = self.face_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=4, minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces.tolist() if len(faces) > 0 else []
```

---

## 2.8 Xử lý ảnh số (Digital Image Processing — DIP)

### 2.8.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

#### Giới thiệu

CLAHE là phiên bản cải tiến của Histogram Equalization, được thiết kế để cân bằng sáng cục bộ (local contrast enhancement). CLAHE chia ảnh thành các ô nhỏ (tiles), thực hiện histogram equalization riêng cho mỗi ô, với giới hạn contrast (clip limit) để tránh khuếch đại nhiễu.

#### Công thức Histogram Equalization chuẩn

```
T(rₖ) = (L - 1) · Σⱼ₌₀ᵏ p(rⱼ) = (L - 1) · Σⱼ₌₀ᵏ (nⱼ / N)
```

Trong đó:
- `rₖ`: mức xám thứ `k` (k = 0, 1, ..., L-1)
- `L = 256`: tổng số mức xám
- `p(rⱼ) = nⱼ / N`: xác suất của mức xám `rⱼ`
- `nⱼ`: số pixel có mức xám `rⱼ`
- `N = H × W`: tổng số pixel
- `T(rₖ)`: mức xám mới sau equalization (CDF × (L-1))

#### Cải tiến CLAHE so với HE chuẩn

1. **Adaptive**: Chia ảnh thành lưới `M × N` tiles (8×8 = 64 tiles), equalize riêng mỗi tile
2. **Contrast Limited**: Giới hạn histogram tại `clipLimit = β` → cắt bớt phần vượt quá → phân bổ lại đều

```
Nếu h(rₖ) > β:  phần dư = h(rₖ) - β  → phân bổ đều cho các bin khác
```

3. **Bilinear Interpolation**: Nội suy song tuyến giữa các tile → tránh hiệu ứng block artifact

#### Quy trình trong đồ án

1. Chuyển ảnh từ RGB sang không gian màu **LAB**: `L ∈ [0, 255]`, `A ∈ [-128, 127]`, `B ∈ [-128, 127]`
2. Tách kênh **L** (Lightness — chỉ chứa thông tin độ sáng)
3. Áp dụng CLAHE trên kênh L với `clipLimit=2.0` và `tileGridSize=(8,8)`
4. Ghép lại 3 kênh LAB (A, B giữ nguyên — bảo toàn màu sắc)
5. Chuyển về RGB

**Lý do xử lý trên LAB thay vì RGB:** Không gian LAB tách riêng độ sáng (L) khỏi màu sắc (A, B) → CLAHE chỉ thay đổi contrast mà **không làm méo màu**.

#### Cài đặt (file `src/preprocessing/image_processing.py`, dòng 35–54)

```python
def apply_histogram_equalization(image_np, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

### 2.8.2 Gaussian Blur

Gaussian Blur sử dụng kernel Gaussian 2D để làm mờ ảnh, giảm nhiễu.

**Công thức kernel Gaussian 2D:**

```
G(x, y) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)
```

Trong đó:
- `(x, y)`: tọa độ pixel so với tâm kernel
- `σ`: độ lệch chuẩn (khi `σ = 0`, OpenCV tự tính `σ = 0.3 × ((ksize-1) × 0.5 - 1) + 0.8`)

**Ví dụ kernel 3×3** (xấp xỉ, đã chuẩn hóa):

```
G = (1/16) × ⎡ 1  2  1 ⎤
              ⎢ 2  4  2 ⎥
              ⎣ 1  2  1 ⎦
```

**Phép tích chập với Gaussian kernel:**

```
I'(x, y) = Σₘ Σₙ I(x+m, y+n) · G(m, n)
```

Trong đồ án, Gaussian Blur là **tùy chọn** (mặc định tắt), với `kernel_size = 3`.

```python
def apply_gaussian_blur(image_np, kernel_size=3):
    return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
```

### 2.8.3 Chuẩn hóa ImageNet (Normalize)

Sau khi chuyển ảnh thành tensor [0, 1], chuẩn hóa theo thống kê ImageNet:

```
x_norm = (x - μ) / σ
```

Với giá trị cho từng kênh RGB:

| Kênh | μ (mean) | σ (std) |
|------|---------|--------|
| R | 0.485 | 0.229 |
| G | 0.456 | 0.224 |
| B | 0.406 | 0.225 |

Lý do: backbone ResNet50 pretrained trên ImageNet → đầu vào phải cùng phân bố thống kê.

### 2.8.4 Pipeline xử lý ảnh tổng hợp

Pipeline DIP trong đồ án (file `src/preprocessing/image_processing.py`, hàm `preprocess_image`, dòng 95–125):

```
Ảnh đầu vào → Ensure RGB → CLAHE (LAB) → Gaussian Blur (tùy chọn) → Resize 224×224
```

---

## 2.9 Các kỹ thuật huấn luyện nâng cao

### 2.9.1 Transfer Learning và Freeze/Unfreeze

Chiến lược huấn luyện 2 pha:
- **Pha 1 (Epoch 1–5)**: Đóng băng (freeze) toàn bộ backbone ResNet50, chỉ huấn luyện SE block + 3 prediction heads, sử dụng `HEAD_LR = 1e-3`.
- **Pha 2 (Epoch 6+)**: Mở khóa (unfreeze) backbone, sử dụng **Differential Learning Rate**: backbone LR = `1e-5` (nhỏ hơn 100× so với heads LR ban đầu) để tránh phá hỏng trọng số pretrained.

### 2.9.2 Mixup Augmentation

Mixup (Zhang et al., 2018) là kỹ thuật tăng cường dữ liệu (data augmentation) trộn ảnh và nhãn của 2 mẫu ngẫu nhiên trong batch.

**Công thức Mixup:**

```
x̃ = λ · xᵢ + (1 - λ) · xⱼ
ỹ = λ · yᵢ + (1 - λ) · yⱼ
```

Trong đó:
- `xᵢ, xⱼ`: 2 ảnh ngẫu nhiên trong batch
- `yᵢ, yⱼ`: nhãn tương ứng
- `λ ~ Beta(α, α)` với `α = 0.2`

**Phân bố Beta(0.2, 0.2):**

```
f(λ; α, α) = λ^(α-1) · (1-λ)^(α-1) / B(α, α)
```

Với α = 0.2, phân bố Beta tạo ra `λ` thường gần 0 hoặc gần 1 → ảnh trộn vẫn giữ được đặc trưng chính, chỉ "pha nhẹ" ảnh khác.

**Mixup Loss:**

```
L_mixup = λ · L(ŷ, yᵢ) + (1 - λ) · L(ŷ, yⱼ)
```

Áp dụng cho 70% batches (`random() > 0.3`). Ngoài ra có **Minority-Only Mixup** — chỉ trộn ảnh lớp thiểu số (Black, Asian) để tránh bias.

### 2.9.3 Progressive Resizing

Kỹ thuật tăng dần kích thước ảnh đầu vào qua quá trình huấn luyện:

| Giai đoạn | Epoch | Kích thước ảnh | Tốc độ (relative) |
|-----------|-------|----------------|-------------------|
| Phase 1 | 1–9 | 160×160 | 1.96× nhanh hơn |
| Phase 2 | 10–19 | 192×192 | 1.36× nhanh hơn |
| Phase 3 | 20–50 | 224×224 | 1.00× (baseline) |

**Công thức tỷ lệ tốc độ** (xấp xỉ):

```
speed_ratio ≈ (224/img_size)²

Phase 1: (224/160)² = 1.96×
Phase 2: (224/192)² = 1.36×
```

Lợi ích: ảnh nhỏ → ít FLOP → training nhanh ở giai đoạn đầu; ảnh lớn ở giai đoạn sau → mô hình học thêm chi tiết.

### 2.9.4 Gradient Accumulation

Tích lũy gradient qua `N` batches trước khi thực hiện optimizer step, tăng effective batch size mà không cần thêm GPU memory.

**Công thức Gradient Accumulation:**

```
g_accumulated = (1/N) · Σₜ₌₁ᴺ ∇L(θ; Bₜ)

θ ← θ - η · g_accumulated
```

Trong đó:
- `N = 4` (accumulation steps)
- `Bₜ`: mini-batch thứ `t` (size = 32)
- `η`: learning rate
- Effective batch size = `32 × 4 = 128`

**Trong code, loss được scale trước khi backward:**

```
loss_scaled = loss / N     (đảm bảo gradient trung bình đúng)
```

Cài đặt (file `scripts/train_v3.py`, dòng 136–155):

```python
loss = loss / accumulation_steps  # Scale loss
scaler.scale(loss).backward()
if (step + 1) % accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 2.9.5 WeightedRandomSampler

Cân bằng phân bố lớp trong training set bằng cách lấy mẫu có trọng số.

**Công thức tính trọng số multi-factor:**

```
w_race(c) = N_total / (C_race × n_c)          (nghịch đảo tần suất race)
w_age(g)  = N_total / (C_age × n_g)           (nghịch đảo tần suất age group)
w_sample(i) = √(w_race(cᵢ) × w_age(gᵢ))      (geometric mean)
```

Trong đó:
- `N_total = 16.592` (tổng mẫu training)
- `C_race = 4` (số lớp race), `C_age = 5` (số nhóm tuổi)
- `n_c`: số mẫu thuộc lớp race `c`
- `n_g`: số mẫu thuộc nhóm tuổi `g`
- `cᵢ, gᵢ`: race và age group của mẫu thứ `i`

**Xác suất lấy mẫu:**

```
P(sample i) = w_sample(i) / Σⱼ w_sample(j)
```

Lớp thiểu số có `w` lớn hơn → được lấy mẫu thường xuyên hơn.

### 2.9.6 Test-Time Augmentation (TTA)

Trong quá trình suy luận, mỗi khuôn mặt được dự đoán `T = 5` lần với các biến đổi khác nhau, kết quả được **lấy trung bình**.

**Công thức TTA:**

```
ŷ_TTA = (1/T) · Σₜ₌₁ᵀ f(Aₜ(x); θ)
```

Trong đó:
- `f(·; θ)`: mô hình với tham số θ
- `Aₜ(x)`: phép biến đổi thứ `t` áp dụng lên ảnh `x`
- `T = 5`: số augmentation

| t | Aₜ(x) | Mô tả | Công thức |
|---|--------|-------|----------|
| 1 | Identity | Ảnh gốc | `A₁(x) = x` |
| 2 | Horizontal Flip | Lật ngang | `A₂(x)(i,j) = x(i, W-1-j)` |
| 3 | Brightness +10% | Tăng sáng | `A₃(x) = x × 1.1` |
| 4 | Brightness -10% | Giảm sáng | `A₄(x) = x × 0.9` |
| 5 | Rotation 5° | Xoay | `A₅(x) = R(5°) · x` |

**Cho mỗi task:**

```
age_TTA     = (1/5) · Σₜ f_age(Aₜ(x))
gender_TTA  = argmax( (1/5) · Σₜ f_gender(Aₜ(x)) )
race_TTA    = argmax( (1/5) · Σₜ f_race(Aₜ(x)) )
```

Age lấy trung bình trực tiếp; Gender/Race lấy trung bình logits rồi argmax.

---

## 2.10 Công nghệ và thư viện sử dụng

| STT | Thành phần | Công nghệ | Phiên bản | Vai trò |
|-----|-----------|-----------|-----------|---------|
| 1 | Ngôn ngữ | **Python** | ≥ 3.9 | Ngôn ngữ lập trình chính |
| 2 | Deep Learning | **PyTorch** | ≥ 2.0.0 | Framework huấn luyện và suy luận |
| 3 | Computer Vision | **torchvision** | ≥ 0.15.0 | Pretrained models, transforms |
| 4 | Xử lý ảnh | **OpenCV** (opencv-python) | ≥ 4.8.0 | CLAHE, Gaussian Blur, Haar Cascade |
| 5 | Xử lý ảnh PIL | **Pillow** | ≥ 9.0.0 | Đọc/ghi ảnh, chuyển đổi format |
| 6 | Giao diện web | **Streamlit** | ≥ 1.28.0 | Xây dựng UI tương tác 3 chế độ |
| 7 | Biểu đồ | **Plotly** | ≥ 5.15.0 | Biểu đồ tương tác (line, bar, heatmap, radar, pie) |
| 8 | Phát hiện khuôn mặt | **facenet-pytorch** | ≥ 2.5.3 | MTCNN face detection |
| 9 | Xử lý dữ liệu | **NumPy** | ≥ 1.24.0 | Mảng số, tính toán ma trận |
| 10 | Xử lý dữ liệu | **Pandas** | ≥ 1.5.0 | Đọc CSV, thao tác DataFrame |
| 11 | Machine Learning | **scikit-learn** | ≥ 1.2.0 | Train/test split, confusion matrix, classification report |
| 12 | GPU | **CUDA** | ≥ 11.8 | Tăng tốc huấn luyện và suy luận trên GPU NVIDIA |
| 13 | Cloud | **Google Colab** | — | Huấn luyện trên GPU Tesla T4/A100 |
| 14 | Quản lý mã nguồn | **Git** | — | Version control |
| 15 | License | **MIT** | — | Giấy phép mã nguồn mở |

### Tài liệu cấu hình thư viện

File `requirements.txt` (gốc project):

```
torch>=2.0.0
torchvision>=0.15.0
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=1.5.0
Pillow>=9.0.0
plotly>=5.15.0
scikit-learn>=1.2.0
facenet-pytorch>=2.5.3
```

File `.streamlit/config.toml` (cấu hình giao diện Streamlit):

```toml
[theme]
primaryColor = "#4F46E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1E293B"
font = "sans serif"

[server]
maxUploadSize = 10
```
