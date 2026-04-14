# CHƯƠNG 4: THIẾT KẾ CHI TIẾT VÀ CÀI ĐẶT

---

## 4.1 Module Models — Kiến trúc CNN

**File**: `src/models/face_attribute_model.py` (139 dòng)

### 4.1.1 Lớp SEBlock

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block cho channel-wise attention."""
    def __init__(self, channels, reduction=16):
        # channels = 2048 (đầu ra ResNet50)
        # reduction = 16 → bottleneck: 2048 → 128 → 2048
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # 2048→128
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),  # 128→2048
            nn.Sigmoid()  # Output: [0, 1] per channel
        )
```

**Giải thích từng dòng**:
- `AdaptiveAvgPool1d(1)`: Nén chiều không gian (spatial) thành 1 giá trị duy nhất cho mỗi kênh → vector [B, C].
- `Linear(2048, 128)`: Giảm chiều, học biểu diễn compact. `bias=False` vì BN ẩn trong kiến trúc đã xử lý bias.
- `ReLU(inplace=True)`: Kích hoạt phi tuyến, `inplace=True` tiết kiệm bộ nhớ.
- `Linear(128, 2048)`: Khôi phục chiều, học trọng số kênh.
- `Sigmoid()`: Scale trọng số về [0, 1], kênh quan trọng → gần 1, kênh không quan trọng → gần 0.

**Phương thức forward**:
```python
def forward(self, x):
    scale = self.excitation(x)  # [B, 2048] → [B, 2048]
    return x * scale  # Element-wise multiplication
```

### 4.1.2 Lớp FaceAttributeModel

```python
class FaceAttributeModel(nn.Module):
    def __init__(self, pretrained=True, age_mode='regression'):
```

**Khởi tạo backbone ResNet50**:
```python
weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
self.backbone = resnet50(weights=weights)
self.backbone.fc = nn.Identity()  # Loại bỏ FC 1000-class, giữ feature 2048
```

**Khởi tạo SE Block**:
```python
self.se = SEBlock(channels=2048, reduction=16)
```

**Age Head** (hồi quy — regression):
```python
self.age_head = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 1),   # Output: 1 giá trị tuổi
)
```

**Gender Head** (phân loại nhị phân):
```python
self.gender_head = nn.Sequential(
    nn.Linear(2048, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 2),   # Output: 2 logits (Male/Female)
)
```

**Race Head** (phân loại 4 lớp):
```python
self.race_head = nn.Sequential(
    nn.Linear(2048, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 4),   # Output: 4 logits (White/Black/Asian/Others)
)
```

**Phương thức forward**:
```python
def forward(self, x):
    features = self.backbone(x)    # [B, 2048]
    features = self.se(features)    # [B, 2048] (attention-weighted)
    
    age = self.age_head(features)
    age = torch.relu(age)           # Clamp tuổi ≥ 0
    age = torch.clamp(age, max=120) # Giới hạn tối đa 120 tuổi
    
    gender = self.gender_head(features)
    race = self.race_head(features)
    
    return age.squeeze(-1), gender, race
    # age: [B], gender: [B, 2], race: [B, 4]
```

---

## 4.2 Module Data — Dataset và DataLoader

### 4.2.1 Lớp UTKFaceDataset

**File**: `src/data/dataset.py` (192 dòng)

```python
class UTKFaceDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None,
                 preprocessor=None, use_mixup=False, mixup_alpha=0.2):
```

**Luồng xử lý trong `__getitem__`**:

1. **Đọc ảnh**: `Image.open(path).convert('RGB')` — đảm bảo 3 kênh RGB.
2. **Race remap**: Áp dụng `RACE_REMAP` (5 → 4 lớp) qua `RACE_REMAP.get(race, 3)`.
3. **Tiền xử lý DIP**: Nếu có `preprocessor` → gọi `preprocessor.full_pipeline(image)` (CLAHE + Blur).
4. **Augmentation**: Nếu có `transform` → gọi `transform(image)` (RandomHorizontalFlip, ColorJitter, v.v.).
5. **Trả về**: `(image_tensor, age, gender, race)`.

**Xử lý lỗi**: Nếu ảnh bị hỏng hoặc không load được, trả về mẫu "thay thế" (ảnh đen + nhãn mặc định) thay vì crash.

### 4.2.2 Mixup Augmentation

**Hàm `mixup_data`** (dòng 119–140):
```python
def mixup_data(images, ages, genders, races, alpha=0.2):
    lam = np.random.beta(alpha, alpha)          # λ ~ Beta(0.2, 0.2)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)  # Hoán vị ngẫu nhiên
    mixed_images = lam * images + (1 - lam) * images[index]
    return (mixed_images, ages, ages[index],
            genders, genders[index], races, races[index], lam)
```

**Hàm `mixup_criterion`** (dòng 146–160):
```python
def mixup_criterion(criterion, age_pred, gender_pred, race_pred,
                    ages_a, ages_b, genders_a, genders_b,
                    races_a, races_b, lam):
    loss_a = criterion(age_pred, gender_pred, race_pred, ages_a, genders_a, races_a)
    loss_b = criterion(age_pred, gender_pred, race_pred, ages_b, genders_b, races_b)
    return tuple(lam * a + (1 - lam) * b for a, b in zip(loss_a, loss_b))
```

**Hàm `mixup_data_minority`** (dòng 163–190) — Minority-Only Mixup:
- Chỉ trộn ảnh thuộc lớp race thiểu số (Black=1, Asian=2)
- Giữ nguyên ảnh lớp đa số (White=0, Others=3)
- Mục đích: Tăng cường dữ liệu lớp thiểu số mà không lãng phí compute cho lớp đã đủ mẫu

### 4.2.3 Module loader.py

**File**: `src/data/loader.py` (175 dòng)

**Hàm `load_data`** — entry point chính:

```python
def load_data(csv_path, img_dir, batch_size=32, num_workers=4, img_size=224):
```

**Cấu trúc Data Augmentation (Training)**:

```python
train_transform = transforms.Compose([
    transforms.Resize((img_size + 32, img_size + 32)),   # Resize lớn hơn
    transforms.RandomCrop(img_size),                       # Random crop
    transforms.RandomHorizontalFlip(p=0.5),                # Lật ngang 50%
    transforms.ColorJitter(brightness=0.2, contrast=0.2,   # Biến đổi màu
                           saturation=0.1, hue=0.05),
    transforms.RandomRotation(10),                         # Xoay ±10°
    transforms.RandomGrayscale(p=0.05),                    # Grayscale 5%
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),   # Random erasing 10%
])
```

**Val/Test Transform** (không augmentation):
```python
eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

**WeightedRandomSampler** (dòng 95–140):

```python
# Tính trọng số race
race_counts = Counter(all_races)
race_weights = {k: total / (NUM_RACE_CLASSES * v) for k, v in race_counts.items()}

# Tính trọng số age group
age_groups = [age_to_group(a) for a in all_ages]
group_counts = Counter(age_groups)
group_weights = {k: total / (NUM_AGE_GROUPS * v) for k, v in group_counts.items()}

# Kết hợp: sqrt(race_weight * age_group_weight)
sample_weights = [math.sqrt(race_weights[r] * group_weights[g])
                  for r, g in zip(all_races, age_groups)]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
```

---

## 4.3 Module Losses — Hàm mất mát

**File**: `src/losses/multi_task_loss.py` (137 dòng)

### 4.3.1 Lớp FocalLoss (dòng 15–37)

Chi tiết đã trình bày ở Mục 2.6.2. Tham số:
- `gamma=2.0`: Mức độ tập trung vào mẫu khó
- `label_smoothing=0.1`: Làm mượt nhãn (0/1 → 0.05/0.95)
- `class_weights`: Trọng số lớp (tùy chọn)

### 4.3.2 Lớp WingLoss (dòng 40–65)

Chi tiết đã trình bày ở Mục 2.6.1. Tham số:
- `w=10.0`: Ngưỡng chuyển đổi
- `epsilon=2.0`: Tham số độ cong
- `torch.clamp(abs_x, max=100.0)`: Tránh overflow khi tính `log1p`

### 4.3.3 Lớp MultiTaskLoss (dòng 68–137)

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, age_mode='regression', adaptive=True):
        # Wing Loss cho age
        self.age_loss = WingLoss(w=10.0, epsilon=2.0)
        
        # Cross Entropy cho gender (2 lớp cân bằng → không cần Focal)
        self.gender_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Focal Loss cho race (4 lớp mất cân bằng)
        self.race_loss = FocalLoss(gamma=2.0, label_smoothing=0.1)
        
        # Adaptive Uncertainty Weighting
        if adaptive:
            self.log_var_age = nn.Parameter(torch.zeros(1))
            self.log_var_gender = nn.Parameter(torch.zeros(1))
            self.log_var_race = nn.Parameter(torch.zeros(1))
```

**Forward** (Adaptive mode):
```python
def forward(self, age_pred, gender_pred, race_pred, age_true, gender_true, race_true):
    l_age = self.age_loss(age_pred, age_true.float())
    l_gender = self.gender_loss(gender_pred, gender_true)
    l_race = self.race_loss(race_pred, race_true)
    
    # Adaptive weighting — precision = exp(-log_var)
    p_age = torch.exp(-self.log_var_age)
    p_gender = torch.exp(-self.log_var_gender)
    p_race = torch.exp(-self.log_var_race)
    
    total = (0.5 * p_age * l_age + 0.5 * self.log_var_age +
             0.5 * p_gender * l_gender + 0.5 * self.log_var_gender +
             0.5 * p_race * l_race + 0.5 * self.log_var_race)
    
    return total, l_age, l_gender, l_race
```

**Ý nghĩa learnable parameters**:
- `log_var_age` khởi tạo = 0 → `precision = exp(0) = 1` → trọng số ban đầu = 0.5
- Trong quá trình training, optimizer tự điều chỉnh `log_var` cho mỗi task:
  - Task dễ → `log_var` tăng → precision giảm → giảm trọng số
  - Task khó → `log_var` giảm → precision tăng → tăng trọng số
- Kết quả thực nghiệm: Weight Age ≈ 0.8, Weight Gender ≈ 1.2, Weight Race ≈ 1.0

### 4.3.4 Optimizer — AdamW

Đồ án sử dụng **AdamW** (Adam với Weight Decay decoupled).

**Công thức cập nhật AdamW:**

```
mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ            (first moment — trung bình động)
vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²            (second moment — phương sai động)

m̂ₜ = mₜ / (1 - β₁ᵗ)                          (bias correction)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θₜ = θₜ₋₁ - η · (m̂ₜ / (√v̂ₜ + ε) + λ · θₜ₋₁)  (update với decoupled WD)
```

Trong đó:
- `gₜ = ∇L(θₜ₋₁)`: gradient tại bước `t`
- `β₁ = 0.9`, `β₂ = 0.999`: momentum coefficients
- `η`: learning rate (`1e-4` cho heads, `1e-5` cho backbone)
- `ε = 1e-8`: hằng số ổn định
- `λ = 1e-4`: weight decay

**Lý do chọn AdamW thay vì Adam/SGD:**
- **vs Adam:** Weight Decay trong Adam bị ghép với adaptive LR → hiệu ứng regularization không đều. AdamW tách riêng WD → regularization đều hơn.
- **vs SGD:** AdamW hội tụ nhanh hơn trên bài toán multi-task với nhiều loss scales khác nhau.

---

## 4.4 Module Detection — Phát hiện khuôn mặt

**File**: `src/detection/face_detector.py` (149 dòng)

### Lớp FaceDetector

```python
class FaceDetector:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Primary: MTCNN
        try:
            self.mtcnn = MTCNN(
                keep_all=True,
                device=device,
                min_face_size=30,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,
            )
        except Exception:
            self.mtcnn = None
        
        # Fallback: Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
```

**Phương thức `detect_faces`** (MTCNN primary):
```python
def detect_faces(self, image_pil, threshold=0.9):
    boxes, probs = self.mtcnn.detect(image_pil)
    # Lọc theo confidence threshold
    faces = [(box, prob) for box, prob in zip(boxes, probs) if prob >= threshold]
    return faces
```

**Phương thức `fast_detect_haar`** (Haar Cascade fallback):
```python
def fast_detect_haar(self, image_np, min_size=(40, 40)):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = self.face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,    # Hệ số scale image pyramid
        minNeighbors=4,       # Số neighbor tối thiểu
        minSize=min_size,     # Kích thước mặt tối thiểu
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces.tolist() if len(faces) > 0 else []
```

**So sánh hai phương pháp**:

| Tiêu chí | MTCNN | Haar Cascade |
|---------|-------|-------------|
| Độ chính xác | Cao (~99%) | Trung bình (~90%) |
| Tốc độ (CPU) | ~100ms/frame | ~5ms/frame |
| Detect nghiêng | Tốt (±30°) | Kém (chỉ frontal) |
| Landmark | Có (5 điểm) | Không |
| Dependency | facenet-pytorch | OpenCV built-in |
| Sử dụng | Ảnh tĩnh (UC01) | Camera live (UC02) |

---

## 4.5 Module Preprocessing — Xử lý ảnh số

**File**: `src/preprocessing/image_processing.py` (126 dòng)

### Lớp ImagePreprocessor

```python
class ImagePreprocessor:
    def __init__(self, target_size=224, use_clahe=True,
                 use_blur=False, clip_limit=2.0, tile_grid=(8,8)):
```

**Pipeline xử lý** (`full_pipeline`, dòng 95–125):

```python
def full_pipeline(self, image):
    """Pipeline: ensure RGB → CLAHE → Gaussian Blur → Resize"""
    # Step 1: Đảm bảo RGB
    image = self.ensure_rgb(image)
    
    # Step 2: Chuyển PIL → NumPy array
    image_np = np.array(image)
    
    # Step 3: CLAHE trên LAB
    if self.use_clahe:
        image_np = self.apply_histogram_equalization(image_np)
    
    # Step 4: Gaussian Blur (tùy chọn)
    if self.use_blur:
        image_np = self.apply_gaussian_blur(image_np)
    
    # Step 5: Resize về target_size×target_size
    image_np = cv2.resize(image_np, (self.target_size, self.target_size))
    
    # Step 6: Chuyển lại PIL Image
    return Image.fromarray(image_np)
```

**Hàm CLAHE** (`apply_histogram_equalization`):
```python
def apply_histogram_equalization(self, image_np, clip_limit=2.0, tile_grid=(8,8)):
    # LAB → tách kênh L → CLAHE → ghép lại → RGB
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
```

---

## 4.6 Module Inference — Pipeline suy luận

**File**: `src/inference/analyzer.py` (466 dòng)

### 4.6.1 Lớp FaceAnalyzer

```python
class FaceAnalyzer:
    def __init__(self, model_path='checkpoints/best_model.pth',
                 device=None, use_tta=True, confidence_threshold=0.9):
```

**Khởi tạo**:
1. Auto-detect device (CUDA nếu khả dụng, fallback CPU)
2. Load model state_dict từ checkpoint `.pth`
3. Đặt model sang `eval()` mode
4. Khởi tạo FaceDetector (MTCNN + Haar)
5. Khởi tạo ImagePreprocessor (CLAHE pipeline)
6. Định nghĩa eval_transform (Resize + Normalize)

**Phương thức `analyze_image`** — entry point chính (dòng 120–200):
```python
def analyze_image(self, image, mode='static'):
    """
    Phân tích ảnh: detect → crop → preprocess → predict
    Args:
        image: PIL.Image hoặc numpy array
        mode: 'static' (TTA + MTCNN) hoặc 'live' (no TTA + Haar)
    Returns:
        List[dict]: [{age, gender, race, confidence, bbox}, ...]
    """
    # Convert input
    image_pil = ensure_pil(image)
    image_np = np.array(image_pil)
    
    # Detect faces
    if mode == 'static':
        faces = self.detector.detect_faces(image_pil)
    else:
        faces = self.detector.fast_detect_haar(image_np)
    
    # Process each face
    results = []
    for face_info in faces:
        bbox = extract_bbox(face_info)
        face_crop = crop_face(image_np, bbox, padding=20)
        
        # Preprocess
        face_processed = self.preprocessor.full_pipeline(face_crop)
        face_tensor = self.eval_transform(face_processed).unsqueeze(0)
        
        # Predict (with or without TTA)
        if self.use_tta and mode == 'static':
            age, gender, race, conf = self.predict_with_tta(face_tensor)
        else:
            age, gender, race, conf = self.predict_single(face_tensor)
        
        results.append({
            'age': int(age),
            'gender': GENDER_LABELS[gender],
            'race': RACE_LABELS[race],
            'confidence': float(conf),
            'bbox': bbox,
        })
    return results
```

### 4.6.2 Test-Time Augmentation (TTA)

**Phương thức `predict_with_tta`** (dòng 250–320):
```python
def predict_with_tta(self, face_tensor):
    """5 augmented predictions, averaged."""
    augmentations = [
        lambda x: x,                                    # Original
        lambda x: torch.flip(x, dims=[3]),               # Horizontal flip
        lambda x: x * 1.1,                               # Brightness +10%
        lambda x: x * 0.9,                               # Brightness -10%
        lambda x: transforms.functional.rotate(x, 5),    # Rotation 5°
    ]
    
    all_ages, all_gender_logits, all_race_logits = [], [], []
    
    with torch.no_grad():
        for aug_fn in augmentations:
            aug_input = aug_fn(face_tensor).to(self.device)
            age, gender, race = self.model(aug_input)
            all_ages.append(age.cpu())
            all_gender_logits.append(gender.cpu())
            all_race_logits.append(race.cpu())
    
    # Average predictions
    avg_age = torch.stack(all_ages).mean(dim=0)
    avg_gender = torch.stack(all_gender_logits).mean(dim=0)
    avg_race = torch.stack(all_race_logits).mean(dim=0)
    
    age_val = avg_age.item()
    gender_idx = avg_gender.argmax(dim=1).item()
    race_idx = avg_race.argmax(dim=1).item()
    confidence = torch.softmax(avg_gender, dim=1).max().item()
    
    return age_val, gender_idx, race_idx, confidence
```

### 4.6.3 Batch Prediction

**Phương thức `analyze_batch`** (dòng 350–420):
```python
def analyze_batch(self, images, batch_size=8):
    """Xử lý hàng loạt ảnh."""
    all_results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        for img in batch:
            result = self.analyze_image(img, mode='static')
            all_results.extend(result)
    return all_results
```

---

## 4.7 Module Utils — Hằng số và cấu hình

**File**: `src/utils/constants.py` (81 dòng)

```python
# Kích thước ảnh
IMG_SIZE = 224

# Nhãn
GENDER_LABELS = {0: 'Male', 1: 'Female'}
RACE_LABELS = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Others'}
NUM_GENDER_CLASSES = 2
NUM_RACE_CLASSES = 4

# Race remap (5 → 4 lớp)
RACE_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}

# Nhóm tuổi (cho WeightedSampler + eval)
AGE_GROUPS = {0: '0-12 (Child)', 1: '13-19 (Teen)', 2: '20-35 (Young Adult)',
              3: '36-55 (Adult)', 4: '56+ (Senior)'}
NUM_AGE_GROUPS = 5

def age_to_group(age):
    if age <= 12: return 0
    elif age <= 19: return 1
    elif age <= 35: return 2
    elif age <= 55: return 3
    else: return 4

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training defaults
DEFAULT_CONFIG = {
    'csv_path': 'data/labels.csv',
    'img_dir': 'data/UTKFace',
    'save_path': 'checkpoints/best_model.pth',
    'metrics_path': 'logs/metrics.json',
    'batch_size': 32,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'patience': 10,
}
```

---

## 4.8 Script huấn luyện (train_v3.py)

**File**: `scripts/train_v3.py` (639 dòng)

### Tổng quan 9 kỹ thuật tích hợp

| # | Kỹ thuật | Cài đặt |
|---|---------|---------|
| 1 | Freeze backbone → Unfreeze | Epoch 1-5 freeze → 6+ unfreeze |
| 2 | Differential LR | backbone=1e-5, heads=1e-4 |
| 3 | Gradient Accumulation | 4 steps → effective batch 128 |
| 4 | Mixup | α=0.2, 70% batches |
| 5 | Wing Loss + Focal Loss | Age regression + Race classification |
| 6 | Adaptive Uncertainty Weighting | Learnable log_var per task |
| 7 | Progressive Resizing | 160→192→224 pixels |
| 8 | Multi-factor WeightedSampler | Race × Age group balancing |
| 9 | Early Stopping | patience=10 epochs |

### Luồng chính (`main()`)

1. **Parse CLI args**: `--resume`, `--extra-epochs`, `--no-mixup`, `--no-progressive`
2. **Init model + criterion**: `FaceAttributeModel(pretrained=True)`, `MultiTaskLoss(adaptive=True)`
3. **Phase 1**: `freeze_backbone(model)` → AdamW optimizer heads only → LR=1e-3
4. **Phase 2**: `unfreeze_backbone(model)` → `get_differential_lr_params()` → LR: backbone=1e-5, heads=1e-4
5. **Training loop** (epoch 1→50):
   - Progressive resize check → reload DataLoader nếu thay đổi
   - `train_one_epoch()` → AMP + Mixup + Gradient Accumulation
   - `validate()` → đánh giá trên val set
   - Save best model nếu val_loss cải thiện
   - Early Stopping nếu no improvement ≥ 10 epochs
6. **Test evaluation**: Reload best model @ 224px → `evaluate_test_set()` → Confusion Matrix + Classification Report
7. **Export**: Save `logs/metrics_v3.json`

### Hàm `train_one_epoch` (dòng 83–183) — Chi tiết

```python
def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    use_mixup=True, mixup_alpha=0.2, accumulation_steps=4):
```

**Luồng xử lý mỗi batch**:
1. Move data to device (GPU)
2. AMP context: `autocast(device_type='cuda')`
3. 70% chance Mixup: trộn ảnh + nhãn → mixed forward → `mixup_criterion`
4. 30% chance: standard forward → `criterion()`
5. NaN detection: nếu loss = NaN → skip batch
6. Scale loss / accumulation_steps → backward
7. Mỗi 4 batches: `scaler.unscale_()` → gradient clipping (max_norm=1.0) → `scaler.step()` → `optimizer.zero_grad()`

---

## 4.9 Script đánh giá (eval.py)

**File**: `scripts/eval.py` (206 dòng)

**Luồng chính**:
1. Load test data
2. Load best model từ checkpoint
3. Evaluate trên toàn bộ test set
4. Report chi tiết:
   - **Age**: MAE, Median, Std, % sai ≤5 năm, % sai ≤10 năm, MAE theo nhóm tuổi
   - **Gender**: Accuracy, Confusion Matrix, Classification Report (P/R/F1)
   - **Race**: Accuracy, Confusion Matrix, Classification Report (P/R/F1)
5. Tính điểm tổng hợp: `score = (100 - age_mae * 2) * 0.3 + gender_acc * 0.35 + race_acc * 0.35`
6. Xếp hạng: A (≥90), B (≥80), C (≥70), D (≥60), F (<60)

---

## 4.10 Script tổ chức dữ liệu (organize_data.py)

**File**: `scripts/organize_data.py` (211 dòng)

**Chức năng**: Chuyển đổi cấu trúc dữ liệu từ flat (data/UTKFace/) sang organized:
```
data/raw/UTKFace/                ← ảnh gốc
data/processed/train/White/      ← chia theo split + class
data/processed/train/Black/
data/processed/val/Asian/
data/processed/test/Others/
data/train.csv, val.csv, test.csv  ← CSV riêng cho mỗi split
```

Sử dụng **symbolic links** (symlinks) để không tốn thêm dung lượng. Nếu symlink thất bại (Windows no admin) → fallback sang file copy.

---

## 4.11 Script xuất biểu đồ (export_charts.py)

**File**: `scripts/export_charts.py` (639 dòng)

Xuất 13 biểu đồ PNG (2x resolution, 300 DPI) bằng Plotly:

| # | Hàm | File output | Mô tả |
|---|-----|-------------|-------|
| 1 | `chart_training_curves()` | `01_training_curves.png` | 4-in-1 subplot (Loss, MAE, Gender, Race) |
| 2 | `chart_loss_detail()` | `02_loss_curve.png` | Loss chi tiết với best epoch marker |
| 3 | `chart_age_mae_detail()` | `03_age_mae_curve.png` | Age MAE với target line ≤4.2 |
| 4 | `chart_accuracy_curves()` | `04_accuracy_curves.png` | Gender + Race accuracy 2-panel |
| 5 | `chart_confusion_matrix()` | `05_gender_confusion_matrix.png` | Heatmap giới tính |
| 6 | `chart_confusion_matrix()` | `06_race_confusion_matrix.png` | Heatmap sắc tộc |
| 7 | `chart_classification_report()` | `07_gender_precision_recall_f1.png` | P/R/F1 grouped bar |
| 8 | `chart_classification_report()` | `08_race_precision_recall_f1.png` | P/R/F1 grouped bar |
| 9 | `chart_age_error_by_group()` | `09_age_error_by_group.png` | MAE theo 5 nhóm tuổi |
| 10 | `chart_radar()` | `10_radar_performance.png` | Radar 5 chỉ số |
| 11 | `chart_accuracy_comparison()` | `11_accuracy_overview.png` | Tổng quan + target 90% |
| 12 | `chart_dataset_distribution()` | `12_dataset_distribution.png` | 3 donut charts |
| 13 | `chart_score_card()` | `13_score_card.png` | Điểm tổng hợp + grade |

---

## 4.12 Ứng dụng web (app.py)

**File**: `app.py` (1.397 dòng)

### 4.12.1 Cấu hình Streamlit

```python
st.set_page_config(
    page_title="FaceVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

### 4.12.2 Thiết kế CSS

Custom CSS injection (~200 dòng) bao gồm:
- **Color system**: primary `#4F46E5`, background `#FFFFFF`, text `#1E293B`
- **Metric cards**: Gradient border, rounded corners, shadow
- **Buttons**: Hover transitions, focus states
- **Responsive columns**: 2-column layout cho kết quả dự đoán
- **Tab styling**: Active tab indicator

### 4.12.3 Tab Ảnh tĩnh (Tab 1)

Luồng xử lý:
1. Input selector: `st.file_uploader()` / `st.camera_input()` / Clipboard paste
2. Display ảnh gốc (col 1) + ảnh kết quả (col 2)
3. Call `analyzer.analyze_image(image, mode='static')` (TTA enabled)
4. Draw bounding box + label overlay bằng PIL.ImageDraw
5. Display metric cards (Age, Gender, Race) với confidence %

### 4.12.4 Tab Camera trực tiếp (Tab 2)

Luồng xử lý:
1. OpenCV `cv2.VideoCapture(0)` → capture frames
2. Detect: `detector.fast_detect_haar()` (ưu tiên tốc độ)
3. Preprocess + Model predict (no TTA)
4. Draw annotation overlay + FPS counter
5. `st.image()` update frame liên tục
6. Start/Stop toggle button

### 4.12.5 Tab Biểu đồ đánh giá (Tab 3)

Luồng xử lý:
1. Load `logs/metrics.json` → parse `history` + `test_metrics`
2. Render 13 Plotly charts inline bằng `st.plotly_chart(fig, use_container_width=True)`
3. Hỗ trợ interactive: zoom, pan, hover tooltip
4. Display summary metrics cards ở đầu trang

---

## 4.13 Notebook huấn luyện trên Google Colab

**File**: `notebooks/train_colab.ipynb`

**Mục đích**: Chạy training trên GPU miễn phí (Tesla T4/A100) khi máy local không đủ mạnh.

**Các cell chính**:
1. Mount Google Drive + clone repo
2. Install dependencies (`pip install -r requirements.txt`)
3. Upload/link dataset UTKFace
4. Run `scripts/train_v3.py` với output logging
5. Download checkpoint về local
6. Run `scripts/eval.py` để đánh giá

**Lưu ý**: Notebook sử dụng `!python scripts/train_v3.py` (shell command) thay vì import trực tiếp, đảm bảo environment consistency.
