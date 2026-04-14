"""
utils/constants.py - Hằng số và cấu hình chung cho hệ thống
"""

# ============================================================
# Image Configuration
# ============================================================
IMG_SIZE = 224

# ImageNet normalization (pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# Label Mappings
# ============================================================
GENDER_LABELS = {0: 'Male', 1: 'Female'}
GENDER_LABELS_VI = {0: 'Nam', 1: 'Nữ'}

# 4-class race (merged from 5-class UTKFace)
RACE_LABELS = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Others'}

# Remap original UTKFace 5-class → 4-class
# Original: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
# New:      0=White, 1=Black, 2=Asian, 3=Others (Indian merged into Others)
RACE_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}

NUM_GENDER_CLASSES = 2
NUM_RACE_CLASSES = 4

# ============================================================
# Age Group Configuration (optional age-group mode)
# ============================================================
AGE_GROUPS = {
    0: '0-12 (Child)',
    1: '13-19 (Teen)',
    2: '20-35 (Young Adult)',
    3: '36-55 (Adult)',
    4: '56+ (Senior)',
}

def age_to_group(age):
    """Convert continuous age to age group index."""
    if age <= 12:
        return 0
    elif age <= 19:
        return 1
    elif age <= 35:
        return 2
    elif age <= 55:
        return 3
    else:
        return 4

NUM_AGE_GROUPS = len(AGE_GROUPS)

# ============================================================
# Training Defaults
# ============================================================
DEFAULT_CONFIG = {
    'csv_path': 'data/labels.csv',
    'img_dir': 'data/raw/UTKFace',
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-4,
    'save_path': 'checkpoints/best_model.pth',
    'metrics_path': 'logs/metrics.json',
    'num_workers': 0,  # Windows compatibility
    'w_age': 1.0,
    'w_gender': 1.0,
    'w_race': 1.0,
}

# ============================================================
# Bounding Box Colors (BGR for OpenCV)
# ============================================================
BBOX_COLORS = {
    'Male': (0, 200, 0),       # Green
    'Female': (255, 105, 180),  # Pink
}
