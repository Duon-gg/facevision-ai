"""
data/loader.py - Data loading, splitting, và transforms (v3)
Chia dữ liệu: Train 70% / Validation 15% / Test 15%
WeightedRandomSampler cho class imbalance (Race + Age group)
Hỗ trợ Progressive Resizing
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import UTKFaceDataset
from src.utils.constants import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, RACE_REMAP, age_to_group


# ============================================================
# Transforms
# ============================================================

def get_transforms(train=True, img_size=IMG_SIZE):
    """
    Lấy torchvision transforms.
    
    Train: RandomFlip + RandomRotation + ColorJitter + Normalize + ...
    Eval: chỉ Resize + Normalize
    
    Args:
        train: training mode
        img_size: kích thước ảnh (hỗ trợ progressive resizing)
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


# ============================================================
# Data Loading
# ============================================================

def _compute_sample_weights(df):
    """
    Tính sample weights cho WeightedRandomSampler.
    Multi-factor: cân bằng theo CẢ Race + Age group.
    """
    # Remap race
    race_remapped = df['race'].map(lambda x: RACE_REMAP.get(x, 3))
    
    # Age group
    age_groups = df['age'].map(age_to_group)
    
    # Race weights
    race_counts = race_remapped.value_counts().to_dict()
    total = len(race_remapped)
    num_race_classes = len(race_counts)
    race_weights = {
        cls: total / (num_race_classes * count)
        for cls, count in race_counts.items()
    }
    
    # Age group weights
    age_counts = age_groups.value_counts().to_dict()
    num_age_groups = len(age_counts)
    age_weights = {
        grp: total / (num_age_groups * count)
        for grp, count in age_counts.items()
    }
    
    # Combined weight = race_weight * age_weight (multi-factor)
    race_w = race_remapped.map(race_weights).values.astype(np.float64)
    age_w = age_groups.map(age_weights).values.astype(np.float64)
    
    # Geometric mean để tránh extreme weights
    sample_weights = np.sqrt(race_w * age_w)
    
    return sample_weights


def load_data(csv_path, img_dir, batch_size=32, num_workers=0,
              apply_blur=False, age_mode='regression', img_size=IMG_SIZE):
    """
    Load dataset và chia: Train 70% / Val 15% / Test 15%.
    Sử dụng WeightedRandomSampler (Race + Age) cho train set.
    
    Args:
        csv_path: đường dẫn file CSV labels
        img_dir: thư mục chứa ảnh UTKFace
        batch_size: kích thước batch
        num_workers: số worker cho DataLoader
        apply_blur: áp dụng Gaussian blur
        age_mode: 'regression' hoặc 'age_group'
        img_size: kích thước ảnh (cho progressive resizing)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    df = pd.read_csv(csv_path)

    # Lọc: chỉ giữ ảnh tồn tại
    df['exists'] = df['image'].apply(
        lambda x: os.path.exists(os.path.join(img_dir, x))
    )
    df = df[df['exists']].drop(columns=['exists']).reset_index(drop=True)
    print(f"[Dataset] Total valid images: {len(df)}")

    # Chia 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['gender']
    )
    # Chia temp: 50% val (= 15% tổng), 50% test (= 15% tổng)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['gender']
    )

    print(f"[Dataset] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Tạo datasets
    train_ds = UTKFaceDataset(
        train_df, img_dir, transform=get_transforms(train=True, img_size=img_size),
        apply_blur=apply_blur, age_mode=age_mode
    )
    val_ds = UTKFaceDataset(
        val_df, img_dir, transform=get_transforms(train=False, img_size=img_size),
        apply_blur=apply_blur, age_mode=age_mode
    )
    test_ds = UTKFaceDataset(
        test_df, img_dir, transform=get_transforms(train=False, img_size=img_size),
        apply_blur=apply_blur, age_mode=age_mode
    )

    # WeightedRandomSampler (cân bằng race + age group)
    sample_weights = _compute_sample_weights(train_df)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
