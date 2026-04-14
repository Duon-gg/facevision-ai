"""
data/dataset.py - PyTorch Dataset cho UTKFace (v3)
Hỗ trợ race remapping (5→4 classes), pipeline xử lí ảnh số, và Mixup
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocessing.image_processing import preprocess_image
from src.utils.constants import RACE_REMAP, IMG_SIZE, age_to_group


class UTKFaceDataset(Dataset):
    """
    UTKFace Dataset cho multi-task learning.
    
    Labels:
        - age: int (0-116)
        - gender: 0=Male, 1=Female
        - race: 0=White, 1=Black, 2=Asian, 3=Others (remapped từ 5 lớp)
    
    Args:
        dataframe: pandas DataFrame với cột image, age, gender, race
        img_dir: thư mục chứa ảnh
        transform: torchvision transforms
        apply_blur: áp dụng Gaussian blur trong preprocessing
        age_mode: 'regression' hoặc 'age_group'
    """

    def __init__(self, dataframe, img_dir, transform=None,
                 apply_blur=False, age_mode='regression'):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.apply_blur = apply_blur
        self.age_mode = age_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback: blank image
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pipeline xử lí ảnh số
        image = preprocess_image(image, apply_blur=self.apply_blur,
                                 resize=True, target_size=IMG_SIZE)

        # Convert to PIL for torchvision transforms
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # Labels
        age_val = int(row['age'])
        if self.age_mode == 'regression':
            age = torch.tensor(age_val, dtype=torch.float32)
        else:
            age = torch.tensor(age_to_group(age_val), dtype=torch.long)

        gender = torch.tensor(int(row['gender']), dtype=torch.long)

        # Race remapping: 5-class → 4-class
        race_original = int(row['race'])
        race_remapped = RACE_REMAP.get(race_original, 3)  # default → Others
        race = torch.tensor(race_remapped, dtype=torch.long)

        return image, age, gender, race


def mixup_data(images, ages, genders, races, alpha=0.2):
    """
    Mixup augmentation: trộn 2 ảnh + labels.
    
    x_mixed = λ * x_i + (1-λ) * x_j
    Với age (regression): age_mixed = λ * age_i + (1-λ) * age_j
    Với gender/race (classification): trả về cả 2 labels + λ
    
    Args:
        images: batch ảnh [B, C, H, W]
        ages: batch tuổi [B]
        genders: batch giới tính [B]
        races: batch sắc tộc [B]
        alpha: Beta distribution parameter
    
    Returns:
        mixed_images, ages_a, ages_b, genders_a, genders_b, 
        races_a, races_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Đảm bảo lam >= 0.5
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    
    return (mixed_images,
            ages, ages[index],
            genders, genders[index],
            races, races[index],
            lam)


def mixup_criterion(criterion, pred_age, pred_gender, pred_race,
                    ages_a, ages_b, genders_a, genders_b,
                    races_a, races_b, lam):
    """
    Tính loss với Mixup labels.
    
    L = λ * L(pred, target_a) + (1-λ) * L(pred, target_b)
    """
    loss_a, l_age_a, l_gender_a, l_race_a = criterion(
        pred_age, pred_gender, pred_race, ages_a, genders_a, races_a
    )
    loss_b, l_age_b, l_gender_b, l_race_b = criterion(
        pred_age, pred_gender, pred_race, ages_b, genders_b, races_b
    )
    
    total = lam * loss_a + (1 - lam) * loss_b
    l_age = lam * l_age_a + (1 - lam) * l_age_b
    l_gender = lam * l_gender_a + (1 - lam) * l_gender_b
    l_race = lam * l_race_a + (1 - lam) * l_race_b
    
    return total, l_age, l_gender, l_race


def mixup_data_minority(images, ages, genders, races, alpha=0.4,
                        minority_classes=(1, 2)):
    """
    Minority-Only Mixup: chỉ trộn ảnh thuộc minority classes.
    Giữ nguyên majority classes (White, Others) → tránh tăng bias.
    
    Args:
        minority_classes: tuple chứa race class indices cần mixup
                         (1=Black, 2=Asian)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1.0

    batch_size = images.size(0)
    
    # Mask: True cho minority samples
    minority_mask = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
    for cls in minority_classes:
        minority_mask |= (races == cls)
    
    minority_count = minority_mask.sum().item()
    
    if minority_count < 2:
        # Không đủ minority samples → dùng mixup thường
        return mixup_data(images, ages, genders, races, alpha)
    
    # Chỉ shuffle index trong minority group
    minority_indices = torch.where(minority_mask)[0]
    perm = minority_indices[torch.randperm(len(minority_indices), device=images.device)]
    
    # Tạo mixed images: chỉ mixup minority, giữ nguyên majority
    mixed_images = images.clone()
    mixed_images[minority_indices] = (
        lam * images[minority_indices] + (1 - lam) * images[perm]
    )
    
    # Index map cho labels: majority giữ nguyên (map tới chính nó)
    index = torch.arange(batch_size, device=images.device)
    index[minority_indices] = perm
    
    return (mixed_images,
            ages, ages[index],
            genders, genders[index],
            races, races[index],
            lam)

