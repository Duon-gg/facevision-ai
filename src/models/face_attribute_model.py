"""
models/face_attribute_model.py - Multi-Task CNN cho dự đoán thuộc tính khuôn mặt
Backbone: ResNet50 (pretrained ImageNet) + SE Attention
Heads: Age (regression), Gender (2 classes), Race (4 classes)
"""

import torch
import torch.nn as nn
from torchvision import models

from src.utils.constants import NUM_GENDER_CLASSES, NUM_RACE_CLASSES, NUM_AGE_GROUPS


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Học channel-wise attention weights để focus vào features quan trọng.
    """
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
        # x: [B, C]
        scale = self.excitation(x)  # [B, C]
        return x * scale


class FaceAttributeModel(nn.Module):
    """
    Multi-task learning model cho face attribute prediction.
    
    Kiến trúc:
    - Shared backbone: ResNet50 (pretrained) → 2048-dim feature vector
    - SE Attention: channel-wise attention (2048 → 2048)
    - Age head: FC(2048→512→128→1) regression
    - Gender head: FC(2048→256→2) classification
    - Race head: FC(2048→256→4) classification
    
    Args:
        pretrained: sử dụng weights pretrained ImageNet
        age_mode: 'regression' hoặc 'age_group'
    """

    def __init__(self, pretrained=True, age_mode='regression'):
        super(FaceAttributeModel, self).__init__()
        self.age_mode = age_mode

        # --- Shared Backbone: ResNet50 ---
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        num_features = self.backbone.fc.in_features  # 2048

        # Bỏ FC layer gốc, thay bằng Identity
        self.backbone.fc = nn.Identity()

        # --- SE Attention ---
        self.se = SEBlock(num_features, reduction=16)

        # --- Age Head ---
        if age_mode == 'regression':
            age_output_size = 1
        else:
            age_output_size = NUM_AGE_GROUPS

        self.age_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, age_output_size)
        )

        # --- Gender Head (2 lớp: Male/Female) ---
        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_GENDER_CLASSES)
        )

        # --- Race Head (4 lớp: White/Black/Asian/Others) ---
        self.race_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_RACE_CLASSES)
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: batch ảnh [B, 3, 224, 224]
        
        Returns:
            age: [B, 1] (regression) hoặc [B, NUM_AGE_GROUPS] (classification)
            gender: [B, 2] logits
            race: [B, 4] logits
        """
        features = self.backbone(x)       # [B, 2048]
        features = self.se(features)       # [B, 2048] (attention-weighted)
        age = self.age_head(features)
        gender = self.gender_head(features)
        race = self.race_head(features)
        return age, gender, race


if __name__ == "__main__":
    # Quick test
    model = FaceAttributeModel(pretrained=False, age_mode='regression')
    x = torch.randn(4, 3, 224, 224)
    age, gender, race = model(x)
    print(f"Age shape: {age.shape}")      # [4, 1]
    print(f"Gender shape: {gender.shape}")  # [4, 2]
    print(f"Race shape: {race.shape}")      # [4, 4]

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
    
    # Test age_group mode
    model2 = FaceAttributeModel(pretrained=False, age_mode='age_group')
    age2, _, _ = model2(x)
    print(f"Age group shape: {age2.shape}")  # [4, 5]
