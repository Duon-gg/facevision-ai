"""
losses/multi_task_loss.py - Hàm loss đa nhiệm cải tiến v5
Loss = σ1 * WingLoss(Age) + σ2 * CE(Gender) + σ3 * FocalLoss(Race)
Adaptive Uncertainty Weighting: tự động learn task weights
Wing Loss: thiết kế cho face regression tasks (with numerical stability)
FocalLoss: hỗ trợ class weights cho class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """
    Focal Loss — tập trung vào hard examples, giúp class thiểu số.
    Hỗ trợ class weights để giảm bias cho majority class.
    
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, label_smoothing=0.1, class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        # class_weights: tensor [num_classes] hoặc None
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


class WingLoss(nn.Module):
    """
    Wing Loss — thiết kế cho face landmark/age regression.
    Numerically stabilized version.
    
    Wing(x) = w * ln(1 + |x|/ε)  if |x| < w
            = |x| - C             otherwise
    where C = w - w * ln(1 + w/ε)
    """
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)
    
    def forward(self, pred, target):
        x = pred - target
        abs_x = torch.abs(x)
        # Clamp to prevent overflow
        abs_x = torch.clamp(abs_x, max=100.0)
        
        log_term = self.w * torch.log1p(abs_x / self.epsilon)
        linear_term = abs_x - self.C
        
        loss = torch.where(abs_x < self.w, log_term, linear_term)
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss cho multi-task learning (v3 - cải tiến).
    
    Adaptive Uncertainty Weighting (Kendall et al.):
    L_total = (1/2σ1²) * L_age + (1/2σ2²) * L_gender + (1/2σ3²) * L_race
            + log(σ1) + log(σ2) + log(σ3)
    """

    def __init__(self, w_age=1.0, w_gender=1.0, w_race=1.0,
                 age_mode='regression', adaptive=True,
                 race_class_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.w_age = w_age
        self.w_gender = w_gender
        self.w_race = w_race
        self.age_mode = age_mode
        self.adaptive = adaptive

        if age_mode == 'regression':
            self.age_loss_fn = WingLoss(w=10.0, epsilon=2.0)
        else:
            self.age_loss_fn = nn.CrossEntropyLoss()

        self.gender_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.race_loss_fn = FocalLoss(
            gamma=2.0, label_smoothing=0.1,
            class_weights=race_class_weights
        )

        # Learnable log(σ²) cho uncertainty weighting
        if adaptive:
            self.log_var_age = nn.Parameter(torch.zeros(1))
            self.log_var_gender = nn.Parameter(torch.zeros(1))
            self.log_var_race = nn.Parameter(torch.zeros(1))

    def forward(self, age_pred, gender_pred, race_pred,
                age_true, gender_true, race_true):
        """Tính loss tổng hợp."""
        if self.age_mode == 'regression':
            l_age = self.age_loss_fn(age_pred.squeeze(), age_true.float())
        else:
            l_age = self.age_loss_fn(age_pred, age_true.long())

        l_gender = self.gender_loss_fn(gender_pred, gender_true)
        l_race = self.race_loss_fn(race_pred, race_true)

        # Clamp individual losses to prevent NaN propagation
        l_age = torch.clamp(l_age, max=100.0)
        l_gender = torch.clamp(l_gender, max=50.0)
        l_race = torch.clamp(l_race, max=50.0)

        if self.adaptive:
            # Clamp log_var to prevent extreme precision values
            log_var_age = torch.clamp(self.log_var_age, -4.0, 4.0)
            log_var_gender = torch.clamp(self.log_var_gender, -4.0, 4.0)
            log_var_race = torch.clamp(self.log_var_race, -4.0, 4.0)

            precision_age = torch.exp(-log_var_age)
            precision_gender = torch.exp(-log_var_gender)
            precision_race = torch.exp(-log_var_race)

            total = (0.5 * precision_age * l_age + 0.5 * log_var_age +
                     0.5 * precision_gender * l_gender + 0.5 * log_var_gender +
                     0.5 * precision_race * l_race + 0.5 * log_var_race)
        else:
            total = self.w_age * l_age + self.w_gender * l_gender + self.w_race * l_race

        return total, l_age, l_gender, l_race
