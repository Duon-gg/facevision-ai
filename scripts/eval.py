"""
run_eval.py - Tự test & đánh giá model trên test set
Tải best_model.pth → chạy trên test set → in kết quả chi tiết
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
import torch
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, classification_report

from src.models.face_attribute_model import FaceAttributeModel
from src.data.loader import load_data
from src.utils.constants import (
    DEFAULT_CONFIG, GENDER_LABELS, RACE_LABELS,
    NUM_GENDER_CLASSES, NUM_RACE_CLASSES, AGE_GROUPS, age_to_group
)


@torch.no_grad()
def evaluate(model, loader, device):
    """Đánh giá chi tiết trên test set."""
    model.eval()
    all_gender_pred, all_gender_true = [], []
    all_race_pred, all_race_true = [], []
    all_age_pred, all_age_true = [], []

    for images, ages, genders, races in loader:
        images = images.to(device)
        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            age_pred, gender_pred, race_pred = model(images)

        # Age
        ages_np = ages.numpy()
        age_preds_np = age_pred.squeeze().cpu().numpy()
        if age_preds_np.ndim == 0:
            age_preds_np = np.array([age_preds_np])
        all_age_true.extend(ages_np)
        all_age_pred.extend(age_preds_np)

        # Gender
        g_pred = gender_pred.argmax(1).cpu().numpy()
        all_gender_pred.extend(g_pred)
        all_gender_true.extend(genders.numpy())

        # Race
        r_pred = race_pred.argmax(1).cpu().numpy()
        all_race_pred.extend(r_pred)
        all_race_true.extend(races.numpy())

    all_age_true = np.array(all_age_true)
    all_age_pred = np.array(all_age_pred)
    all_gender_true = np.array(all_gender_true)
    all_gender_pred = np.array(all_gender_pred)
    all_race_true = np.array(all_race_true)
    all_race_pred = np.array(all_race_pred)

    return (all_age_true, all_age_pred,
            all_gender_true, all_gender_pred,
            all_race_true, all_race_pred)


def main():
    cfg = DEFAULT_CONFIG.copy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 65)
    print("  TỰ ĐÁNH GIÁ MODEL - FACE ATTRIBUTE PREDICTION")
    print("=" * 65)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Check model file
    if not os.path.exists(cfg['save_path']):
        print(f"\n❌ Không tìm thấy model: {cfg['save_path']}")
        return

    # Load data
    print("\n[1/3] Đang tải dữ liệu...")
    _, _, test_loader = load_data(
        cfg['csv_path'], cfg['img_dir'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers']
    )

    # Load model
    print("[2/3] Đang tải model...")
    model = FaceAttributeModel(pretrained=False, age_mode='regression').to(device)
    checkpoint = torch.load(cfg['save_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    version = checkpoint.get('model_version', 'unknown')
    best_epoch = checkpoint.get('epoch', '?')
    print(f"  Model version: {version}")
    print(f"  Best epoch: {best_epoch}")

    # Evaluate
    print("[3/3] Đang đánh giá trên test set...\n")
    (age_true, age_pred,
     gender_true, gender_pred,
     race_true, race_pred) = evaluate(model, test_loader, device)

    n = len(age_true)

    # ── AGE ──
    age_errors = np.abs(age_pred - age_true)
    age_mae = np.mean(age_errors)
    age_median = np.median(age_errors)
    age_std = np.std(age_errors)
    age_within_5 = np.mean(age_errors <= 5) * 100
    age_within_10 = np.mean(age_errors <= 10) * 100

    print("=" * 65)
    print("  📊 KẾT QUẢ ĐÁNH GIÁ TUỔI (Age Regression)")
    print("-" * 65)
    print(f"  MAE (Mean Absolute Error):   {age_mae:.2f} năm")
    print(f"  Median Error:                {age_median:.2f} năm")
    print(f"  Std Error:                   {age_std:.2f} năm")
    print(f"  % sai ≤ 5 năm:              {age_within_5:.1f}%")
    print(f"  % sai ≤ 10 năm:             {age_within_10:.1f}%")

    # Age error by age group
    print("\n  Sai số theo nhóm tuổi:")
    print(f"  {'Nhóm tuổi':<22} {'MAE':>6} {'Median':>8} {'Số lượng':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*10}")
    for gid, gname in AGE_GROUPS.items():
        mask = np.array([age_to_group(a) == gid for a in age_true])
        if mask.sum() > 0:
            grp_mae = np.mean(age_errors[mask])
            grp_med = np.median(age_errors[mask])
            print(f"  {gname:<22} {grp_mae:>6.2f} {grp_med:>8.2f} {mask.sum():>10d}")

    # ── GENDER ──
    gender_acc = np.mean(gender_pred == gender_true) * 100
    gender_names = [GENDER_LABELS[i] for i in range(NUM_GENDER_CLASSES)]

    print("\n" + "=" * 65)
    print("  👤 KẾT QUẢ ĐÁNH GIÁ GIỚI TÍNH (Gender Classification)")
    print("-" * 65)
    print(f"  Accuracy: {gender_acc:.2f}%\n")

    print("  Confusion Matrix:")
    gender_cm = confusion_matrix(gender_true, gender_pred, labels=list(range(NUM_GENDER_CLASSES)))
    print(f"  {'':>10} | " + " | ".join(f"{n:>8}" for n in gender_names))
    print(f"  {'-'*10}-+-" + "-+-".join("-"*8 for _ in gender_names))
    for i, row in enumerate(gender_cm):
        print(f"  {gender_names[i]:>10} | " + " | ".join(f"{v:>8d}" for v in row))

    print("\n  Classification Report:")
    print(classification_report(gender_true, gender_pred, target_names=gender_names, digits=4))

    # ── RACE ──
    race_acc = np.mean(race_pred == race_true) * 100
    race_names = [RACE_LABELS[i] for i in range(NUM_RACE_CLASSES)]

    print("=" * 65)
    print("  🌍 KẾT QUẢ ĐÁNH GIÁ SẮC TỘC (Race Classification)")
    print("-" * 65)
    print(f"  Accuracy: {race_acc:.2f}%\n")

    print("  Confusion Matrix:")
    race_cm = confusion_matrix(race_true, race_pred, labels=list(range(NUM_RACE_CLASSES)))
    print(f"  {'':>10} | " + " | ".join(f"{n:>8}" for n in race_names))
    print(f"  {'-'*10}-+-" + "-+-".join("-"*8 for _ in race_names))
    for i, row in enumerate(race_cm):
        print(f"  {race_names[i]:>10} | " + " | ".join(f"{v:>8d}" for v in row))

    print("\n  Classification Report:")
    print(classification_report(race_true, race_pred, target_names=race_names, digits=4))

    # ── TỔNG KẾT ──
    print("=" * 65)
    print("  🏆 TỔNG KẾT")
    print("=" * 65)
    print(f"  Tổng số mẫu test:      {n}")
    print(f"  Age MAE:                {age_mae:.2f} năm")
    print(f"  Gender Accuracy:        {gender_acc:.2f}%")
    print(f"  Race Accuracy:          {race_acc:.2f}%")
    print()

    # Grade
    overall_score = (100 - age_mae * 2) * 0.3 + gender_acc * 0.35 + race_acc * 0.35
    if overall_score >= 90:
        grade = "A (Xuất sắc)"
    elif overall_score >= 80:
        grade = "B (Tốt)"
    elif overall_score >= 70:
        grade = "C (Khá)"
    elif overall_score >= 60:
        grade = "D (Trung bình)"
    else:
        grade = "F (Cần cải thiện)"

    print(f"  Điểm tổng hợp:         {overall_score:.1f}/100")
    print(f"  Xếp hạng:              {grade}")
    print("=" * 65)


if __name__ == '__main__':
    main()
