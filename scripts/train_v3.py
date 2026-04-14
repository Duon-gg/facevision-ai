"""
DU_train_v3.py - Training script NÂNG CAO cho face attribute prediction
Tích hợp TẤT CẢ cải thiện:
  1. Freeze backbone → unfreeze với differential LR
  2. Warmup + OneCycleLR
  3. Gradient Accumulation (effective batch=128)
  4. Mixup augmentation (α=0.2)
  5. Wing Loss + Adaptive Uncertainty Weighting
  6. Early Stopping patience=10
  7. Progressive Resizing (160→192→224)
  8. Multi-factor weighted sampling (Race + Age group)
  9. Dropout 0.3 uniform
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import time
import math
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from sklearn.metrics import confusion_matrix, classification_report

# Modular imports
from src.models.face_attribute_model import FaceAttributeModel
from src.losses.multi_task_loss import MultiTaskLoss
from src.data.loader import load_data
from src.data.dataset import mixup_data, mixup_criterion
from src.utils.constants import (
    DEFAULT_CONFIG, GENDER_LABELS, RACE_LABELS,
    NUM_GENDER_CLASSES, NUM_RACE_CLASSES, NUM_AGE_GROUPS
)


# ============================================================
# Freeze / Unfreeze Backbone
# ============================================================

def freeze_backbone(model):
    """Đóng băng backbone ResNet50 — chỉ train heads."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    # SE block vẫn train
    for param in model.se.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Freeze] Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_backbone(model):
    """Mở khóa backbone — train toàn bộ model."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Unfreeze] Backbone unfrozen. Trainable params: {trainable:,}")


def get_differential_lr_params(model, backbone_lr, head_lr):
    """Differential Learning Rate: backbone LR nhỏ hơn heads 10x."""
    return [
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.se.parameters(), 'lr': head_lr},
        {'params': model.age_head.parameters(), 'lr': head_lr},
        {'params': model.gender_head.parameters(), 'lr': head_lr},
        {'params': model.race_head.parameters(), 'lr': head_lr},
    ]


# ============================================================
# Training & Validation
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    use_mixup=True, mixup_alpha=0.2,
                    accumulation_steps=4):
    """
    Huấn luyện 1 epoch với:
    - AMP (Mixed Precision)
    - Mixup augmentation
    - Gradient Accumulation
    - NaN detection & skip
    """
    model.train()
    total_loss = 0
    total_age_mae = 0
    gender_correct = 0
    race_correct = 0
    total_samples = 0
    nan_batches = 0
    optimizer.zero_grad()

    for step, (images, ages, genders, races) in enumerate(loader):
        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)
        races = races.to(device)

        # Mixed precision forward
        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            if use_mixup and np.random.random() > 0.3:  # 70% chance Mixup
                # Mixup augmentation
                (mixed_images, ages_a, ages_b,
                 genders_a, genders_b,
                 races_a, races_b, lam) = mixup_data(
                    images, ages, genders, races, alpha=mixup_alpha
                )
                age_pred, gender_pred, race_pred = model(mixed_images)
                loss, l_age, l_gender, l_race = mixup_criterion(
                    criterion, age_pred, gender_pred, race_pred,
                    ages_a, ages_b, genders_a, genders_b,
                    races_a, races_b, lam
                )
            else:
                # Standard forward
                age_pred, gender_pred, race_pred = model(images)
                loss, l_age, l_gender, l_race = criterion(
                    age_pred, gender_pred, race_pred, ages, genders, races
                )

            # NaN detection — skip bad batches
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Mixed precision backward
        scaler.scale(loss).backward()

        # Gradient accumulation: step every N batches
        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # Check for NaN gradients
            valid_grads = True
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    valid_grads = False
                    break
            if valid_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_size = images.size(0)
        loss_val = loss.item() * accumulation_steps
        age_val = l_age.item()
        if not (math.isnan(loss_val) or math.isnan(age_val)):
            total_loss += loss_val * batch_size
            total_age_mae += age_val * batch_size
        gender_correct += (gender_pred.argmax(1) == genders).sum().item()
        race_correct += (race_pred.argmax(1) == races).sum().item()
        total_samples += batch_size

    # Flush remaining gradients
    if total_samples > 0 and (step + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if nan_batches > 0:
        print(f"  [Warning] {nan_batches} NaN batches skipped")

    return {
        'loss': total_loss / max(total_samples, 1),
        'age_mae': total_age_mae / max(total_samples, 1),
        'gender_acc': gender_correct / max(total_samples, 1) * 100,
        'race_acc': race_correct / max(total_samples, 1) * 100,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Đánh giá trên validation/test set."""
    model.eval()
    total_loss = 0
    total_age_mae = 0
    gender_correct = 0
    race_correct = 0
    total_samples = 0

    for images, ages, genders, races in loader:
        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)
        races = races.to(device)

        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            age_pred, gender_pred, race_pred = model(images)
            loss, l_age, l_gender, l_race = criterion(
                age_pred, gender_pred, race_pred, ages, genders, races
            )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_age_mae += l_age.item() * batch_size
        gender_correct += (gender_pred.argmax(1) == genders).sum().item()
        race_correct += (race_pred.argmax(1) == races).sum().item()
        total_samples += batch_size

    return {
        'loss': total_loss / total_samples,
        'age_mae': total_age_mae / total_samples,
        'gender_acc': gender_correct / total_samples * 100,
        'race_acc': race_correct / total_samples * 100,
    }


@torch.no_grad()
def evaluate_test_set(model, loader, device):
    """Đánh giá chi tiết trên test set. Returns: metrics dict"""
    model.eval()
    all_gender_pred, all_gender_true = [], []
    all_race_pred, all_race_true = [], []
    all_age_errors = []

    for images, ages, genders, races in loader:
        images = images.to(device)
        with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            age_pred, gender_pred, race_pred = model(images)

        g_pred = gender_pred.argmax(1).cpu().numpy()
        all_gender_pred.extend(g_pred)
        all_gender_true.extend(genders.numpy())

        r_pred = race_pred.argmax(1).cpu().numpy()
        all_race_pred.extend(r_pred)
        all_race_true.extend(races.numpy())

        age_errors = torch.abs(age_pred.squeeze().cpu() - ages.float()).numpy()
        all_age_errors.extend(age_errors)

    gender_cm = confusion_matrix(
        all_gender_true, all_gender_pred,
        labels=list(range(NUM_GENDER_CLASSES))
    )
    race_cm = confusion_matrix(
        all_race_true, all_race_pred,
        labels=list(range(NUM_RACE_CLASSES))
    )

    gender_names = [GENDER_LABELS[i] for i in range(NUM_GENDER_CLASSES)]
    race_names = [RACE_LABELS[i] for i in range(NUM_RACE_CLASSES)]

    gender_report = classification_report(
        all_gender_true, all_gender_pred,
        target_names=gender_names, output_dict=True
    )
    race_report = classification_report(
        all_race_true, all_race_pred,
        target_names=race_names, output_dict=True
    )

    return {
        'age_mae': float(np.mean(all_age_errors)),
        'gender_accuracy': float(np.mean(np.array(all_gender_pred) == np.array(all_gender_true))) * 100,
        'race_accuracy': float(np.mean(np.array(all_race_pred) == np.array(all_race_true))) * 100,
        'gender_confusion_matrix': gender_cm.tolist(),
        'race_confusion_matrix': race_cm.tolist(),
        'gender_classification_report': gender_report,
        'race_classification_report': race_report,
    }


# ============================================================
# Progressive Resizing Schedule
# ============================================================

PROGRESSIVE_SCHEDULE = [
    # (start_epoch, img_size)
    (1, 160),     # Phase 1: small images, fast learning
    (10, 192),    # Phase 2: medium images
    (20, 224),    # Phase 3: full resolution
]


def get_img_size_for_epoch(epoch):
    """Lấy kích thước ảnh cho epoch hiện tại."""
    img_size = 160
    for start_ep, size in PROGRESSIVE_SCHEDULE:
        if epoch >= start_ep:
            img_size = size
    return img_size


# ============================================================
# Main Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train face attribute model v3")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from best_model_v3.pth")
    parser.add_argument('--extra-epochs', type=int, default=30,
                        help="Extra epochs when resuming (default: 30)")
    parser.add_argument('--no-mixup', action='store_true',
                        help="Disable Mixup augmentation")
    parser.add_argument('--no-progressive', action='store_true',
                        help="Disable progressive resizing (always 224)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg['save_path'] = 'checkpoints/best_model_v3.pth'
    cfg['metrics_path'] = 'logs/metrics_v3.json'

    # ── Config v3 ──
    FREEZE_EPOCHS = 5        # Freeze backbone for first N epochs
    TOTAL_EPOCHS = 50
    HEAD_LR = 1e-3
    BACKBONE_LR = 1e-5
    ACCUMULATION_STEPS = 4   # Effective batch = 32 * 4 = 128
    MIXUP_ALPHA = 0.2
    PATIENCE = 10
    USE_MIXUP = not args.no_mixup
    USE_PROGRESSIVE = not args.no_progressive

    print("=" * 65)
    print("  FACE ATTRIBUTE PREDICTION - TRAINING v3 (ALL IMPROVEMENTS)")
    print("=" * 65)
    print("  Improvements applied:")
    print("    ✓ Wing Loss + Adaptive Uncertainty Weighting")
    print("    ✓ Freeze backbone (5 epochs) → Differential LR")
    print("    ✓ Gradient Accumulation (eff. batch=128)")
    print(f"    {'✓' if USE_MIXUP else '✗'} Mixup augmentation (α={MIXUP_ALPHA})")
    print(f"    {'✓' if USE_PROGRESSIVE else '✗'} Progressive Resizing (160→192→224)")
    print(f"    ✓ Early Stopping (patience={PATIENCE})")
    print("    ✓ Multi-factor sampling (Race + Age group)")
    print("    ✓ Dropout 0.3 uniform")
    print("=" * 65)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    if device.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # Model
    model = FaceAttributeModel(pretrained=True, age_mode='regression').to(device)
    criterion = MultiTaskLoss(age_mode='regression', adaptive=True).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Phase 1: Freeze backbone, train heads only ──
    print(f"\n{'='*65}")
    print(f"  PHASE 1: FREEZE BACKBONE (Epochs 1-{FREEZE_EPOCHS})")
    print(f"{'='*65}")
    freeze_backbone(model)

    # Optimizer for heads only + criterion params
    head_params = [
        {'params': model.se.parameters(), 'lr': HEAD_LR},
        {'params': model.age_head.parameters(), 'lr': HEAD_LR},
        {'params': model.gender_head.parameters(), 'lr': HEAD_LR},
        {'params': model.race_head.parameters(), 'lr': HEAD_LR},
        {'params': criterion.parameters(), 'lr': HEAD_LR},  # adaptive weights
    ]
    optimizer = optim.AdamW(head_params, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    epoch_history = []
    no_improve_count = 0
    start_epoch = 1
    total_epochs = TOTAL_EPOCHS
    current_img_size = 160 if USE_PROGRESSIVE else 224

    # ── Resume ──
    if args.resume and os.path.exists(cfg['save_path']):
        checkpoint = torch.load(cfg['save_path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_metrics']['loss']
        total_epochs = checkpoint['epoch'] + args.extra_epochs

        if os.path.exists(cfg['metrics_path']):
            with open(cfg['metrics_path'], 'r', encoding='utf-8') as f:
                prev = json.load(f)
            epoch_history = prev.get('epoch_history', [])

        print(f"\n[Resume] Loaded from epoch {checkpoint['epoch']}")
        print(f"[Resume] Best val loss: {best_val_loss:.4f}")
        print(f"[Resume] Epochs {start_epoch} → {total_epochs}")

        # If past freeze phase, unfreeze and setup differential LR
        if start_epoch > FREEZE_EPOCHS:
            unfreeze_backbone(model)
            diff_params = get_differential_lr_params(model, BACKBONE_LR, HEAD_LR * 0.1)
            diff_params.append({'params': criterion.parameters(), 'lr': HEAD_LR * 0.1})
            optimizer = optim.AdamW(diff_params, weight_decay=1e-4)
    elif args.resume:
        print("[Warning] --resume specified but checkpoint not found, training from scratch")

    # ── Load initial data ──
    print(f"\n[Loading Data...] (img_size={current_img_size})")
    train_loader, val_loader, test_loader = load_data(
        cfg['csv_path'], cfg['img_dir'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        img_size=current_img_size
    )

    print(f"\n[Training] Epochs: {start_epoch}→{total_epochs}")
    print(f"[Training] AMP: {'ON' if device.type == 'cuda' else 'OFF'} | "
          f"Grad Clip: 1.0 | Accum Steps: {ACCUMULATION_STEPS}")
    print("-" * 65)

    phase2_started = (start_epoch > FREEZE_EPOCHS)

    for epoch in range(start_epoch, total_epochs + 1):
        start = time.time()

        # ── Phase transition: unfreeze backbone at epoch FREEZE_EPOCHS+1 ──
        if epoch == FREEZE_EPOCHS + 1 and not phase2_started:
            print(f"\n{'='*65}")
            print(f"  PHASE 2: UNFREEZE BACKBONE (Differential LR)")
            print(f"  Backbone LR: {BACKBONE_LR} | Heads LR: {HEAD_LR * 0.1}")
            print(f"{'='*65}")
            unfreeze_backbone(model)

            # New optimizer with differential learning rates
            diff_params = get_differential_lr_params(model, BACKBONE_LR, HEAD_LR * 0.1)
            diff_params.append({'params': criterion.parameters(), 'lr': HEAD_LR * 0.1})
            optimizer = optim.AdamW(diff_params, weight_decay=1e-4)
            scaler = GradScaler(enabled=(device.type == 'cuda'))
            phase2_started = True

        # ── Progressive Resizing ──
        if USE_PROGRESSIVE:
            new_img_size = get_img_size_for_epoch(epoch)
            if new_img_size != current_img_size:
                current_img_size = new_img_size
                print(f"\n  [Progressive Resize] → {current_img_size}x{current_img_size}")
                train_loader, val_loader, test_loader = load_data(
                    cfg['csv_path'], cfg['img_dir'],
                    batch_size=cfg['batch_size'],
                    num_workers=cfg['num_workers'],
                    img_size=current_img_size
                )
                # Reset GradScaler to prevent NaN from stale AMP state
                scaler = GradScaler(enabled=(device.type == 'cuda'))
                print(f"  [GradScaler] Reset for new image size")

        # ── Train & Validate ──
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA,
            accumulation_steps=ACCUMULATION_STEPS
        )
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        lr_now = optimizer.param_groups[0]['lr']

        # Log
        phase_str = "P1-Freeze" if epoch <= FREEZE_EPOCHS else "P2-Full"
        print(f"Epoch {epoch:02d}/{total_epochs} ({elapsed:.1f}s) "
              f"[{phase_str}] [LR: {lr_now:.6f}] [img: {current_img_size}]")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Age MAE: {train_metrics['age_mae']:.2f} | "
              f"Gender: {train_metrics['gender_acc']:.1f}% | "
              f"Race: {train_metrics['race_acc']:.1f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Age MAE: {val_metrics['age_mae']:.2f} | "
              f"Gender: {val_metrics['gender_acc']:.1f}% | "
              f"Race: {val_metrics['race_acc']:.1f}%")

        # Log adaptive weights
        if hasattr(criterion, 'log_var_age'):
            w_a = torch.exp(-criterion.log_var_age).item()
            w_g = torch.exp(-criterion.log_var_gender).item()
            w_r = torch.exp(-criterion.log_var_race).item()
            print(f"  Weights - Age: {w_a:.3f} | Gender: {w_g:.3f} | Race: {w_r:.3f}")

        # Save history
        epoch_history.append({
            'epoch': epoch,
            'time': round(elapsed, 1),
            'lr': lr_now,
            'img_size': current_img_size,
            'phase': phase_str,
            'train': train_metrics,
            'val': val_metrics,
        })

        # Skip saving if loss is NaN
        if math.isnan(val_metrics['loss']):
            print("  [Warning] Val loss is NaN, skipping save")
            sys.stdout.flush()
            continue

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'val_metrics': val_metrics,
                'age_mode': 'regression',
                'num_race_classes': NUM_RACE_CLASSES,
                'model_version': 'resnet50_se_v3',
                'img_size': current_img_size,
            }, cfg['save_path'])
            print(f"  ★ Best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= PATIENCE:
                print(f"\n[Early Stopping] No improvement for {PATIENCE} epochs.")
                break

        sys.stdout.flush()
        print()

    # ================================================================
    # Test Set Evaluation (reload best model at full resolution)
    # ================================================================
    print("=" * 65)
    print("  TEST SET EVALUATION (v3)")
    print("=" * 65)

    # Reload data at full resolution for fair comparison
    if current_img_size != 224:
        print("[Reload] Loading test data at 224x224 for evaluation...")
        _, _, test_loader = load_data(
            cfg['csv_path'], cfg['img_dir'],
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            img_size=224
        )

    checkpoint = torch.load(cfg['save_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_test_set(model, test_loader, device)

    print(f"  Age MAE:         {test_metrics['age_mae']:.2f} years")
    print(f"  Gender Accuracy: {test_metrics['gender_accuracy']:.1f}%")
    print(f"  Race Accuracy:   {test_metrics['race_accuracy']:.1f}%")
    print()

    # Print confusion matrices
    print("  Gender Confusion Matrix:")
    gender_names = [GENDER_LABELS[i] for i in range(NUM_GENDER_CLASSES)]
    print(f"    {'':>10} | " + " | ".join(f"{n:>8}" for n in gender_names))
    for i, row in enumerate(test_metrics['gender_confusion_matrix']):
        print(f"    {gender_names[i]:>10} | " + " | ".join(f"{v:>8d}" for v in row))
    print()

    print("  Race Confusion Matrix:")
    race_names = [RACE_LABELS[i] for i in range(NUM_RACE_CLASSES)]
    print(f"    {'':>10} | " + " | ".join(f"{n:>8}" for n in race_names))
    for i, row in enumerate(test_metrics['race_confusion_matrix']):
        print(f"    {race_names[i]:>10} | " + " | ".join(f"{v:>8d}" for v in row))
    print()

    # ================================================================
    # Compare with v2
    # ================================================================
    print("=" * 65)
    print("  SO SÁNH VỚI V2")
    print("=" * 65)
    v2_metrics_path = 'logs/metrics.json'
    if os.path.exists(v2_metrics_path):
        with open(v2_metrics_path, 'r', encoding='utf-8') as f:
            v2 = json.load(f)
        v2t = v2.get('test_metrics', {})
        print(f"  {'Metric':<20} {'v2':>10} {'v3':>10} {'Δ':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

        v2_age = v2t.get('age_mae', 0)
        v3_age = test_metrics['age_mae']
        d_age = v3_age - v2_age
        print(f"  {'Age MAE':<20} {v2_age:>10.2f} {v3_age:>10.2f} {d_age:>+10.2f}")

        v2_gender = v2t.get('gender_accuracy', 0)
        v3_gender = test_metrics['gender_accuracy']
        d_gender = v3_gender - v2_gender
        print(f"  {'Gender Acc':<20} {v2_gender:>9.1f}% {v3_gender:>9.1f}% {d_gender:>+9.1f}%")

        v2_race = v2t.get('race_accuracy', 0)
        v3_race = test_metrics['race_accuracy']
        d_race = v3_race - v2_race
        print(f"  {'Race Acc':<20} {v2_race:>9.1f}% {v3_race:>9.1f}% {d_race:>+9.1f}%")
    else:
        print("  [Info] V2 metrics not found, skip comparison")

    # ================================================================
    # Export metrics
    # ================================================================
    metrics_output = {
        'config': {**cfg, 'freeze_epochs': FREEZE_EPOCHS, 'head_lr': HEAD_LR,
                   'backbone_lr': BACKBONE_LR, 'accumulation_steps': ACCUMULATION_STEPS,
                   'mixup_alpha': MIXUP_ALPHA, 'patience': PATIENCE},
        'model_version': 'resnet50_se_v3',
        'improvements': [
            'Wing Loss', 'Adaptive Uncertainty Weighting',
            'Freeze backbone + Differential LR',
            'Gradient Accumulation (eff. batch=128)',
            'Mixup augmentation',
            'Progressive Resizing (160→192→224)',
            'Early Stopping (patience=10)',
            'Multi-factor sampling (Race + Age group)',
            'Dropout 0.3 uniform',
        ],
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss,
        'test_metrics': test_metrics,
        'epoch_history': epoch_history,
    }

    with open(cfg['metrics_path'], 'w', encoding='utf-8') as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)
    print(f"\n[Metrics] Saved to {cfg['metrics_path']}")

    print("=" * 65)
    print("Training v3 complete!")


if __name__ == '__main__':
    main()
