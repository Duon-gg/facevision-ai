"""
organize_data.py — Tổ chức lại thư mục data thành cấu trúc chuyên nghiệp.

TRƯỚC:
  data/
    UTKFace/       ← 23,707 ảnh gộp chung
    labels.csv

SAU:
  data/
    raw/UTKFace/   ← ảnh gốc (giữ nguyên để tham chiếu)
    processed/
      train/
        White/
        Black/
        Asian/
        Others/
      val/
        White/...
      test/
        White/...
    labels.csv
    train.csv
    val.csv
    test.csv
    README.md

Sử dụng SYMBOLIC LINKS (symlinks) để không tốn thêm dung lượng ổ đĩa.
Nếu symlink lỗi (cần admin trên Windows) → copy file thay thế.
"""

import os
import sys
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Config
# ============================================================
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw', 'UTKFace')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
CSV_PATH = os.path.join(DATA_DIR, 'labels.csv')
OLD_IMG_DIR = os.path.join(DATA_DIR, 'UTKFace')

RACE_REMAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
RACE_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Others'}
SPLITS = ['train', 'val', 'test']


def main():
    print("=" * 65)
    print("  TỔ CHỨC LẠI THƯ MỤC DATA")
    print("=" * 65)

    # ── Load CSV ──
    df = pd.read_csv(CSV_PATH)
    df['exists'] = df['image'].apply(lambda x: os.path.exists(os.path.join(OLD_IMG_DIR, x)))
    df = df[df['exists']].drop(columns=['exists']).reset_index(drop=True)
    print(f"\n[Dataset] Total valid: {len(df)}")

    # Remap race
    df['race_remapped'] = df['race'].map(RACE_REMAP)
    df['race_label'] = df['race_remapped'].map(RACE_NAMES)

    # ── Split 70/15/15 (same seed as loader.py) ──
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['gender']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['gender']
    )

    splits_data = {'train': train_df, 'val': val_df, 'test': test_df}
    for name, sdf in splits_data.items():
        print(f"  {name}: {len(sdf)} images")

    # ── Tạo thư mục ──
    print("\n[1/4] Tạo cấu trúc thư mục...")
    for split in SPLITS:
        for race_name in RACE_NAMES.values():
            dir_path = os.path.join(PROCESSED_DIR, split, race_name)
            os.makedirs(dir_path, exist_ok=True)
    print("  ✓ Tạo xong processed/train|val|test/White|Black|Asian|Others")

    # ── Di chuyển ảnh gốc vào raw/ ──
    print("\n[2/4] Di chuyển ảnh gốc vào data/raw/UTKFace/...")
    if os.path.exists(OLD_IMG_DIR) and not os.path.exists(RAW_DIR):
        os.makedirs(os.path.dirname(RAW_DIR), exist_ok=True)
        os.rename(OLD_IMG_DIR, RAW_DIR)
        print(f"  ✓ Đã rename UTKFace/ → raw/UTKFace/")
    elif os.path.exists(RAW_DIR):
        print(f"  ✓ raw/UTKFace/ đã tồn tại, skip")
    else:
        print(f"  ✗ Không tìm thấy {OLD_IMG_DIR}")
        return

    # ── Copy ảnh vào processed/ theo split + class ──
    print("\n[3/4] Phân phối ảnh vào processed/ (symlink hoặc copy)...")
    use_symlink = True
    total_copied = 0
    errors = 0

    for split_name, sdf in splits_data.items():
        count = 0
        for _, row in sdf.iterrows():
            src = os.path.abspath(os.path.join(RAW_DIR, row['image']))
            race_label = RACE_NAMES[RACE_REMAP.get(row['race'], 3)]
            dst = os.path.join(PROCESSED_DIR, split_name, race_label, row['image'])

            if os.path.exists(dst):
                count += 1
                continue

            if not os.path.exists(src):
                errors += 1
                continue

            try:
                if use_symlink:
                    os.symlink(src, dst)
                else:
                    shutil.copy2(src, dst)
                count += 1
            except OSError:
                # Symlink failed (Windows no admin) → switch to copy
                if use_symlink:
                    print(f"  [Info] Symlink failed, switching to file copy...")
                    use_symlink = False
                shutil.copy2(src, dst)
                count += 1

        total_copied += count
        print(f"  {split_name}: {count} images distributed")

    if errors > 0:
        print(f"  [Warning] {errors} files not found, skipped")

    # ── Export split CSVs ──
    print("\n[4/4] Xuất CSV cho từng split...")
    for split_name, sdf in splits_data.items():
        csv_out = os.path.join(DATA_DIR, f'{split_name}.csv')
        sdf_out = sdf[['image', 'age', 'gender', 'race', 'race_remapped', 'race_label']].copy()
        sdf_out.to_csv(csv_out, index=False)
        print(f"  ✓ {csv_out} ({len(sdf_out)} rows)")

    # ── Tạo README ──
    readme = f"""# Dataset — UTKFace (Organized)

## Cấu trúc

```
data/
├── raw/UTKFace/          # {len(df):,} ảnh gốc (backup)
├── processed/
│   ├── train/            # {len(train_df):,} ảnh (70%)
│   │   ├── White/        # {(train_df.race_label=='White').sum():,}
│   │   ├── Black/        # {(train_df.race_label=='Black').sum():,}
│   │   ├── Asian/        # {(train_df.race_label=='Asian').sum():,}
│   │   └── Others/       # {(train_df.race_label=='Others').sum():,}
│   ├── val/              # {len(val_df):,} ảnh (15%)
│   │   ├── White/
│   │   ├── Black/
│   │   ├── Asian/
│   │   └── Others/
│   └── test/             # {len(test_df):,} ảnh (15%)
│       ├── White/
│       ├── Black/
│       ├── Asian/
│       └── Others/
├── labels.csv            # Labels gốc (5 class)
├── train.csv             # Labels train split
├── val.csv               # Labels val split
└── test.csv              # Labels test split
```

## Thống kê

| Split | Total | White | Black | Asian | Others |
|-------|-------|-------|-------|-------|--------|
| Train | {len(train_df):,} | {(train_df.race_label=='White').sum():,} | {(train_df.race_label=='Black').sum():,} | {(train_df.race_label=='Asian').sum():,} | {(train_df.race_label=='Others').sum():,} |
| Val   | {len(val_df):,} | {(val_df.race_label=='White').sum():,} | {(val_df.race_label=='Black').sum():,} | {(val_df.race_label=='Asian').sum():,} | {(val_df.race_label=='Others').sum():,} |
| Test  | {len(test_df):,} | {(test_df.race_label=='White').sum():,} | {(test_df.race_label=='Black').sum():,} | {(test_df.race_label=='Asian').sum():,} | {(test_df.race_label=='Others').sum():,} |

## Ghi chú
- Race mapping: Indian (3) + Others (4) → Others (3)
- Split seed: `random_state=42`, stratify by gender
- Ảnh format: RGB JPEG, filename encoding: `age_gender_race_timestamp.jpg`
"""
    with open(os.path.join(DATA_DIR, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"  ✓ data/README.md")

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  ✅ HOÀN TẤT!")
    print("=" * 65)
    print(f"  Tổng: {total_copied:,} ảnh đã phân phối")
    print(f"  Mode: {'symlink' if use_symlink else 'copy'}")
    print()
    print("  Cấu trúc mới:")
    print("    data/raw/UTKFace/     ← ảnh gốc")
    print("    data/processed/       ← chia theo split + class")
    print("    data/train|val|test.csv")
    print("=" * 65)


if __name__ == '__main__':
    main()
