# Dataset — UTKFace (Organized)

## Cấu trúc

```
data/
├── raw/UTKFace/          # 23,704 ảnh gốc (backup)
├── processed/
│   ├── train/            # 16,592 ảnh (70%)
│   │   ├── White/        # 7,051
│   │   ├── Black/        # 3,141
│   │   ├── Asian/        # 2,425
│   │   └── Others/       # 3,975
│   ├── val/              # 3,556 ảnh (15%)
│   │   ├── White/
│   │   ├── Black/
│   │   ├── Asian/
│   │   └── Others/
│   └── test/             # 3,556 ảnh (15%)
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
| Train | 16,592 | 7,051 | 3,141 | 2,425 | 3,975 |
| Val   | 3,556 | 1,508 | 722 | 482 | 844 |
| Test  | 3,556 | 1,518 | 663 | 527 | 848 |

## Ghi chú
- Race mapping: Indian (3) + Others (4) → Others (3)
- Split seed: `random_state=42`, stratify by gender
- Ảnh format: RGB JPEG, filename encoding: `age_gender_race_timestamp.jpg`
