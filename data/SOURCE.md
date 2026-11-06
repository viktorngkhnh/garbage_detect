# Dataset Sources

The training dataset contains exactly 500 valid images for each class:
`cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash` (3,000 images total).

## Original Selection

- Source: Garbage Classification by CCHANGCS on Kaggle
- URL: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Retrieved: 2026-09-16
- Selection: 50 valid images per class
- License shown by Kaggle: Data files (c) Original Authors

## Added Selection

- Source: Garbage Dataset by Suman Kunwar on Kaggle
- URL: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
- Dataset version: 12
- Retrieved: 2026-09-21
- Source variant: `standardized_256`
- Selection: first 450 images per target class after ordinal path sorting
- Destination names: `v2_0001` through `v2_0450`, preserving file extensions
- License shown by Kaggle: MIT

All images retain their source class labels through the directory names under
`garbage_classification/`. SHA-256 verification found no duplicate images within
or across classes, and Pillow successfully decoded all 3,000 files.