# Dataset Sources

## Current Dataset

The current training folders contain 1,983 images across these classes:

| Class folder | Images |
| --- | ---: |
| `01_Waste_Paper` | 200 |
| `02_Paper_Containers` | 200 |
| `03_Plastics` | 400 |
| `04_Iron_Aluminum_Cans` | 200 |
| `05_Glass` | 571 |
| `06_PET_Bottles` | 149 |
| `07_Waste_Batteries` | 169 |
| `08_Other_Recyclables` | 94 |

The provenance and license of the current images have not yet been recorded. Add the source and license for each new class before redistributing this dataset. `ImageFolder` uses these directory names as class labels, and `pytorch_resnet.py` stores them in the trained checkpoint.

## Previous Dataset History

The following source notes describe the previous six-class dataset, not the current image folders. That dataset contained 500 images per class (3,000 images total): `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.

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