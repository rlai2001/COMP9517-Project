# COMP9517 Group Project: Remote Sensing Image Classification

## Overview
This project implements and compares three classification approaches on the SkyView dataset:

- **CNN**:
- **KNN**:
- **ResNet18**: Transfer learning using a pretrained ResNet18 from PyTorch, modified for multi-class classification.

The goal is to evaluate the strengths and limitations of deep learning versus traditional feature-based methods for remote sensing scene classification.

## Dataset
**SkyView Dataset (Kaggle)**  
15 aerial classes × 800 images per class = 12,000 images total.

> 🔗 Download from: [https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset]

After downloading, organize it as follows:
```
skyview_dataset/
└── split/
    ├── train/
    └── test/
```
Each `train/` and `test/` directory should contain 15 class folders: `Agriculture`, `Airport`, ..., `River`

Dataset is **not included** in this repository to keep the size minimal.

---

## Directory Structure
```
project_root/
├── models/               
├── outputs/              # Evaluation plots, confusion matrices, reports
├── scripts/              # Python scripts for training
├── skyview_dataset/      # Redirectory to the original dataset
├── report/               # Final report, figures, and demo video
└── README.md
```
## Method 1: CNN
## Method 2: KNN
## Method 3:

## Method 4: ResNet18 (by Richard Lai)

This implementation fine-tunes a pretrained ResNet18 on the SkyView dataset.

### Key Features
- Pretrained on ImageNet (`weights=IMAGENET1K_V1`)
- Custom classifier head for 15 output classes
- Tracked training & validation metrics:
  - Loss
  - Accuracy
  - F1 Score
- Confusion matrix & classification report saved per epoch
- Best model checkpoint saved automatically

### File:
- `scripts/train_resnet.py`

### How to Run
```bash
python scripts/train_resnet.py --data_dir skyview_dataset/split --epochs 10 --batch_size 32
```
Output will be saved to:
- `models/`: All model checkpoints (including `best_model.pth`)
- `outputs/`: 
  - Classification reports per epoch
  - Confusion matrices
  - `training_curves.png`

---
## Dependencies
- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- tqdm

Install via:
```bash
pip install -r requirements.txt
```

If training ResNet, ensure you have:
- PyTorch (with GPU support if possible)
- torchvision
- scikit-learn
---

## Results 

**(ResNet18)**
Final evaluation:
- Accuracy: ~94%
- Macro F1: ~94%
- Most confusion: between Airport vs Port, Grassland vs Forest

Visuals:
-  Confusion matrices are saved to: `outputs/confusion_matrixResnet.png`
- All training curves (loss, accuracy, F1) are saved to: `outputs/training_curvesResnet.png`
- Final classification reports are saved to: `outputs/classification_report_epoch*.txt`
---


## Contributors
| Name         | zID       | Contribution                   |
|--------------|-----------|--------------------------------|
| Teammate     | zxxxxxx   | CNN training & analysis        |
| Teammate     | zxxxxxx   | KNN pipeline & feature methods |
| Teammate     | zxxxxxx   |                                |
| Richard Lai  | zxxxxxx   | ResNet18 model + evaluation    |     

## Report and Submission
- Final Report: `report/Final_Report.pdf`
- Project Video: `report/demo.mp4`

## Acknowledgements
- Dataset: SkyView Aerial Imagery (Kaggle)
- Pretrained model: torchvision.models.resnet18
- Traditional feature extraction: OpenCV SIFT / LBP

