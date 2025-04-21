# COMP9517 Group Project: Remote Sensing Image Classification

## Overview
This project implements and compares six classification approaches on the SkyView dataset:

- **ViT-Base-224**
- **Swin-Base-224**
- **DenseNet-121** 
- **ResNet18**
- **Plus_Extreme_CNN**
- **SVM**

The goal is to evaluate the strengths and limitations of deep learning versus traditional feature-based methods for remote sensing scene classification.

## Dataset
**SkyView Dataset (Kaggle)**  
15 aerial classes × 800 images per class = 12,000 images total.

> 🔗 Download from: [https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset]

After downloading, organise it as follows:
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
## Methods Implemented

The project includes the implementation and evaluation of the following models:

### Traditional Machine Learning
- Support Vector Machine (SVM) with handcrafted features

### Deep Learning Models (Transfer Learning)
- ResNet-18
- DenseNet-121
- Vision Transformer (ViT-Base-224)
- Swin Transformer (Swin-Base-224)
- Plus_Extreme_CNN (custom CNN variant)

Each model was trained on 80% of the dataset and validated on the remaining 20%. Evaluation metrics include accuracy, precision, recall, and F1-score.

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
**(DenseNet-121)**

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

## Group Members and Contributions

| Name           | Student ID | Contribution                          |
|----------------|------------|----------------------------------------|
| **Shixun Li**  | z5505146   | Swin-Base-224, Plus_Extreme_CNN       |
| **Qiyun Li**   | z5504759   | Vision Transformer (ViT-Base-224)     |
| **Xinbo Li**   | z5496624   | DenseNet-121                          |
| **Richard Lai**| z5620374   | ResNet-18                             |
| **Junle Zhao** | z5447039   | SVM with traditional features         |   

## Report and Submission
- Final Report: `report/Final_Report.pdf`
- Project Video: `report/demo.mp4`

## How to Run

bash
# Step 1: Clone the repository
git clone https://github.com/rlai2001/COMP9517-Project.git
cd COMP9517-Project

# Step 2: Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # For Windows: venv\\Scripts\\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Execute training scripts
python train_resnet.py
python train_densenet.py
# COMP9517 T1 2025 Group Project - Aerial Scene Classification

## Acknowledgements
- Dataset: SkyView Aerial Imagery (Kaggle)
- Pretrained model: torchvision.models.resnet18
- Traditional feature extraction: OpenCV SIFT / LBP

This repository contains the code and documentation for the COMP9517 Computer Vision group project at UNSW, Term 1 2025.  
The objective of this project is to develop and compare multiple computer vision methods for the classification of aerial landscape imagery into 15 predefined categories.
