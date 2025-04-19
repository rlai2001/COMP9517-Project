# COMP9517 Group Project: Remote Sensing Image Classification

## Overview
This project implements and compares three classification approaches on the SkyView dataset:

- **CNN**:
- **KNN**:
- **ResNet18**: Transfer learning using a pretrained ResNet18 from PyTorch, modified for multi-class classification.

The goal is to evaluate the strengths and limitations of deep learning versus traditional feature-based methods for remote sensing scene classification.

## Directory Structure
```
project_root/
├── models/               # Saved PyTorch .pth model weights
├── outputs/              # Evaluation plots, confusion matrices, reports
├── scripts/              # Python scripts for training and evaluation
├── skyview_dataset/      # Dataset folder with train/test split
├── report/               # Final report, figures, and demo video
└── README.md
```

## How to Run

### 1. Train ResNet18
```bash
python scripts/train_resnet.py --data_dir path/to/split --epochs 10
```

### 2. Evaluate ResNet18 Model
```bash
python scripts/eval_model.py --model_path models/resnet_epoch10.pth
```

### 3. Train CNN
```bash
python scripts/train_cnn.py --epochs 15
```

### 4. Run KNN Classifier
```bash
python scripts/run_knn.py --features sift
```

## Dependencies
- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- tqdm

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Results
- All training curves (loss, accuracy, F1) are saved to: `outputs/training_curves.png`
- Confusion matrices are saved per epoch to: `outputs/confusion_matrix_epoch*.png`
- Final classification reports are saved to: `outputs/classification_report_epoch*.txt`

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

