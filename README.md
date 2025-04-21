# COMP9517 Group Project: Remote Sensing Image Classification

## Overview 
This project aims to develop and compare multiple computer vision methods for classifying aerial landscape imagery into 15 predefined categories, further evaluate the strengths and limitations of deep learning versus traditional feature-based methods for remote sensing scene classification.

## Dataset
**SkyView Dataset (Kaggle)**  
15 aerial classes × 800 images per class = 12,000 images total.

> 🔗 Download from: [https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset]

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

'''bash
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
'''

## Evaluation Criteria

Models were evaluated based on:
- Classification Accuracy
- Precision
- Recall
- F1-Score (Macro and Weighted)
- Confusion Matrix Analysis
- 
## References
- SkyView Dataset: https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset  
- ResNet: https://arxiv.org/abs/1512.03385  
- DenseNet: https://arxiv.org/abs/1608.06993  
- ViT: https://arxiv.org/abs/2010.11929  
- Swin Transformer: https://arxiv.org/abs/2103.14030  
- Grad-CAM: https://arxiv.org/abs/1610.02391  
