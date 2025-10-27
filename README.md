# Comparison of ResNet-50 and EfficientNet-B0 for Binary Image Classification: A Case Study on Cancer vs Non-Cancer Detection

## Overview
This project compares the performance of two pre-trained deep learning models— ResNet-50 and EfficientNet-B0 for binary classification of medical images into cancer and non-cancer categories.  
The primary objective is to determine which model provides superior accuracy, generalization, and computational efficiency for cancer detection tasks.

## Objectives
- Build and fine-tune two CNN-based models (ResNet-50 and EfficientNet-B0).  
- Evaluate their performance on medical image data using precision, recall, and F1-score.  
- Identify which model balances accuracy and efficiency for clinical application scenarios.

## Data Preparation
- Dataset: Medical image dataset consisting of two classes — cancer and non-cancer.  
- Split: 80% training, 20% testing  using StratifiedShuffleSplit for class balance.  
- Transformations:
  - Training: RandomResizedCrop, RandomHorizontalFlip, Normalization  
  - Testing: Resize, CenterCrop, Normalization  

## Model Implementation
- ResNet-50:
  - Base: Pre-trained on ImageNet  
  - Modified fully connected layer for binary classification  
  - Optimizer: `Adam(lr=1e-5)`  
  - Weighted CrossEntropyLoss to handle class imbalance  

- EfficientNet-B0:
  - Base: Pre-trained on ImageNet  
  - Modified final fully connected layer for binary classification  
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`  
  - Weighted CrossEntropyLoss for balanced learning  

## Results Summary
- ResNet-50:
  - Accuracy: ~92%  
  - F1-score: 0.92  
  - Strong generalization and higher recall for non-cancer detection  

- EfficientNet-B0:
  - Accuracy: ~71%  
  - F1-score: 0.69  
  - Computationally efficient but less accurate under the same training setup  

## Discussion
Both models showed promising results in classifying cancerous vs non-cancerous images.  
ResNet-50 achieved higher overall performance and better recall for the non-cancer class, while EfficientNet-B0 offered lighter computation with acceptable accuracy.  
The findings suggest that ResNet-50 is more reliable for clinical image classification tasks, though EfficientNet-B0 remains a viable choice for low-resource environments.

## Tools and Libraries
`Python`, `PyTorch`, `Torchvision`, `EfficientNet-PyTorch`, `Matplotlib`, `Seaborn`, `Scikit-learn`

4. The training and evaluation logs will display classification reports and confusion matrices for both models.

