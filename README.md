# Breast Cancer Classification using DenseNet-121 (Hybrid Model)

- This repository contains a hybrid deep-learning model built using DenseNet-121 + engineered radiomics features to classify breast histopathology images (benign vs malignant).
The project is part of my PBL work on early detection of breast cancer using AI.

##  Project Overview

- Breast cancer detection from microscopic images is challenging because cancer cells often show very subtle texture changes. To improve accuracy, this project combines:

-  DenseNet-121 deep features (pretrained on ImageNet)
-  Engineered radiomics features (GLCM, LBP, HOG, FFT, intensity stats, morphology)
-  A fully-connected classifier trained on the fused feature vector

- This hybrid format helps the model learn both high-level patterns and fine-texture details, leading to powerful performance even with limited data.

## Dataset — BreakHis

- The model is trained on the BreakHis dataset, which contains:

- 7,909 breast cancer histopathology images

- H&E-stained slides

- Classes: Benign and Malignant

- Magnifications: 40×, 100×, 200×, 400×

Dataset Link:
https://www.kaggle.com/datasets/ambarish/breakhis

## Method Summary
1. Preprocessing

- Resize images → 224×224

- Normalize using ImageNet mean & std

- Augmentation: flips, rotation, color jitter

2. DenseNet-121 Feature Extraction

- Load pretrained DenseNet-121

- Remove classifier → output 1024-dim vector

- Freeze early layers + fine-tune last dense block

3. Engineered Feature Extraction

- Extracted features:

- GLCM texture statistics

- LBP micro-texture features

- HOG gradients

- Intensity statistics (mean, variance, skewness, kurtosis)

- Edge/gradient descriptors (Canny, Sobel, Laplacian variance)

- FFT frequency features

- Morphology features (area, perimeter, solidity, eccentricity)

4. Feature Fusion

- DenseNet features (1024) + Radiomics (~147)

- Apply StandardScaler

- Apply PCA (retain 95% variance)

- Feed into a fully connected classifier

5. Training

- Loss: BCEWithLogits

- Optimizer: AdamW

- Scheduler: ReduceLROnPlateau

- Metrics: Accuracy, Precision, Recall, AUC, F1, Confusion Matrix

## 📊 Results

- The hybrid model achieved strong performance on the BreakHis test split:

- Metric	Score
- Accuracy	0.9560
- Balanced Accuracy	0.9438
- Precision	0.9658
- Recall (Sensitivity)	0.9727
- Specificity	0.9150
- F1 Score	0.9692
- ROC-AUC	0.9854
- Cohen’s Kappa	0.8923
- Matthews CC	0.8923
- Brier Score	0.0360
- Confusion Matrix

- True Positives: 818

- True Negatives: 312

- False Positives: 29

- False Negatives: 23

- Optimal Threshold (Youden Index)

- Optimal threshold: 0.598

- Accuracy @ optimal threshold: 0.9552

## Future Improvements

- In future updates, the model will be extended with:

- Training on larger datasets like CBIS-DDSM

- Adding EfficientNet / Vision Transformers

- Multi-magnification learning

- Improved explainability (Score-CAM, Integrated Gradients)

- Full web deployment version

## Author
Rohit Ranjan Kumar
B.Tech Manipal University Jaipur