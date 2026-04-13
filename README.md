<div align="center">

# Breast Cancer Detection

### Hybrid DenseNet-121 with Radiomic Feature Fusion and NLP-Powered Report Generation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini-NLP%20Reports-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)

---

A hybrid deep learning system that fuses **DenseNet-121 CNN features** with **engineered radiomic descriptors** for accurate breast histopathology classification (benign vs malignant), enhanced with **AI-generated clinical reports** via Google Gemini.

</div>

---

## Architecture

```
                    Input Image (224 x 224)
                           |
              +------------+------------+
              |                         |
     DenseNet-121 Backbone     Radiomic Feature Extraction
     (ImageNet pretrained)     (GLCM, LBP, HOG, FFT, etc.)
              |                         |
        1024-D features           147-D features
              |                         |
              |                    MLP Encoder
              |                    (256 -> 128)
              |                         |
              +-------Concatenate-------+
                           |
                     Fusion Classifier
                     (1152 -> 512 -> 128 -> 1)
                           |
                    Benign / Malignant
                           |
                  Grad-CAM Heatmap + Gemini NLP Report
```

## Key Features

- **Hybrid Fusion Model** -- Combines deep CNN features with hand-crafted radiomic descriptors for robust classification
- **Grad-CAM Explainability** -- Visual heatmaps highlighting regions the model focuses on for its decision
- **NLP Clinical Reports** -- Automatic generation of professional pathology-style reports using Google Gemini
- **Interactive Web App** -- Streamlit-based interface for image upload, prediction, and interpretation
- **Graceful Fallback** -- Works fully offline with static explanations when no API key is configured

## Dataset

The model is trained on the [BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis) dataset:

| Property          | Value                                   |
|-------------------|-----------------------------------------|
| Total images      | 7,909                                   |
| Staining          | H&E (Hematoxylin and Eosin)             |
| Classes           | Benign, Malignant                       |
| Magnifications    | 40x, 100x, 200x, 400x                  |
| Image format      | PNG (700 x 460)                         |

## Method

### 1. Preprocessing

- Resize to 224 x 224
- Normalize with ImageNet mean and standard deviation
- Augmentation: random flips, rotation, color jitter

### 2. DenseNet-121 Feature Extraction

- Pretrained DenseNet-121 with classifier head removed
- Outputs a 1024-dimensional feature vector per image
- Early layers frozen; last dense block fine-tuned

### 3. Radiomic Feature Engineering (147 features)

| Category                | Features                                           |
|-------------------------|----------------------------------------------------|
| Texture (GLCM)          | Contrast, correlation, energy, homogeneity         |
| Micro-texture (LBP)     | Local binary pattern histogram                     |
| Gradient (HOG)          | Histogram of oriented gradients                    |
| Intensity Statistics    | Mean, variance, skewness, kurtosis                 |
| Edge Descriptors        | Canny edges, Sobel gradients, Laplacian variance   |
| Frequency Domain (FFT)  | Fourier transform magnitude features               |
| Morphology              | Area, perimeter, solidity, eccentricity             |

### 4. Feature Fusion and Classification

- Concatenate DenseNet (1024-D) + Radiomic (128-D after MLP encoding)
- StandardScaler normalization and PCA (95% variance retained)
- Fully connected classifier with BatchNorm and Dropout

### 5. Training Configuration

| Parameter     | Value              |
|---------------|--------------------|
| Loss          | BCEWithLogitsLoss  |
| Optimizer     | AdamW              |
| Scheduler     | ReduceLROnPlateau  |
| Batch size    | 32                 |

### 6. NLP Report Generation

After classification, the model's prediction, confidence score, and Grad-CAM activation statistics are passed to **Google Gemini** to generate a structured clinical report including:

- Clinical summary interpreting the computational findings
- Key tissue/cellular features associated with the classification
- Recommended next steps

Reports include a disclaimer that they are AI-generated and require pathologist review.

## Results

| Metric                | Score   |
|-----------------------|---------|
| Accuracy              | 0.9560  |
| Balanced Accuracy     | 0.9438  |
| Precision             | 0.9658  |
| Recall (Sensitivity)  | 0.9727  |
| Specificity           | 0.9150  |
| F1 Score              | 0.9692  |
| ROC-AUC               | 0.9854  |
| Cohen's Kappa         | 0.8923  |
| Matthews Correlation  | 0.8923  |
| Brier Score           | 0.0360  |

**Confusion Matrix:**

|                  | Predicted Benign | Predicted Malignant |
|------------------|------------------|---------------------|
| Actual Benign    | 312 (TN)         | 29 (FP)             |
| Actual Malignant | 23 (FN)          | 818 (TP)            |

**Optimal Threshold (Youden Index):** 0.598 | Accuracy at threshold: 0.9552

## Setup

```bash
# Clone the repository
git clone https://github.com/Rohit1x52/Breast-Cancer-Detection-using-DenseNet-121.git
cd Breast-Cancer-Detection-using-DenseNet-121

# Install dependencies
pip install -r requirements.txt

# (Optional) Configure NLP report generation
# Get a free API key from https://aistudio.google.com/apikey
echo GEMINI_API_KEY="your-api-key-here" > .env

# Run the application
streamlit run densenet.py
```

## Project Structure

```
.
├── densenet.py               # Main Streamlit app (inference + NLP reports)
├── densenet.ipynb             # Training notebook
├── best_hybrid_densenet.pth   # Trained model weights
├── requirements.txt           # Python dependencies
├── .env                       # API key (not tracked by git)
├── BreakHis/                  # Dataset directory
└── figures/                   # Training plots and visualizations
```

## Future Work

- Training on larger datasets (CBIS-DDSM, VinDr-Mammo)
- Vision Transformer (ViT) backbone comparison
- Multi-magnification learning pipeline
- Advanced explainability (Score-CAM, Integrated Gradients)
- RAG-based Q&A chatbot for interactive diagnosis queries

---

<div align="center">

**Author:** Rohit Ranjan Kumar | B.Tech, Manipal University Jaipur

</div>