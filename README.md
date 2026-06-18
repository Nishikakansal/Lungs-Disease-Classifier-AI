# Lung Disease Classifier AI 🫁

A deep learning model that classifies 5 lung diseases from chest X-ray images 
using DenseNet121 with transfer learning.

## Results
- Validation Accuracy: ~85%
- 5-class classification (Normal, Bacterial Pneumonia, Viral Pneumonia, COVID-19, Tuberculosis)
- No overfitting — validation accuracy consistently outperformed training accuracy

## Model Architecture
- Base: DenseNet121 pretrained on ImageNet
- Global Average Pooling → Batch Normalization → Dense (512, ReLU) → Dropout (0.5) → Softmax
- Two-phase training: freeze base (20 epochs) → fine-tune last 30 layers (10 epochs, lr=1e-5)

## Dataset
Lungs Disease Dataset from Kaggle — 5 categories of labeled chest X-ray images,
pre-split into train/validation/test sets.

## Tech Stack
Python, TensorFlow, Keras, OpenCV, NumPy, Scikit-learn, Matplotlib

## How to Run
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Run: python app.py

## Team
Built as a Team of 2 and part of Bennett University coursework (SCSET Department)

## Model Weights
The trained DenseNet121 model weights (.keras file) are not included 
in this repo due to file size. 
Contact: nishikakansal@gmail.com to request access.
