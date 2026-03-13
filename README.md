# Kidney Disease Detection – FSGS Classification

This project implements a deep learning-based system to detect **Focal Segmental Glomerulosclerosis (FSGS)** from kidney biopsy images.

## Features
- Image classification using deep learning
- Transfer learning with ResNet50 and VGG16
- Image preprocessing and augmentation
- Prediction of Normal vs Sclerosed glomeruli

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib

## Model
The model uses **transfer learning** on pretrained CNN architectures to improve classification accuracy on medical image data.

## Dataset
Kidney biopsy histopathological image dataset.

## Results
The model achieved strong classification performance in detecting glomerular sclerosis patterns.

## Future Improvements
- Increase dataset size
- Deploy as web application
- Add Grad-CAM visualization for interpretability
