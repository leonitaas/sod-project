# End-to-End Salient Object Detection using Deep Learning

## Overview
This project implements an end-to-end Salient Object Detection (SOD) system using PyTorch and Convolutional Neural Networks (CNNs).  
The model predicts saliency masks that highlight the most visually important regions in an image.

The project includes:
- Data preprocessing pipeline
- CNN encoder-decoder architecture
- Custom loss function (BCE + IoU)
- Model training and evaluation
- Checkpointing system
- Visualization and demo interface

---

## Dataset
The project uses the **ECSSD dataset**, containing:
- 1000 natural images
- Corresponding pixel-level saliency masks

Dataset split:
- 70% Training
- 15% Validation
- 15% Testing

---

## Model Architecture
A custom encoder-decoder CNN architecture was implemented.

### Encoder
- Convolutional layers
- ReLU activations
- MaxPooling

### Decoder
- Transposed convolutions for upsampling
- Sigmoid output layer for mask prediction

An improved version of the model also includes:
- Batch Normalization
- Dropout regularization

---

## Loss Function
The training process uses a combined loss function:

Loss = BCE + 0.5 × (1 − IoU)

Where:
- BCE = Binary Cross Entropy
- IoU = Intersection over Union

---

## Evaluation Metrics
The model was evaluated using:
- IoU (Intersection over Union)
- Precision
- Recall
- F1 Score

### Baseline Results
| Metric | Score |
|---|---|
| IoU | 0.3000 |
| Precision | 0.5250 |
| Recall | 0.4212 |
| F1 Score | 0.4572 |

---

## Features
- Custom PyTorch Dataset class
- Training & validation pipeline
- Model checkpointing and resume support
- Prediction visualization
- Interactive demo in Google Colab

---

## Repository Structure

```text
SOD_Project.ipynb
README.md
report.pdf

best_model.pth

comparison_chart.png
training_curve.png
predictions_best.png
```

---

## Technologies Used
- Python
- GPU T4
- PyTorch
- NumPy
- Matplotlib
- Google Colab

---

## Demo
The notebook includes a simple interactive demo:
1. Upload an image
2. Generate saliency mask
3. Visualize overlay predictions

---

## Author
Leonita Sinani
