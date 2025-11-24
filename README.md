# Handwritten Digit Classifier (CNN)

A convolutional neural network (CNN) for classifying handwritten digits using TensorFlow/Keras.  
<img width="485" height="349" alt="ML_pic" src="https://github.com/user-attachments/assets/61f9e837-a7b7-45fe-aced-36dca6681410" />


<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/61f9e837-a7b7-45fe-aced-36dca6681410" />
    alt="ML_pic"
    width="420"
  />
</p>

## 🧠 Project Overview

This repository implements an end-to-end pipeline for digit classification:

- Loads training and validation data from CSV files (`A4train.csv`, `A4val.csv`)
- Preprocesses the images into 84×28 grayscale tensors
- Trains a deep CNN with **convolutions, batch normalization, pooling, and dropout**
- Uses data augmentation to improve generalization
- Evaluates the trained model on a held-out validation set and reports accuracy

The main goal is to explore modern CNN design patterns (BN, dropout, data augmentation) on a digit-recognition task.

---

## 🏗 Model Architecture

The model is defined in `learn(X, y)` and consists of:

- **Input:** 84×28×1 grayscale image
- **Convolutional blocks:**
  - 2 × Conv2D(32, 3×3, ReLU) + BatchNorm + MaxPooling + Dropout
  - 2 × Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPooling + Dropout
  - 2 × Conv2D(128, 3×3, ReLU) + BatchNorm + MaxPooling + Dropout
- **Fully connected head:**
  - Flatten
  - Dense(256, ReLU) + BatchNorm + Dropout(0.5)
  - Dense(10, Softmax) for digit classes 0–9

**Optimization**

- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metrics: `accuracy`

**Regularization & Training Tricks**

- `ImageDataGenerator` for data augmentation:
  - random rotations, zoom, width/height shifts
- `EarlyStopping` on validation accuracy
- `ReduceLROnPlateau` to lower the learning rate when validation accuracy plateaus

---

## 📊 Dataset

Training and validation data are stored as CSV files:

- `A4data/A4train.csv`
- `A4data/A4val.csv`

Each row has the format:

- **Column 0:** digit label (0–9)  
- **Columns 1–2352:** flattened pixel intensities for an 84×28 grayscale image

The code reshapes these vectors as:

```python
X_images = X.reshape(-1, 84, 28, 1) / 255.0
