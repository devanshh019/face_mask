# 🧠 Face Mask Detection using CNN

A deep learning project that detects whether a person is wearing a mask or not using **Convolutional Neural Networks (CNNs)** and **OpenCV** for real-time video detection.

---

## 🚀 Project Overview

This project implements a real-time **Face Mask Detection** system that can identify masked and unmasked faces using a webcam or video feed.

The goal is to use deep learning (CNNs) to assist in maintaining safety protocols by automatically detecting mask usage in real-world environments.

---

## 🧩 Features

- Real-time detection using **OpenCV**
- Trained **CNN model** from scratch on a labeled dataset
- **92% validation accuracy**
- Data augmentation for better generalization
- Easy to integrate with cameras or CCTV streams

---

## 🧠 Model Architecture

The CNN model consists of:
- 3 convolutional layers (ReLU activation)
- MaxPooling and Dropout layers
- Flatten and Dense layers
- Output layer with Softmax activation (2 classes: Mask / No Mask)

**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Accuracy:** ~92% on validation data

---

## 🗂️ Dataset

- Custom dataset with ~3,000 labeled images  
- Two classes:  
  - `with_mask`  
  - `without_mask`

(You can use your own dataset or public datasets such as Kaggle’s [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection))
