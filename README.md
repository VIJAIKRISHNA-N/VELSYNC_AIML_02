# VELSYNC_AIML_02
# CIFAR-10 Image Classification using CNN

![CIFAR-10 Sample](screenshots/sample_images.png)

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** for classifying images from the **CIFAR-10 dataset**. CIFAR-10 contains 60,000 32x32 color images across **10 classes**:


The CNN is built using **TensorFlow/Keras** and is trained to recognize and classify these images with high accuracy. This project serves as a hands-on example for beginners in **deep learning and computer vision**.

---

## Dataset
- Dataset: **CIFAR-10**
- Training samples: 50,000 images
- Testing samples: 10,000 images
- Image size: 32x32 pixels, 3 color channels (RGB)
- Classes: 10

The dataset is loaded directly from **TensorFlow Keras datasets**.

---

## Model Architecture
The CNN model consists of:

- **Conv2D + MaxPooling2D** layers for feature extraction
- **Flatten layer** to convert 2D features into 1D
- **Dense layers** for classification
- **Output layer** with **softmax** activation for 10 classes


---

## Features
- Train CNN on CIFAR-10 dataset
- Normalize image pixel values
- Track **training and validation accuracy**
- Evaluate model on test data
- Visualize predictions on sample images

---

## Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/VIJAIKRISHNA-N/cifar10_image_classification.git
cd cifar10_image_classification
