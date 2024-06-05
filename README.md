# Segmentation-CBSA-Module

## Convolution Based Self-Attention Module (CBSA): Segmentation for Metallographic Image 

### Description

This repository contains the implementation of CBSA (Convolution Based Self-Attention) Module, a deep learning module specifically designed for the task of metallographic image segmentation, which can be fused with any convolutional network. Metallographic image segmentation is a critical step in the analysis of metal microstructures, which provides valuable information about the properties and quality of metal materials.

### requirements

torch>=1.10.0
torchvision>=0.11.1
numpy>=1.21.2
Pillow>=8.3.2
opencv-python>=4.5.3.56
tqdm>=4.62.3
matplotlib>=3.4.3
scipy>=1.7.1
tensorboard>=2.7.0

### Features

- **Contextual Boundary-Aware Mechanism:** CBSA leverages contextual information and boundary awareness to enhance the segmentation accuracy of metallographic images.
- **Flexible Backbone Choices:** The model based on U-Net structure,  supports multiple backbone architectures, including VGG16 and ResNet50.
- **Mixed Precision Training:** Supports FP16 mixed precision training to reduce memory usage and speed up the training process.
- **Customizable Loss Functions:** Integrates various loss functions, including Dice Loss and Focal Loss, to tackle class imbalance and improve segmentation performance.

### Dataset

The model is trained and evaluated on metallographic images, which are typically acquired using optical or electron microscopy. The dataset should be formatted in the VOC format, with input images in the `JPEGImages` directory and corresponding segmentation masks in the `SegmentationClass` directory.
