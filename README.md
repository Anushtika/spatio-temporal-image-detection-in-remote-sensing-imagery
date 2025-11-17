## Generative Deep Learning Framework for Spatio-Temporal Change Detection in Remote Sensing Imagery

This repository contains the official implementation of ChangeFormerPlusPlus, a hybrid CNN + Transformer–based generative deep learning model designed for spatio-temporal change detection in high-resolution remote sensing imagery.
The model automatically identifies building construction, demolition, modification, and other structural changes by analyzing bi-temporal satellite images.

## Overview

ChangeFormerPlusPlus introduces a Siamese ResNet-like encoder, multi-head self-attention, and six Transformer layers to jointly learn spatial and temporal relationships between two time-stamped images.
A generative decoder with transposed convolutions reconstructs precise binary change maps.

The method is evaluated on the LEVIR-CD dataset, achieving reliable IoU and strong generalization through data augmentation and hybrid feature learning.

## Key Features

1. Siamese CNN Encoder for extracting spatial features from T1 and T2 images.

2. Multi-Head Attention + Transformer Blocks to capture long-range temporal dependencies.

3. Generative Decoder with transposed convolution layers for pixel-level reconstruction.

4. Hybrid Loss Function: Dice Loss + Binary Cross-Entropy to handle class imbalance.

5. Large-scale Augmentation: geometric transforms, color jitter, ImageNet normalization.

6. Trained on LEVIR-CD using 445 train, 64 validation, and 128 test image pairs 



## Dataset

Used the LEVIR-CD dataset (Google Earth, 0.5 m resolution), containing:

637 bi-temporal image pairs

Each resized to 256 × 256 pixels

Change labels include construction, demolition, structural modification

## Dataset split:

Training: 445 pairs

Validation: 64 pairs

Testing: 128 pairs 



Data augmentation includes:

Random flips

Rotations

Color jitter

ImageNet normalization


## Model Architecture

ChangeFormerPlusPlus consists of three components:

1. Siamese CNN Encoder

ResNet-style blocks

Shared weights for T1 and T2

Extracts hierarchical spatial features

2. Transformer Module

6 Transformer layers

Multi-Head Self-Attention

Learns temporal correlations and long-range spatial context

Attention computed as:

Attention(Q, K, V) = softmax((QKᵀ) / √dₖ) V


Generative_Deep_Learning_Framew…

3. Generative Decoder

Multiple transposed convolution layers

Upsampling and feature fusion

Outputs binary change map

🔧 Training Details

Framework: PyTorch 2.0

GPU: NVIDIA Tesla P100 (Kaggle Runtime)

Batch size: 8

Optimizer: Adam

Learning Rate: 1e-4

Weight Decay: 1e-5

Epochs: 10

Loss Function:

Dice Loss

Binary Cross-Entropy Loss

Total parameters: ~7.7M 

Model checkpoints saved on improvement of validation IoU.


## Results
Validation Metrics
Epoch	Val Loss	Val IoU
9	0.2844	0.4134
10	0.2862	0.4121




		
## Performance Summary

Best Validation IoU: 0.4134

Training IoU: 0.395

Overall Accuracy: 95.8%

Change Class F1-Score: 59.2%

No-Change Class F1-Score: 97.8%

Macro F1: 78.5% 



The model successfully captures:

Building construction

Building removal

Shape/structural changes

With strong boundary preservation and stable convergence.


## Visual Results

Change maps show:

Clear building outlines

Accurate segmentation of changed regions

Good separation between minor and major changes





## Project Structure
│── data/
│── src/
│   ├── model.py
│   ├── transformer.py
│   ├── dataset.py
│   ├── train.py
│   └── utils.py
│── saved_models/
│── README.md
│── requirements.txt

## Future Work

Extend to multi-sensor data (SAR + optical).

Add semantic change detection capabilities.

Optimize for real-time inference and edge devices.

Expand dataset and benchmark comparisons.




Cited: IEEE




