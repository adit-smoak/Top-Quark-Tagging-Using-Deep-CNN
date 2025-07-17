# Top Quark Classification using Deep Learning

A deep learning solution for real-time classification of particle jets into **standard background** or **top quark candidates**, leveraging custom-designed convolutional neural networks (CNN) in **MATLAB**.

## 🚀 Project Overview

This project applies modern deep learning techniques to high-energy physics data to solve a real-world classification problem. The network processes **multi-channel jet images** (local features) along with **global event-level features** like:

- **Energy Skewness**
- **pT Skewness**
- **Energy Kurtosis**
- **pT Kurtosis**

The goal is to enable fast, reliable filtering of collision data streams, potentially applicable to particle detectors like those at CERN.

## 📊 Dataset

The dataset consists of **jet images** (37x37x12 tensors) and corresponding **global features**, pre-processed into MATLAB `.mat` files.

To access the dataset:  
[Click here to download the dataset](https://zenodo.org/records/2603256#.Y20xysvMLmE)  <!-- Replace with actual dataset link -->

## Network Architecture

- **Input 1:** Jet Images (processed via 2D CNN layers)
- **Input 2:** 4 Global Features (processed via fully connected layers)
- Fusion of both streams followed by classification.

Built using **MATLAB's Deep Network Designer** for fast prototyping and visualization.

## Key Hyperparameters

| Hyperparameter        | Value       | Reason                                         |
|------------------------|------------|------------------------------------------------|
| Initial Learn Rate     | 1e-2       | Enables faster initial convergence.            |
| Learn Rate Schedule    | Piecewise  | Allows controlled learning rate decay.         |
| Learn Rate Drop Period | 5 epochs   | Reduces LR after every 5 epochs.               |
| L2 Regularization      | 1e-4       | Prevents overfitting by penalizing large weights. |
| Batch Size             | 64         | Balanced between training speed and VRAM limits. |
| Max Epochs             | 20         | Prevents overtraining; saturation after ~14 epochs. |
| Optimizer              | Adam       | Efficient and adaptive learning optimization.  |

## Performance

- Achieved **89.4% validation accuracy** using 60k training samples.
- Achieved **89.85% validation accuracy** using 90k training samples.
- Suitable for **real-time event filtering** in experimental setups.

## 💻 Hardware Requirements

- **Minimum**: NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM, 16 GB RAM.
- Optimized for MATLAB 2024b or later.

## 📂 Folder Structure

