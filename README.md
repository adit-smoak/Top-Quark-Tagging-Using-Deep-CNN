# Top Quark Classification using Deep Learning

A deep CNN based solution for real-time classification of particle jets into **standard background** or **top quark**, using custom convolutional blocks in **MATLAB Deep Network Designer**.

## Project Overview

This project applies modern deep learning techniques to high-energy physics data to solve a real-world classification problem. The network processes **multi-channel jet images** (local features) along with **global event-level features** like:

- **Energy Skewness**
- **pT Skewness**
- **Energy Kurtosis**
- **pT Kurtosis**

The goal is to enable fast, reliable filtering of collision data streams, potentially applicable to particle detectors like those at CERN.

## Dataset

Link to the dataset: [CERN's Zenodo Dataset](https://zenodo.org/records/2603256#.Y20xysvMLmE)  
The dataset consists of particle data(Energy, Momentum_X, Momentum_Y, Momentum_z) for 200 constituents per Jet (if fewer were recorded, the remaining columns are 0 padded)


### To use the dataset:

Use this code to extract data into parquet files and compress into gzip:

```python
import pandas as pd
df = pd.read_hdf('train.h5', 'table').sample(n = 90000)   # Change the number of samples if required
df.to_parquet('jets90000.parquet.gzip', compression = 'gzip') 

```
Use the generated file in `gen_dataset.m` and the images will be generated and saved for training, testing and validation.

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

