# Top Quark Classification using Deep Learning

A deep CNN based solution for real-time classification of particle jets into **standard background** or **top quark**, using custom convolutional blocks in **MATLAB Deep Network Designer**.

## Project Overview

This project applies modern deep learning techniques to high-energy particle data to solve the classification problem. The network processes **multi-channel jet images**, which essentially represent various statisticaal metrics along with **global jet-level features** like: Energy Skewness, pT Skewness, Energy Kurtosis, pT Kurtosis. The multiple pixel-level channels are described in great detail in the `literature` folder.

The goal is to enable fast, reliable filtering of collision data streams, potentially applicable to particle detectors like the LHC pr maybe the high luminosity at CERN .

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
The network utilizes **aggregated residual transformations** coupled with **Squeeze and Excite (SE)** Blocks. 
Using Aggregated Residual transformations over the standard residual additions has been proven much effective due to the use of **grouped convolution** layers, where channels are divided into different convolution blocks and each block extracts featues better than one block extracting over all the channels. 
Another issue observed in almost all Convolutional Networks is as the depth increases, usually the spatial dimensions of the image decreases and the number of channels increases. This calls the need for channel-level attention blocks, which calculate the "importance" of each channel in deciding the final output and assign weights to each channel.  
The main block of the architecture is the 
![MATLAB AI - Frame 1](https://github.com/user-attachments/assets/a1c00e79-5b14-4fb9-9662-a46f8dd4989e)


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


## Folder Structure
├── trainedModel90k.mat # Final trained network  
├── checkpoints/ # Intermediate training checkpoints
├── trainData60k.mat # Training dataset
├── valData.mat # Validation dataset
├── networkDesign.png # Model architecture diagram (optional)
├── dlnet_train.m # Custom training script (if applicable)
├── README.md # Project documentation
