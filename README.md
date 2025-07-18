# Top Quark Classification using Deep Learning

A deep CNN-based solution for real-time classification of particle jets into **standard background** or **top quark**, using custom convolutional blocks in **MATLAB Deep Network Designer**.

## Project Overview

This project applies modern deep learning techniques to high-energy particle data to solve the classification problem. The network processes **multi-channel jet images**, which essentially represent various statistical metrics along with **global jet-level features** like: Energy Skewness, pT Skewness, Energy Kurtosis, pT Kurtosis. The multiple pixel-level channels are described in great detail in the `literature` folder.

The goal is to enable fast and reliable filtering of collision data streams, potentially applicable to particle detectors like the LHC or the high-luminosity upgrade at CERN.

[YouTube Video Link](https://youtu.be/JDQx7aJyvFw)

## Dataset

Link to the dataset: [CERN's Zenodo Dataset](https://zenodo.org/records/2603256#.Y20xysvMLmE)  
The dataset consists of particle data(Energy, Momentum_X, Momentum_Y, Momentum_z) for 200 constituents per Jet (if fewer were recorded, the remaining columns are 0 padded)


### To use the dataset:

Use this code to extract data into parquet files and compress them into gzip:

```python
import pandas as pd
df = pd.read_hdf('train.h5', 'table').sample(n = 90000)   # Change the number of samples if required
df.to_parquet('jets90000.parquet.gzip', compression = 'gzip') 

```
Use the generated file in `gen_dataset.m` and the images will be generated and saved for training, testing, and validation.

## Network Architecture
The network utilizes **aggregated residual transformations** coupled with **Squeeze and Excite (SE)** Blocks.  
Using Aggregated Residual transformations over the standard residual additions has been proven to be more effective due to the use of **grouped convolution** layers, where channels are divided into different convolution blocks and each block extracts features better than one block extracting over all the channels.  
Another issue observed in almost all Convolutional Networks is that as the depth increases, usually the spatial dimensions of the image decrease and the number of channels increase. This calls the need for channel-level attention blocks, which calculate the "importance" of each channel in deciding the final output and assign weights to each channel; hence, I used Squeeze-and-Excite blocks after major grouped convolution blocks to amplify important channels and reduce the impact of insignificant channels on the output.  

  
Link to the **complete** architecture of the network: [Architecture Flowchart on Miro](https://miro.com/app/board/uXjVJckxRRw=/?share_link_id=381620801824).  

  
One of the block of the architecture is shown, which contains a residual network connection, the Grouped Convolution Block, and the SE Block
![MATLAB AI - Frame 1](https://github.com/user-attachments/assets/a1c00e79-5b14-4fb9-9662-a46f8dd4989e)


The entire CNN is built using **MATLAB's Deep Network Designer** for fast prototyping and testing.

## Key Hyperparameters for training with 90k images

| Hyperparameter        | Value       | Reason                                         |
|------------------------|------------|------------------------------------------------|
| Initial Learn Rate     | 5e-3       | Reduces the risk of overshooting the optimal solution.            |
| Learn Rate Schedule    | Piecewise  | Allows controlled learning rate decay.         |
| Learn Rate Decay       | 0.3        | Multiplies learning rate by 0.3 at each decay step. |
| Learn Rate Drop Period | 3 epochs   | Reduces LR after every 5 epochs.               |
| L2 Regularization      | 1e-4       | Prevents overfitting by penalizing large weights. |
| Batch Size             | 64         | Balanced between training speed and VRAM limits. |
| Max Epochs             | 10         | Prevents overtraining; overfitting after ~14 epochs. |
| Optimizer              | Adam       | Efficient and adaptive learning optimization.  |

## Key Hyperparameters for training with 60k images

| Hyperparameter        | Value       | Reason                                         |
|------------------------|------------|------------------------------------------------|
| Initial Learn Rate     | 1e-2       | Reduces the risk of overshooting the optimal solution.            |
| Learn Rate Schedule    | Piecewise  | Allows controlled learning rate decay.         |
| Learn Rate Decay       | 0.5        | Multiplies learning rate by 0.3 at each decay step. |
| Learn Rate Drop Period | 5 epochs   | Reduces LR after every 5 epochs.               |
| Batch Size             | 64         | Balanced between training speed and VRAM limits. |
| Max Epochs             | 20         | Prevents overtraining; plateauing after ~17 epochs. |
| Optimizer              | Adam       | Efficient and adaptive learning optimization.  |

## Performance

- Achieved **90.33% testing accuracy** using 90k training samples.

  Training vs. Loss Curves
  <img width="1336" height="664" alt="acc_loss_90dot33_graph" src="https://github.com/user-attachments/assets/003b6bb8-1c7f-4c2d-8d5a-b83a8b03177f" />
  <img width="1523" height="805" alt="image" src="https://github.com/user-attachments/assets/7a7a4714-0cfb-40da-b432-ea397e643f09" />
  <img width="657" height="487" alt="confusion_matrix_90dot33_ac" src="https://github.com/user-attachments/assets/38147f02-f546-449f-a5de-405b74518548" />
  <img width="657" height="492" alt="roc_90dot33_acc" src="https://github.com/user-attachments/assets/6b8939c5-5fd6-4914-aca4-5b2ad86d4e7b" />


  
- Achieved **89.4% validation accuracy** using 60k training samples (Epochs were increased to 20 and Learning rate to 1e-2 with 0.5 Decay factor and drop period after every 5 epochs).

  Training vs. Loss Curves
  <img width="1536" height="794" alt="acc_loss_89dot40_graaph" src="https://github.com/user-attachments/assets/88138f3e-494e-4a58-89e5-95afc6bda04b" />
  <img width="1915" height="914" alt="confusion_matrix_test_89dot4_acc" src="https://github.com/user-attachments/assets/10f0852a-2315-4764-8a08-1e3f8d65ac2a" />


- Suitable for **real-time event filtering** in experimental setups.

## Deployment on FPGA
Use the HDL Code Generation process when HDL source files are needed for synthesis, simulation, or integration into custom FPGA designs. Use the FPGA Deployment process when deploying and running the trained model directly on supported FPGA boards using MATLAB’s Deep Learning HDL Toolbox.  

HDL code can be generated using `hdl_code_gen.m`  

The model can be deployed on FPGA using `deploy_on_fpga.m`  

Both these files can be found inside `deploy` folder
## Folder Structure 
<pre>
├── checkpoints/                      # Last 3 training checkpoints  
├── deploy/                           # Codes to generate HDL code and deploy on FPGA
    ├── deploy_on_fpga.m              # Code to deploy on FPGA
    ├── hdl_code_gen.m                # Code to generate the HDL code
├── image_generation                  # Contains files to generate images for training, validation, and testing  
    ├── align_img.m                   # function to rotate image for preprocessing  
    ├── gen_dataset.m                 # run this function with parquet files downloaded and in the same directory
    ├── mapping_approach.m            # creates images using my logic of mapping coordinates and performs preprocessing
    ├── visualize_averaged.m          # see how all the same kind of jets accumulated together would look like
    ├── visualize_jet.m               # see individual jets  
├── literature  
    ├── description.md                # contains a link for the architecture of the model in Miro
    ├── MATLAB AI Student Challenge Solution.pdf   # Contains the entire documentation about every steps taken and all plots for visualization
├── model
    ├── evaluate.m                    # run this to test your model
    ├── latest_lgraph.mlx             # script to generate a network in Deep Network Designer and analyze dimensions at every stage, run this before training
    ├── train_network.mlx             # script to train the network and choose hyperparameters
</pre>
