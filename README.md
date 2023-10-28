# MCNMF-Unet
MCNMF-Unet: A Mixture Conv-MLP Network with Multi-scale Features Fusion U-Net for Medical Image Segmentation
This repository contains code for a image segmentation model based on MCNMF-Unet: A Mixture Conv-MLP Network with Multi-scale Features Fusion U-Net for Medical Image Segmentation implemented in PyTorch.




## Requirements
- PyTorch 1.x 

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.8 anaconda
conda activate <env_name>
```

2. Install pip packages.
```sh
Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

```

##  Prepare Datasets
Dataset name: Breast Ultra Sound Images (BUSI). URL:https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset. 
Dataset name: CVC_ClinicDB. URL: https://www.kaggle.com/datasets/balraj98/cvcclinicdb. 
Dataset name: International Skin Imaging Collaboration (ISIC 2018). URL: https://www.isic-archive.com/
```
inputs
└── CVC_ClinicDB
    ├── images
    |   ├── 1.png
    |   ├── 2.png             
    │   ├── ...
    |
    ├── masks
    |    ├──  0
    |       ├── 1.png
    |       ├── 2.png  
    │       ├── ...
```
##  Train
```
Train the model.

python train.py --dataset <dataset name> --arch MCNMFUnet
```
##  Test
```sh
python val.py --name CVC_ClinicDB_MCNMFUnet_woDS
```

