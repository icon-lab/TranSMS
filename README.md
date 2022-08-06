# TranSMS
Official Implementation of Transformers for System Matrix Super-resolution (TranSMS)

A. Güngör, B. Askin, D. A. Soydan, E. U. Saritas, C. B. Top and T. Çukur, "TranSMS: Transformers for Super-Resolution Calibration in Magnetic Particle Imaging," in IEEE Transactions on Medical Imaging, 2022, doi: 10.1109/TMI.2022.3189693.

# Demo
You can use the following links to download training, validation, test datasets. 

# Dataset
https://drive.google.com/drive/folders/1Zru6u-GmxPUktAFB41sCIEQEgZEwHKcf?usp=sharing

# Pretrained Networks:
https://drive.google.com/drive/folders/1jJs-9kDkzHTpVsoGSgdYjzihXPt9Lkef?usp=sharing

# Training

Generic training code code:

```python checkTranSMSAselFFLTrain.py --useGPUno 0 --wd 0 --lr 1e-4 --scale_factor 2 --snrThreshold 5 --useNoisyProjection 1 --bs 64 --resultFolder . --n1 32 --n2 32 --trainFolder ./train --testFolder ./val```

useGPUno: Selected GPU
wd: weight decay, default is 0
lr: learning rate
scale_factor: 2, 4, 8, etc.
snrThreshold: SNR threshold for SM training, i.e. values below threshold are not used for training
useNoisyProjection: 0 ablates the data consistency block from TranSMS, 1 is regular TranSMS with data consistency
bs: batch size
resultFolder: path for saving model outputs
n1: SM dimension x
n2: SM dimension y
trainFolder: folder containing training SMs
testFolder: folder containing validation SMs

# Code for Open MPI dataset

Code for 2x, 4x and 8x training using Open MPI dataset:

```python checkTranSMSAselFFLTrain.py --useGPUno 0 --lr 5e-4 --scale_factor 2 --resultFolder . --trainFolder ./train --testFolder ./val```

```python checkTranSMSAselFFLTrain.py --useGPUno 0 --lr 1e-4 --scale_factor 4 --resultFolder . --trainFolder ./train --testFolder ./val```

```python checkTranSMSAselFFLTrain.py --useGPUno 0 --lr 5e-5 --scale_factor 8 --resultFolder . --trainFolder ./train --testFolder ./val```

# Inference

Code for inference using all trained networks:

```python inferenceOnOpenMPI.py --useGPUno 0 --bs 256 --n1 32 --n2 32 --modelFolder ./outs/ --saveOutFolder ./results/ --testFolder ./test --interpolationMatrixPath interpolaters.mat```

useGPUno: Selected GPU
bs: batch size during inference
n1: SM dimension x
n2: SM dimension y
modelFolder: folder containing trained networks
saveOutFolder: path for saving "mat" file outputs
testFolder: folder containing test SMs
interpolationMatrixPath: path containing the interpolation matrix from 4x4, 8x8, 16x16 to 32x32, for fast interpolation purposes

**************************************************************************************************************************************
# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{transms,
  author={Gungor, Alper and Askin, Baris and Soydan, Damla Alptekin and Saritas, Emine Ulku and Top, Can Baris; and Cukur, Tolga},
  journal={IEEE Transactions on Medical Imaging}, 
  title={TranSMS: Transformers for Super-Resolution Calibration in Magnetic Particle Imaging}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3189693}}
```
(c) ICON Lab 2022

# Prerequisites

- Python 3.6
- CuDNN 8.2.1
- PyTorch 1.10.0

# Acknowledgements

This code uses libraries from CvT, SRCNN and VDSR repositories.

For questions/comments please send an email to: alperg@ee.bilkent.edu.tr