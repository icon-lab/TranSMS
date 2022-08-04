# TranSMS
Official Implementation of Transformers for System Matrix Super-resolution (TranSMS)

A. Güngör, B. Askin, D. A. Soydan, E. U. Saritas, C. B. Top and T. Çukur, "TranSMS: Transformers for Super-Resolution Calibration in Magnetic Particle Imaging," in IEEE Transactions on Medical Imaging, 2022, doi: 10.1109/TMI.2022.3189693.

# Demo
You can use the following links to download training, validation, test datasets. 

Datasets: https://drive.google.com/drive/folders/1_n8JynaPRQcPmu4TwYF6x6zOFPqLYlx8?usp=sharing

Pretrained Networks: https://drive.google.com/drive/folders/15FiUVr7w3NmW92PFc-tQPJA_sfUta2aF?usp=sharing

Use below for singlecoil inference:
```python run_projector_kspace.py project-images --network_pkl pretrained_networks/FedGIMP-840-1gpu-withoutgrowing-singlecoil-cond/G-snapshot-100.pkl --tfr-dir datasets/TFRecords/singlecoil/IXI/test/ --h5-dir datasets/h5files/singlecoil/ --result_dir results/inference --case singlecoil --dataset IXI --us_case poisson --acc_rate 3 --num-images 210 --gpu 0```

Use below for multicoil inference:
```python run_projector_kspace.py project-images --network_pkl pretrained_networks/FedGIMP-864-1gpu-withoutgrowing-multicoil-cond/G-snapshot-100.pkl --tfr-dir datasets/TFRecords/multicoil/fastMRI_brain/test/ --h5-dir datasets/h5files/multicoil/ --result_dir results/inference --case multicoil --dataset fastMRI_brain --us_case poisson --acc_rate 3 --num-images 216 --gpu 0```

# Dataset
- ASELSAN dataset: https://drive.google.com/drive/folders/1_n8JynaPRQcPmu4TwYF6x6zOFPqLYlx8?usp=sharing
- OpenMPI dataset: https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/

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
- Tensorflow 2.5.0

# Acknowledgements

This code uses libraries from XXX repositories.

For questions/comments please send an email to: alpergungor@windowslive.com
