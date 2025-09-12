## CS 180 Project 5 Code

**Fun with Diffusion Models**&emsp;[Webpage](https://jianglanwei.github.io/cs180/proj5/)

Code for Part A is stored in `proj5b.ipynb`, it is based on the [starter code](https://colab.research.google.com/drive/1RiLmbbnhr4YsogD3dWZ39asPVNKuPsah?usp=sharing) given by course staff. For Part B, implementation code for _Unconditioned UNet_, _Time-Conditioned UNet_, _Class-Conditioned UNet_ are stored in `unconditioned_unet.py`, `time_conditioned_unet.py`, `class_conditioned_unet.py`, respectively. They can be run separately. Set `TRAINING_MODE` as `True` to train UNet, as `False` to load trained UNet to denoise or generate MNIST images. The net modules are stored in `module.py`. `animator.py` contains an animator class that virtualizes the training loss curve. For both Part A and B, GPU with CUDA compatibility are required.    

The following libraries are required to run Part B:    

1. `torch`
2. `torch.nn`
3. `torch.utils.data`
4. `torchvision.transforms`
5. `torchvision.datasets`
6. `torchsummary`
7. `matplotlib.pyplot`


**Introduction of the functions included in Part B:**   

1. `load_data()`: Load data from MNIST dataset.
2. `add_noise`: Add noise to image.
3. `ddpm_schedule()`: Compute constants for DDPM training and sampling.
4. `init_weights()`: Initialize net weights.
5.  `one_hot_vec()`: Generate One-hot vectors for network input.
6. `spread_imgs()`: spread 4d image tensor into a 2d image for better virtualization.
    
**Acknowledgement**   
Part A is run on [Google Colab](https://colab.research.google.com/)'s NVIDIA Tesla T4, while part B is run on [Berkeley Hybrid Robotics](https://hybrid-robotics.berkeley.edu)'s NVIDIA RTX A4500.

Finished on Nov 16, 2024.
