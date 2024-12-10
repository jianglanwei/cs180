## CS 180 Final Project Code

**Neural Radiance Field**&emsp;[Webpage](https://davidpaulwei.github.io/cs180/proj6/)

Run Part A in `neural_field_2d.py`, run Part B in `NeRF.py`. Set `TRAINING_MODE` as `True` to train MLP net, as `False` to load trained neural field and generate images. `animator.py` contains an animator class that virtualizes the MSE/PSNR curve. For both Part A and B, GPU with CUDA compatibility are required.    

The following libraries are required to run `neural_field_2d.py`:
1. `torch`
2. `torch.nn`
3. `torchsummary`
4. `matplotlib.pyplot`

The following libraries are required to run `NeRF.py`:    
1. `torch`
2. `torch.nn`
3. `numpy`
4. `torchsummary`
5. `matplotlib.pyplot`

**Introduction of the functions and class included in Part A:**  
**Functions:**
1. `sinusoidal_position(pos: torch.Tensor, L: int) -> torch.Tensor`: Conduct Sinusoidal Positional Encoding (PE) on given position.
2. `psnr(mse: torch.Tensor) -> torch.Tensor`: Calculate the Peak Signal-to-Noise Ratio (PSNR) given 
the input Mean Square Error (MSE).
**Class:**
3. `img_iter`: An image iterator class that randomly iterates through an image.
    - `__init__(self, img, batch_size: int, num_steps: int)`: Initilize class.
    - `__iter__(self)`: Returns self as generator.
    - `__next__(self)`: Returns sinusoidal positional encoded pixel coordinates and their corresponding rgb color.


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

