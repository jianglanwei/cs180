## CS 180 Final Project Code

**Neural Radiance Field**&emsp;[Webpage](https://davidpaulwei.github.io/cs180/proj6/)

Run Part A in `neural_field_2d.py`, run Part B in `NeRF.py`, run them within this folder. Set `TRAINING_MODE` as `True` to train MLP net, as `False` to load trained neural field and generate images. `animator.py` contains an animator class that virtualizes the MSE/PSNR curve. For both Part A and B, GPU with CUDA compatibility are required.    

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

**Introduction of the functions and class included in `neural_field_2d.py`:**  
1. `sinusoidal_position(pos: torch.Tensor, L: int) -> torch.Tensor`: Conduct Sinusoidal Positional Encoding (PE) on given position.
2. `psnr(mse: torch.Tensor) -> torch.Tensor`: Calculate the Peak Signal-to-Noise Ratio (PSNR) given 
the input Mean Square Error (MSE).   
3. `img_iter`: An image iterator class that randomly iterates through an image.
    - `__init__(self, img: array_like, batch_size: int, num_steps: int)`: Initilize class.
    - `__iter__(self)`: Returns self as generator.
    - `__next__(self)`: Returns sinusoidal positional encoded pixel coordinates and their corresponding rgb color.
  
**Introduction of the functions and classes included in `NeRF.py`:**  
1. `camera2world(x: np.ndarray, c2ws: np.ndarray) -> np.ndarray`: Transform 3D coordinates in camera view (x) to world view (out) using transformation matrix T.
2. `pixel2camera(uv: np.ndarray, K: np.ndarray, s: np.ndarray) -> np.ndarray`: Map pixel coordinates (uv) to 3D camera-view coordinates (out) using intrinsic matrix K.
3. `pixel2ray(uv: np.ndarray, K: np.ndarray, c2ws: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`: Map pixel coordinates (uv) to rays with origin and normalized direction.
4. `ray_samples(ray_ori: np.ndarray, ray_dir_norm: np.ndarray, near: float, far: float, num_samples_per_ray: int, perturbation_width: float = 0) -> np.ndarray`: Sample 3D points along a given ray.
5. `vol_rend(rgbs: torch.Tensor, densities: torch.Tensor, step_size: float, background_rgb: torch.Tensor = torch.zeros(3)) -> torch.Tensor`: Volume rendering for NeRF.
6. `sinusoidal_position(x: torch.Tensor, L: int) -> torch.Tensor`: Conduct sinusoidal positional encoding (PE) on x.
7. `psnr(mse: torch.Tensor) -> float`: Calculate the Peak Signal-to-Noise Ratio (PSNR) given the input Mean Square Error (MSE).
8. `to_gpu(gpu: torch.device, *objs) -> Tuple`: Converts iterable objects into Tensor and move them to selected GPU.   
9. `nerf_iter`: A sample iterator class that samples random rays from image batch, and fetch sample locations on those rays.
    - `__init__(self, imgs: np.ndarray, c2ws: np.ndarray, focal: int, batch_size: int, num_steps: int, near: float, far: float, num_samples_per_ray: int, perturbation_width: float = 0.)`: Initilize class.
    - `__iter__(self)`: Returns self as generator.
    - `__next__(self)`: Fetch samples on random rays.
    - `find_sample_range(self)`: Find the maximum and minimum world-view coordinates on each axis. They are used to normalize the input 3D world-view coordinates.
10. `nerf_net(nn.Module)`: Multilayer Perceptron (MLP) network desgned for NeRF task.
    - `__init__(self, num_hidden: int)`: Initilize class.
    - `forward(self, sample_coords: torch.Tensor, ray_dir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`: Forward function for NeRF net.
    
    
**Acknowledgement**   
Both parts are run on [Berkeley Hybrid Robotics](https://hybrid-robotics.berkeley.edu)'s server with four NVIDIA RTX A4500.

Finished on Dec 9, 2024.
