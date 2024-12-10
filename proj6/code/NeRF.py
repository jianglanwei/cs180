import numpy as np
import torch
from torch import nn
from animator import animator
import torchsummary
from typing import Tuple
import matplotlib.pyplot as plt

ROOT_DIR = ""
NET_DIR = "nerf_nets/"
GEN_IMG_DIR = "pred_3d_objs/"

SAMPLE_NEAR = 2.
SAMPLE_FAR = 6.
NUM_SAMPLES_PER_RAY = 32
PERTURBATION_WIDTH = .02

NUM_INPUT_FREQ_LEVELS = 10
NUM_HIDDEN = 256
BATCH_SIZE = 10_000
LR = 5e-4
NUM_STEPS = 2000

TRAIN_MODE = True
SAVE_FREQ = 50
GEN_IMG_BACKGROUND_RGB = torch.tensor([0, 0, 1])

GPU_NO = 0

data = np.load(ROOT_DIR + "lego_200x200.npz")

# Training images: [100, 200, 200, 3]
train_imgs = data["images_train"] / 255.0

# Cameras for the training images 
# (camera-to-world transformation matrix): [100, 4, 4]
train_c2ws = data["c2ws_train"]

# Validation images: 
val_imgs = data["images_val"] / 255.0

# Cameras for the validation images: [10, 4, 4]
# (camera-to-world transformation matrix): [10, 200, 200, 3]
val_c2ws = data["c2ws_val"]

# Test cameras for novel-view video rendering: 
# (camera-to-world transformation matrix): [60, 4, 4]
test_c2ws = data["c2ws_test"]

# Camera focal length
focal = data["focal"]  # float


def camera2world(x: np.ndarray, c2ws: np.ndarray) -> np.ndarray:
    """transform 3d coordinates in camera view (x) to world view (out)
       using transformation matrix T.
       inputs:
       - x: 3d camera coordinates with shape: [N, 3];
       - c2ws: camera to world transformation matrix with shape [N, 4, 4].
       output: 3d world coordinates with shape [N, 3]."""
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)[:,np.newaxis,:]
    out = x @ np.swapaxes(c2ws, 1, 2)
    return out[:, 0, :3]

def pixel2camera(uv: np.ndarray, K: np.ndarray, s: np.ndarray) -> np.ndarray:
    """map pixel coordinates (uv) to 3d camera-view coordinates (out)
       using intrinsic matrix K.
       inputs:
       - uv: 2d pixel coordinates with shape [N, 2];
       - K: intrinsic matrix with shape [3, 3];
       - s: depth of point along the optical axis with shape [N].
       output: 3d camera coordinates with shape [N, 3]."""
    uv = np.concatenate((uv, np.ones((len(uv), 1))), axis=1)
    out = (uv * s[:,np.newaxis]) @ np.linalg.inv(K).T
    return out

def pixel2ray(uv: np.ndarray, K: np.ndarray, c2ws: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    """map pixel coordinates (uv) to rays with origin and normalized direction.
       inputs:
       - uv: 2d pixel coordinates with shape [N, 2];
       - K: intrinsic matrix with shape [3, 3];
       - c2ws: camera to world transformation matrix with shape [N, 4, 4].
       outputs: 
       - ray origin with shape [N, 3]; 
       - normalized ray direction with shape [N, 3]."""
    ray_ori = c2ws[:, :3, 3]
    camera_coords_3d = pixel2camera(uv, K, np.ones(len(uv)))
    world_coords_3d = camera2world(camera_coords_3d, c2ws)
    ray_dir = world_coords_3d - ray_ori
    ray_dir_norm = ray_dir / np.linalg.norm(ray_dir, axis=1)[:,np.newaxis]
    return ray_ori, ray_dir_norm

class nerf_iter:
    """a sample iterator class that samples random rays from image batch,
       and fetch sample locations on those rays."""
    def __init__(self, imgs: np.ndarray, c2ws: np.ndarray, focal: int, 
                 batch_size: int, num_steps: int, near: float, far: float,
                 num_samples_per_ray: int, perturbation_width: float = 0.):
        """initialize sample iterator class.
           inputs:
           - imgs: all images with shape [num_imgs, img_height, img_width];
           - c2ws: camera to world transformation matrix with shape [num_imgs, 4, 4];
           - focal: camera focal length;
           - batch size: number of rays sampled in each iteration;
           - num_steps: total number of steps/iterations;
           - near: distance of nearest sample to ray origin;
           - far: distance of furthest sample to ray origin; 
           - num_samples_per_ray: number of samples obtained on each ray;
           - perturbation_width: range of perturbation added to samples' coordinates."""
        self.imgs = imgs
        self.img_c2ws = c2ws
        self.num_steps = num_steps
        self.current_step = 0
        self.batch_size = batch_size
        self.num_imgs = imgs.shape[0]
        self.img_shape = imgs.shape[1:]
        self.near = near
        self.far = far
        self.num_samples_per_ray = num_samples_per_ray
        self.perturbation_width = perturbation_width
        self.K = np.array([[focal, 0,     imgs.shape[2]/2],
                           [0,     focal, imgs.shape[1]/2],
                           [0,     0,     1              ]])
        self.world_range_min, self.world_range_max = self.find_sample_range()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """fetch samples on random rays.
           outputs:
           - 3d samples' coordinates with shape [N, num_samples_per_ray, 3];
           - normalized ray direction with shape [N, 3];
           - rgb color values with shape [N, 1, 3]."""
        if self.current_step == self.num_steps: raise StopIteration
        rand_imgs = np.random.randint(0, self.num_imgs, (self.batch_size,))
        rand_rows = np.random.randint(0, self.img_shape[0], (self.batch_size,))
        rand_cols = np.random.randint(0, self.img_shape[1], (self.batch_size,))
        rgbs = self.imgs[rand_imgs, rand_rows, rand_cols].squeeze()
        c2ws = self.img_c2ws[rand_imgs]
        uv = np.stack((rand_cols, rand_rows), axis=1)
        ray_ori, ray_dir_norm = pixel2ray(uv+0.5, self.K, c2ws)
        sample_coords = ray_samples(ray_ori, ray_dir_norm, self.near, self.far,
                              self.num_samples_per_ray, self.perturbation_width)
        sample_coords = (sample_coords - self.world_range_min) \
                      / (self.world_range_max - self.world_range_min)
        self.current_step += 1
        return sample_coords, ray_dir_norm[:,np.newaxis,:], rgbs
    
    def find_sample_range(self):
        """find the maximum and minimum world-view coordinates on each axis."""
        img_idxs, row_idxs, col_idxs = np.indices((self.num_imgs, 2, 2))
        img_idxs = img_idxs.flatten()
        row_idxs = (row_idxs * (self.img_shape[0] - 1)).flatten()
        col_idxs = (col_idxs * (self.img_shape[1] - 1)).flatten()
        c2ws = self.img_c2ws[img_idxs]
        uv = np.stack((col_idxs, row_idxs), axis=1)
        ray_ori, ray_dir_norm = pixel2ray(uv+0.5, self.K, c2ws)
        sample_coords = ray_samples(ray_ori, ray_dir_norm, self.near, self.far,
                              self.num_samples_per_ray)
        sample_range = (np.array([sample_coords[:,:,i].min() for i in range(3)]),
                        np.array([sample_coords[:,:,i].max() for i in range(3)]))
        return sample_range

def ray_samples(ray_ori: np.ndarray, ray_dir_norm: np.ndarray,
                near: float, far: float, num_samples_per_ray: int, 
                perturbation_width: float = 0) -> np.ndarray:
    """sample 3d points along a given ray.
       inputs:
       - ray_ori: origin of ray with shape [N, 3];
       - ray_dir_norm: normalized direction of ray with shape [N, 3];
       - near: distance of nearest sample to ray origin;
       - far: distance of furthest sample to ray origin;
       - num_samples_per_ray: number of samples obtained on each ray;
       - perturbation_width: range of perturbation added to samples' coordinates.
       output: 3d samples' coordinates with shape [N, num_samples_per_ray, 3]."""
    distance_to_origin = np.linspace(near, far, num_samples_per_ray)[:,np.newaxis]
    perturbation = (np.random.rand(num_samples_per_ray, 1) - 1/2) * perturbation_width
    distance_to_origin += perturbation
    ray_ori, ray_dir_norm = ray_ori[:,np.newaxis], ray_dir_norm[:,np.newaxis]
    samples = ray_ori + ray_dir_norm * distance_to_origin
    return samples

def vol_rend(rgbs: torch.Tensor, densities: torch.Tensor, step_size: float,
             background_rgb: torch.Tensor = torch.zeros(3)) -> torch.Tensor:
    """volume rendering for nerf.
       inputs:
       - rgbs: rgb color at the sample points with shape [N, num_samples_per_ray, 3];
       - density: color densities (sigmas) with shape [N, num_samples_per_ray, 1];
       - step size: length between sample points on the ray.
       - background_rgb: background color of the rendered image with shape [3].
       output: rendered color with shape [N, 3]."""
    terminate_prob = 1 - torch.exp(-densities * step_size)
    arrive_prob = torch.exp(-torch.cumsum(densities * step_size, dim=1))
    arrive_background_prob = arrive_prob[:,-1]
    arrive_prob = torch.cat((torch.ones(len(rgbs),1,1).to(arrive_prob.device), 
                             arrive_prob[:,:-1]), dim=1)
    color_gain = rgbs * terminate_prob * arrive_prob
    background_rgb = background_rgb.to(rgbs.device)
    rendered_color = color_gain.sum(dim=1) + background_rgb * arrive_background_prob
    return rendered_color

def sinusoidal_position(x: torch.Tensor, L: int) -> torch.Tensor:
    """conduct sinusoidal positional encoding (pe) on x.
       inputs:
       - x: object with shape [N, num_samples_per_ray, D].
       - L: highest frequency level.
       output: position encoded 'x' (pe_x) with shape
               [N, num_samples_per_ray, (2 * L + 1) * D]."""
    pe_shape = (x.shape[0], x.shape[1], (2 * L + 1), x.shape[2])
    pe_x = torch.zeros(pe_shape).to(x.device)
    pe_x[:,:,0] = x
    weights = 2 ** torch.arange(L).reshape(1, 1, -1, 1).to(x.device)
    pe_x[:,:,1::2] = torch.sin(weights * torch.pi * x[:,:,torch.newaxis,:])
    pe_x[:,:,2::2] = torch.cos(weights * torch.pi * x[:,:,torch.newaxis,:])
    pe_x = pe_x.reshape(x.shape[0], x.shape[1], -1)
    return pe_x

def psnr(mse: torch.Tensor) -> float:
    """calculate the Peak Signal-to-Noise Ratio (psnr) given 
       the input Mean Square Error (mse).
       input: mse, Mean Square Error with shape [1].
       output: psnr."""
    return (-10 * torch.log10(mse)).item()


class nerf_net(nn.Module):
    """multilayer perceptron (mlp) network designed for nerf task."""
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.pe = sinusoidal_position
        pe_len = 6 * NUM_INPUT_FREQ_LEVELS + 3
        self.lin1 = nn.Linear(pe_len, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_hidden)
        self.lin4 = nn.Linear(num_hidden, num_hidden)
        self.lin5 = nn.Linear((num_hidden + pe_len), num_hidden)
        self.lin6 = nn.Linear(num_hidden, num_hidden)
        self.lin7 = nn.Linear(num_hidden, num_hidden)
        self.lin8 = nn.Linear(num_hidden, num_hidden)
        self.lin9 = nn.Linear(num_hidden, 1)
        self.lin10 = nn.Linear(num_hidden, num_hidden)
        self.lin11 = nn.Linear((num_hidden + pe_len), (num_hidden // 2))
        self.lin12 = nn.Linear((num_hidden // 2), 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, sample_coords: torch.Tensor, ray_dir: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward function for nerf net.
           inputs: 
           - 3d coordinates of sample points with shape [N, num_samples_per_ray, 3],
             the coordinates need to be normalized to approximately [0, 1];
           - ray direction with shape [N, 1, 3].
           outputs:
           - predicted color density with shape [N, num_samples_per_ray, 1];
           - predicted rgb color with shape [N, num_samples_per_ray, 3]."""
        ray_dir = ray_dir.repeat(1, sample_coords.shape[1], 1)
        pe_pos = self.pe(sample_coords, NUM_INPUT_FREQ_LEVELS)
        pe_dir = self.pe(ray_dir, NUM_INPUT_FREQ_LEVELS)
        out = self.lin1(pe_pos)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.lin3(out)
        out = self.relu(out)
        out = self.lin4(out)
        out = self.relu(out)
        out = torch.cat((out, pe_pos), dim=2)
        out = self.lin5(out)
        out = self.relu(out)
        out = self.lin6(out)
        out = self.relu(out)
        out = self.lin7(out)
        out = self.relu(out)
        out = self.lin8(out)
        out_den = self.lin9(out)
        out_den = self.relu(out_den)
        out_rgb = self.lin10(out)
        out_rgb = torch.cat((out_rgb, pe_dir), dim=2)
        out_rgb = self.lin11(out_rgb)
        out_rgb = self.relu(out_rgb)
        out_rgb = self.lin12(out_rgb)
        out_rgb = self.sigmoid(out_rgb)
        return out_den, out_rgb


def to_gpu(gpu: torch.device, *objs) -> Tuple:
    """converts iterable objects into Tensor and move them to selected gpu."""
    objs = tuple(torch.tensor(obj, dtype=torch.float).to(gpu) for obj in objs)
    return objs


gpu = torch.device(f'cuda:{GPU_NO}')
if TRAIN_MODE:
    net = nerf_net(NUM_HIDDEN).to(gpu)
    train_iter = nerf_iter(train_imgs, train_c2ws, focal, BATCH_SIZE, NUM_STEPS,
                        SAMPLE_NEAR, SAMPLE_FAR, NUM_SAMPLES_PER_RAY, 
                        PERTURBATION_WIDTH)
    loss_func = torch.nn.MSELoss().to(gpu)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR) # type: ignore
    grf = animator(xlabel="Step", y1label="MSE", y2label="PSNR")
    step_size = (SAMPLE_FAR - SAMPLE_NEAR) / NUM_SAMPLES_PER_RAY
    torchsummary.summary(net, input_size=[(NUM_SAMPLES_PER_RAY, 3), (1, 3)])
    for step, (sample_coords, ray_dir, rgbs) in enumerate(train_iter):
        sample_coords, ray_dir, real_rgbs = to_gpu(gpu, sample_coords, ray_dir, rgbs)
        optimizer.zero_grad()
        pred_densities, pred_rgbs = net(sample_coords, ray_dir)
        rendered_rgbs = vol_rend(pred_rgbs, pred_densities, step_size)
        loss = loss_func(rendered_rgbs, real_rgbs).mean()
        loss.backward()
        optimizer.step()
        psnr_ = psnr(loss)
        grf.add(step, loss.item(), psnr_)
        grf.save(ROOT_DIR + "nerf_loss_graph.pdf")
        print(f"Step {step}: MSE: {loss.item(): .4f}; PSNR: {psnr_: .4f}.")
        if step % SAVE_FREQ == 0:
            torch.save(net, NET_DIR + f"net{step}.pth")
            torch.save(grf, ROOT_DIR + "nerf_loss_grf.pth")
else:
    MODEL_NO = 1950 # which trained net to use.
    with torch.no_grad():
        net = torch.load(NET_DIR + f"net{MODEL_NO}.pth")
        train_iter = nerf_iter(train_imgs, train_c2ws, focal, BATCH_SIZE, NUM_STEPS,
                            SAMPLE_NEAR, SAMPLE_FAR, NUM_SAMPLES_PER_RAY, 
                            PERTURBATION_WIDTH)
        step_size = (SAMPLE_FAR - SAMPLE_NEAR) / NUM_SAMPLES_PER_RAY
        for img_idx, c2w in enumerate(test_c2ws):
            pred_img = np.zeros(train_iter.img_shape)
            img_w, img_h = train_iter.img_shape[:2]
            c2ws = np.tile(c2w, (img_w, 1, 1))
            col_idxs = np.arange(img_w)
            for row in range(img_h): # predict image by row due to gpu memory limit.
                row_idxs = row * np.ones(img_w)
                uv = np.stack((col_idxs, row_idxs), axis=1)
                ray_ori, ray_dir = pixel2ray(uv+0.5, train_iter.K, c2ws)
                sample_coords = ray_samples(ray_ori, ray_dir, SAMPLE_NEAR, 
                                            SAMPLE_FAR, NUM_SAMPLES_PER_RAY)
                sample_coords = (sample_coords - train_iter.world_range_min) \
                        / (train_iter.world_range_max - train_iter.world_range_min)
                sample_coords, ray_dir = to_gpu(gpu, sample_coords, ray_dir)
                pred_densities, pred_rgbs = net(sample_coords, ray_dir.unsqueeze(1))
                rendered_rgbs = vol_rend(pred_rgbs, pred_densities, 
                                         step_size, GEN_IMG_BACKGROUND_RGB)
                pred_img[row] = rendered_rgbs.detach().cpu().numpy()
            plt.imsave(GEN_IMG_DIR + f"img{img_idx}.png", pred_img)
            print(f"image {img_idx} generated.")
