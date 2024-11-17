This folder contains trained unconditional UNet denoiser after 1 and 5 training epochs.    
**Input:** Noisy image with size [N, 1, 28, 28];     
**Output:** Denoised image with size [N, 1, 28, 28].    

_P.S. Unfortunately the .pth file storing the trained params of the UNet denoiser is too large to upload to Github. Run `unconditioned_unet.py` in training mode to train the UNet. The UNet will be stored in this path._
