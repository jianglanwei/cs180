This folder contains trained class-conditioned UNet denoiser after 5 and 20 training epochs.    
**Input:** Noisy image with size [N, 1, 28, 28]; Extent of noise _T_ ∈ [0, 1] with size [N, 1, 1, 1]; Image One-hot label with size [N, 1, 1, 10]    
**Output:** Predicted noise with size [N, 1, 28, 28].    

_P.S. Unfortunately the .pth file storing the trained params of the UNet denoiser is too large to be uploaded to Github. Run `class_conditioned_unet.py` in training mode to train the UNet. The UNet will be stored in this path._
