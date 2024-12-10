## Project 5

**Fun with Diffusion Models**&emsp;[Webpage](https://davidpaulwei.github.io/cs180/proj5/) | [My Code](https://github.com/davidpaulwei/cs180/tree/main/proj5/code)  
&emsp;    
Project 5 consists two parts: <b>The power of Diffusion Models</b> (Part A) and <b>Diffusion Models from Scratch</b> (Part B).    
&emsp;   
In Part A, I mainly played around with a pretrained diffusion model called <b>DeepFloyd IF</b>. First, I used the model to conduct <b>denoising</b>. I blurred a sample image using random noise, and used the model to predict that noise. I also denoised the image using <b>Gaussian Blur</b> and compared the denoised results. Then, I denoised a random noise image to obtain a computer-generated image. I adapted the <b>Classifier-Free Guidance</b> technic. Later, I conducted image-to-image translation, where images are translated into similar images either based on masks or text prompts. At last, I produced <b>Visual Anagrams</b>, <b>Hybrid Images</b>, and a course logo.    
&emsp;    

In Part B, I built and trained a diffusion model from scratch. First, I trained a UNet to denoise half-noisy <b>MNIST</b> images (original image + 50% pure noise). Then, to denoise images with different amount of noise, I added <b>Time Conditioning</b> to the UNet, where the UNet is told how noisy each images are. The trained UNet can accurately predict the noise that had been added to the images. Using the trained UNet denoiser, I generated MNIST-like images by denoising pure noise in 300 steps, only to find the computer generated images looks little like human-written numbers. To boost the result, I added <b>Class Conditioning</b> to the UNet, where the UNet is not only told how noisy the images are, but also the labels (0 to 9) of the images. 10% of the images are not provided with a label. I adapted <b>Classifier-Free Guidance</b> to generate MNIST-like images. The results are much better compared the previous attempt.

Part A finished on Nov 6, 2024;   
Part B finished on Nov 16, 2024.
