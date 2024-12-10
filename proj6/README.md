## Final Project

**Neural Radiance Field (NeRF)**&emsp;[Webpage](https://davidpaulwei.github.io/cs180/proj6/) | [My Code](https://github.com/davidpaulwei/cs180/tree/main/proj6/code)  
&emsp;    
My final project consists two parts: **Fit a Neural Field to a 2D Image** (Part A), and **Fit a Neural Radiance Field from Multi-view Images** (Part B).      
&emsp;   
In Part A, I built a **Multiplayer Perceptron (MLP) network** to fit a single 2D image so that given any pixel's coordinate, the network can predict the pixel's RGB color. When the image's shape is provided, the network can **reconstruct the whole image**. In Part B, I trained a larger MLP network, or a **Neural Radiance Field (NeRF)**, to fit a 3D _Lego_ object through inverse rendering from **multi-view calibrated images**. The pixels on the images were bounded with rays represented in 3D world coordinate system. Sample locations were gathered along the rays, and their volume rendering results were used to fit the RGB colors on the images' pixels. In this way the _Lego_ object was modeled into the NeRF. Using the trained NeRF, I'm able to predict the images of the _Lego_ taken from **any given perspectives**. I rendered these images into a video to create a rotating effect of the _Lego_.    
&emsp;    

Finished on Dec 9, 2024.
