## CS 180 Project 4 Code

**Stitching Photo Mosaics**&emsp;[Webpage](https://davidpaulwei.github.io/cs180/proj4/)

`main.py` contains the necessary code for this project, run it within this folder. `animator.py` contains an _animator_ class that helps visualizing the gradient descent process. The following libraries are required:

1. `numpy`
2. `pickle`
3. `matplotlib.pyplot`
4. `skimage`
5. `torch`
6. `scipy.interpolate`
7. `scipy.ndimage`

Here is a brief introduction of the functions included in `main.py`:

1. `get_points()`: Used to manually select correspondences of image. Since the correspondences are already selected and stored in `xxx_pt.pickle` files, this function is not called.
2. `read_points()`: Read recorded correspondences from file.
3. `show_points()`: Show image and its correspondences.
4. `snap()`: Snap values within range.
5.  `perspective_trans()`: Use given matrix to conduct perspective transformation on pixels.
6. `compute_mat()`: Compute perspective transformation matrix using given correspondences. When number of correspondences on each image exceeds 4, an additional gradient descent will be applied to adjust the matrix.
7. `warp_img()`: Warp image using perspective transformation.
8. `alpha_blend()`: Blend the overlapping region of images using alpha blend. The alpha value for each pixels is obtained by distance transofrmation.
10. `mosaic()`: Blends two images into a mosaic image according to their correspondences.
11. `rectification()`: Rectificate image according to its correspondences. Idealy, the rectangular object in the image will be streched into a rectangle after transformation.""
    
    
Functions `get_points()`, `read_points()`, `show_points()`, `snap()` are mostly or entirely from [Project 3](https://github.com/davidpaulwei/cs180/tree/main/proj3/code).

Finished on Oct 14, 2024.
