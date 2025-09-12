## CS 180 Project 4 Code

**Stitching Photo Mosaics**&emsp;[Webpage](https://jianglanwei.github.io/cs180/proj4/)

Run the project in `main.py`, run it within this folder. `proj4a_func.py` and `proj4b_func.py` contains the functions for **Project 4A** and **Project 4B**, respectively. The functions are called in `main.py`. `animator.py` contains an _animator_ class that helps visualizing the gradient descent process. The following libraries are required:

1. `numpy`
2. `torch`
3. `pickle`
4. `skimage`
5. `skimage.feature`
6. `matplotlib.pyplot`
7. `matplotlib.patches`
8. `scipy.interpolate`
9. `scipy.ndimage`

**Introduction of the functions included in `proj4a_func.py`:**   

1. `get_points()`: Used to manually select correspondences of image. Since the correspondences are already selected and stored in `xxx_pt.pickle` files, this function is not called.
2. `read_points()`: Read recorded correspondences from file.
3. `show_points()`: Show image and its correspondences.
4. `snap()`: Snap values within range.
5.  `perspective_trans()`: Use given matrix to conduct perspective transformation on pixels.
6. `compute_mat()`: Compute perspective transformation matrix using given correspondences. When number of correspondences on each image exceeds 4, an additional gradient descent will be applied to adjust the matrix.
7. `warp_img()`: Warp image using perspective transformation.
8. `alpha_blend()`: Blend the overlapping region of images using alpha blend. The alpha value for each pixels is obtained by distance transformation.
9. `mosaic()`: Blend two images into a mosaic image according to their correspondences.
10. `rectification()`: Rectificate image according to its correspondences. Idealy, the rectangular object in the image will be streched into a rectangle after transformation.

**Introduction of the functions included in `proj4b_func.py`:**   

11. `get_harris_corners`: Find harris corners in the image. 
12. `dist2()`: Calculate squared distance between two sets of points.
13. `anms()`: Conduct Adaptive Non-Maximum Suppression (ANMS) on interest points.
14. `match()`: Match potential correspondences on two images.
15. `ransac()`: Use Random Sample Consensus (RANSAC) to calculate transformation matrix.
16. `auto_mosaic()`: Blend image1 and image2 into a mosaic image automatically.
17. `warp_img()`: Warp image using perspective transformation.
18. `show_matches()`: Show potential correspondences and their matches.
    
**Acknowledgement**   
Functions `get_points()`, `read_points()`, `show_points()`, `snap()` are mostly or entirely from [Project 3](https://github.com/jianglanwei/cs180/tree/main/proj3/code); Functions `get_harris_corners()` and `dist2()` are adapted from sample code given by course staff.

Finished on Oct 24, 2024.
