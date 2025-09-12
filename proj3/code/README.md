## CS 180 Project 3 Code

**Face Morphing and Modelling a Photo Collection**&emsp;[Webpage](https://jianglanwei.github.io/cs180/proj3/)

**main.py** contains all the necessary code for Project 3, run it within this folder. The following libraries are required:

1. `numpy`
2. `pickle`
3. `matplotlib.pyplot`
4. `skimage`
5. `scipy.spatial`

Functions `get_points()` and `get_tri()` in **main.py** are used to manually select correspondences of image and generate triangulation. These functions are not called as correspondences and their triangulation are already generated and stored in _.pickle_ files. So instead, functions that reads these datas, `read_points()` and `read_tri()` respectively, are called.

Finished on Oct 2, 2024.
