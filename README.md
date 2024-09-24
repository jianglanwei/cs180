## Computer Vision Projects

This repository contains my projects for UC Berkeley's course [COMPSCI 180 - Intro to Computer Vision and Computational Photography](https://inst.eecs.berkeley.edu/~cs180/fa24/) on fall 2024 semester. For detailed introduction, visit [website](https://davidpaulwei.github.io/cs180/).  

1. **Project 1: Colorizing the _Prokudin-Gorskii_ Photo Collection** &emsp; [Website](https://davidpaulwei.github.io/cs180/proj1/) | [My Code](https://github.com/davidpaulwei/cs180/tree/main/proj1/code) | [Github Folder](https://github.com/davidpaulwei/cs180/tree/main/proj1)  
   _Prokudin-Gorskii_ photographed the Russian Empire using black-and-white negatives with red, green and blue filters, hoping future technologies could stack the three layers to produce colored image. I take the digitized negatives of _Prokudin-Gorskii_'s work to produce the RGB color image. The original three color layers from _Prokudin-Gorskii_'s work are not accurately aligned, so I designed an alignment algorithm using **Image Pyramid** and **Edge Detection** to preprocess the layers before stacking them together.

2. **Project 2: Fun with Filters and Frequencies** &emsp; [Website](https://davidpaulwei.github.io/cs180/proj2/) | [My Code](https://github.com/davidpaulwei/cs180/tree/main/proj2/code) | [Github Folder](https://github.com/davidpaulwei/cs180/tree/main/proj2)   
   By applying filters and analyzing frequencies, images can be processed and combined in interesting ways. In the first part of this project, edge detection is conducted by applying the **_Finite Difference Filter_**. _**Gaussian Filter**_ is applied to get rid of the unnecessary wrinkles. Then, images can be sharpened by stacking its edges onto itself. The second part of this project consists of two image binding tasks. The first task generates **Hybrid Image** by adding the high frequency of one image to the low frequency of the another.Both successful and failing attempts are introduced.The second task **blends images** by applying the _**Gaussian Stack**_ and the _**Laplacian Stack**_.

Updated on Sept 23, 2024.
