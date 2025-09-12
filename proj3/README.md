## Project 3: Face Morphing and Modelling a Photo Collection

[Official Spec](https://github.com/jianglanwei/cs180/tree/main/proj3/OfficialSpec.pdf)
/
[My Webpage](https://jianglanwei.github.io/cs180/proj3/) 
/
[My Code](https://github.com/jianglanwei/cs180/tree/main/proj3/code) 

In the first part of this project, I morphed two face images using **Affine Transformation**. I obtained 100 correspondences for each of the two faces and computed their average coordinates. **Triangulation** was conducted on the correspondences and for each triangle, **Affine Matrixes** were generated to stretch the triangles from the original image to the Midway image. I used **Cross Dissolve** to bind the color. I furtherly generated a sequence of 51 morphed images using different **Morph Weight** to produce the morphing GIF. In the second part, I computed the mean face of 12 Brazilian faces, and stretched my face into the shape of the mean face. I also computed the mean face of 12 smiling Brazilian faces to add a smile on my grim portrait.

> **Disclaimer:**  All materials, including assignments, documents, and source code, are provided solely for personal coursework. All rights reserved by the UC Berkeley CS180 course staff. Redistribution or commercial use is strictly prohibited.

Finished on Oct 2, 2024.
