## Project 4: Stitching Photo Mosaics

Official Spec (
    [Part A](https://github.com/jianglanwei/cs180/tree/main/proj4/Official-Spec-A)
    /
    [Part B](https://github.com/jianglanwei/cs180/tree/main/proj4/Official-Spec-B)
)
/
[My Webpage](https://jianglanwei.github.io/cs180/proj4/) 
/
[My Code](https://github.com/jianglanwei/cs180/tree/main/proj4/code) 

Project 4 consists two parts: <b>Image Warping and Mosaicing</b> (Part A) and <b>Feature Matching for Autostitching</b> (Part B).     

In Part A, I rectified images using <b>Perspective Transform</b>. I manually selected correspondences on the images, and warped themso that their transformed correspondences form a rectangle.I also produced <b>mosaics images</b> by blending pairs of images that overlap with each other. First, I manually matched a few pixels that represent the same corner of an object on the two images. Then, I treated these pixel matches as correspondences, and warped the first image so that after warping, the correspondences on the first image aligns with the correspondences on the second images. In this way, the same objects on the two images would match. Finally, I conducted <b>Alpha Blend</b> on the output mosaic to erase the edge between the 
two images.     

In Part B, I also produced mosaic images, only this time instead of manually matching the pixels, the pixel matches are automatically detected and selected.Corners serve as great symbols of objects on an image, so I used <b>Harris Corner Detector</b> to find the corners on the images, and treat them as interest points. Then, I used <b>Adaptive Non-Maximal Suppression (ANMS)</b> to select a few interest points that are not only high in "corner strength", but also as uniformly distributed in the image as possible. They are the potential correspondences. Later, I matched the potential correspondences using <b>Feature Descriptors</b>. If the best match of a potential correspondence did not score significantly higher than its second-best match, I would abandon this pixel. The matched pixels still may contain error. I found the optimal set of matches using the idea of <b>Random Sample Consensus (RANSAC)</b>. At last, similar to Part A, I used the optimal matches to conduct perspective transform on the first image so that it aligns with the second image, and blended the overlapping region to erase the edge.

> **Disclaimer:**  All materials, including assignments, documents, and source code, are provided solely for personal coursework. All rights reserved by the UC Berkeley CS180 course staff. Redistribution or commercial use is strictly prohibited.

Finished on Oct 19, 2024 (Part A), Oct 24, 2024 (Part B).
