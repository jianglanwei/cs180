import numpy as np
from numpy import ndarray
import skimage as ski
from skimage import filters, measure, io
import os


def exhaustive_search(img1: ndarray, 
                      img2: ndarray, 
                      search_center: tuple = (0, 0),  # the area for exhaustive search will be located near search center.
                      search_range: range = range(-10, 10), # range of exhaustive search.
                      frame_len: int = 30) -> tuple: # length of image frame located on the side of image. 
    """find optimal alignment by exhaustive searching for the match corresponding to the maximum simularity (minimum squared loss).
       - img1 will be aligned to img2;
       - only pixels within the frame will be calculated in loss."""
    min_loss = float("inf")
    img2_frameless = img2[frame_len:-frame_len, frame_len:-frame_len] # image frames are cut out for a more accurate alignement.
    for search_y in search_range:
        for search_x in search_range:
            shift_y, shift_x = search_center[0] + search_y, search_center[1] + search_x
            img1_frameless = img1[(frame_len - shift_y):(-frame_len - shift_y), (frame_len - shift_x):(-frame_len - shift_x)] # shift img1 while cutting out frame.
            loss = np.sum((img1_frameless - img2_frameless) ** 2)
            if loss < min_loss: 
                opt_shift = (shift_y, shift_x)
                min_loss = loss
    return opt_shift

def pyramid(edges: list, # detected edge in three layers.
            ds_ratio: int = 3, # downsampling ratio.
            exhaustive_search_range: range = range(-5, 5), 
            frame_len: int = 300, # length of image frame located on the side of image. 
            recursive_end: int = 500) -> tuple: # recursion stops when image height is smaller than this number.
    """recursive function that finds optimal shift length for blue and green layer."""
    # if image is small enough do exhaustive search directly:
    if edges[0].shape[0] < recursive_end:
        b_shift = exhaustive_search(edges[0], edges[2])
        g_shift = exhaustive_search(edges[1], edges[2])
        return (b_shift, g_shift)
    
    # downsample image to find the approximate location for optimal alignment.
    (b_shift, g_shift) = pyramid(edges = [measure.block_reduce(edge, (ds_ratio, ds_ratio), np.mean) for edge in edges], # downsampled edges.
                                 ds_ratio = ds_ratio, 
                                 exhaustive_search_range = exhaustive_search_range, 
                                 frame_len = frame_len // ds_ratio, # frame length decreases as image becomes smaller.
                                 recursive_end = recursive_end)
    
    # exhaustive search for more accurate alignment near the approximate result obtained by downsampled image.
    b_shift = exhaustive_search(edges[0], edges[2], (b_shift[0] * ds_ratio, b_shift[1] * ds_ratio), exhaustive_search_range, frame_len)
    g_shift = exhaustive_search(edges[1], edges[2], (g_shift[0] * ds_ratio, g_shift[1] * ds_ratio), exhaustive_search_range, frame_len)
    return (b_shift, g_shift)

def align_img(img_name: str, compress: bool = False) -> ndarray:
    """aligns blue and green layer to red layer and stacks them into colored image."""
    # read and preprocess image.
    img = io.imread(img_name, as_gray=True)
    img = ski.img_as_float(img)
    height = np.floor(img.shape[0] / 3.0).astype(np.int32)
    layers = [img[(i * height):((i + 1) * height)] for i in range(3)]

    # edge detection. image are aligned by aligning their edges.
    edges = [filters.sobel(layer) for layer in layers]


    # obtain optimal alignment using image pyramid and exhaustive search.
    (b_shift, g_shift) = pyramid(edges)


    # align and stack image according to optimal alignment.
    layers[0] = np.roll(layers[0], b_shift[1], axis=1)
    layers[0] = np.roll(layers[0], b_shift[0], axis=0)
    layers[1] = np.roll(layers[1], g_shift[1], axis=1)
    layers[1] = np.roll(layers[1], g_shift[0], axis=0)
    layers.reverse() # change BGR image into RGB order.
    if compress:    layers = [measure.block_reduce(layer, (2, 2), np.mean) for layer in layers] # compress image if required.
    colored_img = np.dstack(layers)
    return colored_img, (b_shift, g_shift)

for file in os.listdir():
    file_type = os.path.splitext(file)[1]
    if file_type == '.jpg':      img, shifts = align_img(file)
    elif file_type == '.tif':    img, shifts = align_img(file, compress=True)
    else:                        continue
    print(f"{file}:\nblue shift: {shifts[0]}\tgreen shift: {shifts[1]}\n ")
    io.imshow(img)
    io.show()
    img = (img * 255).astype(np.uint8)
    io.imsave("out_path/unaligned_" + file, img)
