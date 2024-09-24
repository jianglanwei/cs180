import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import cv2

GAUSSIAN_STACK_DEPTH = 5

def to_bin(img: np.ndarray, threshold) -> np.ndarray:
    """convert image to binary image."""
    return np.where(abs(img) >= threshold, 1, 0)

def blur_img(img: np.ndarray, ksize: int) -> tuple[np.ndarray, np.ndarray]:
    """return 2 dimensional gaussian filter"""
    filter1d = cv2.getGaussianKernel(ksize, ksize / 6)
    filter2d = filter1d @ filter1d.T
    if len(img.shape) == 2:
        blurred_img = scipy.signal.convolve2d(img, filter2d, mode="same", boundary="symm")
    else:
        blurred_layers = [scipy.signal.convolve2d(img[:,:,layer_idx], filter2d, mode="same", boundary="symm") for layer_idx in range(img.shape[2])]
        blurred_img = np.stack(blurred_layers, axis=2)
    return blurred_img


def get_edge(img: np.ndarray, # input image.
             ksize: int, # kernel size.
             combine_filter: bool = True, # combine the gaussian and edge filter instead of doing convolution twice.
             bin_edge: bool = True, # return binary edge.
             therehold: float = None # therehold of edge binaryzation.
    ) -> np.ndarray:
    """get edge from image."""
    assert bin_edge is False or therehold is not None
    dx, dy = np.array([[1, -1]]), np.array([[1, -1]]).T
    blurred_img = blur_img(img, ksize)
    if combine_filter:
        filter1d = cv2.getGaussianKernel(ksize, ksize / 6)
        filter2d = filter1d @ filter1d.T
        smooth_dx = scipy.signal.convolve2d(filter2d, dx)
        smooth_dy = scipy.signal.convolve2d(filter2d, dy)
        ver_edge = scipy.signal.convolve2d(img, smooth_dx, mode="same")
        hor_edge = scipy.signal.convolve2d(img, smooth_dy, mode="same")
    else:
        ver_edge = scipy.signal.convolve2d(blurred_img, dx, mode="same")
        hor_edge = scipy.signal.convolve2d(blurred_img, dy, mode="same")
    edge = np.sqrt(ver_edge ** 2 + hor_edge ** 2)
    if bin_edge:  edge = to_bin(edge, therehold)
    return edge

def sharpen_img(img: np.ndarray, alpha: int, ksize: int) -> np.ndarray:
    """sharpen image using unsharp masking technic."""
    high_freq_details = img - blur_img(img, ksize)
    sharp_img = img + high_freq_details * alpha
    sharp_img = np.where(sharp_img < 0, 0, sharp_img)
    sharp_img = np.where(sharp_img > 1, 1, sharp_img)
    return sharp_img

def hybrid_img(low_freq_img: np.ndarray, # image contributing to the low frequency features.
               high_freq_img: np.ndarray, # image contributing to the high frequency details.
               low_freq_ksize: int, # kernel size for gaussian filter on low_freq_img.
               high_freq_ksize: int, alpha: int = 1 # kernel size for gaussian filter on high_freq_img. 
    ) -> np.ndarray:
    """combine two images by adding the low frequency features of one image and the high frequency details of the other."""
    low_freq_img = blur_img(low_freq_img, ksize=low_freq_ksize)
    high_freq_img -= blur_img(high_freq_img, ksize=high_freq_ksize)
    high_freq_img_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_freq_img[:,:,0]))))
    low_freq_img_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_freq_img[:,:,0]))))
    plt.imsave("media/spiderman_high_fft_filtered.jpg", high_freq_img_fft)
    plt.imsave("media/spiderman_low_fft_filtered.jpg", low_freq_img_fft)
    hybird_img = low_freq_img + high_freq_img * alpha
    hybird_img = np.where(hybird_img < 0, 0, hybird_img)
    hybird_img = np.where(hybird_img > 1, 1, hybird_img)
    return hybird_img

def gaussian_stack(img: np.ndarray, # input image.
            ksize: int = 3, # kernel size for gaussian filter.
            dksize: int = 10, # difference of kernel size between neighbouring layers.
            recursion_count: int = 0 # number of recursions.
    ) -> list: # returns the layers in this stack. 
    """gaussian stack implemented by recursive function."""
    if recursion_count == GAUSSIAN_STACK_DEPTH - 1:
        return [img]
    blurred_img = blur_img(img, ksize)
    gstack = gaussian_stack(blurred_img, ksize=(ksize + dksize), recursion_count=(recursion_count + 1))
    gstack.insert(0, img)
    return gstack

def laplacian_stack(gstack: list) -> list:
    """lapliacians stack obtained by gaussian stack."""
    lstack = [gstack[i] - gstack[i + 1] for i in range(GAUSSIAN_STACK_DEPTH - 1)]
    lstack.append(gstack[-1])
    return lstack

def blend(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray):
    """multiresolution blending using laplacian stack."""
    gstack1, gstack2 = gaussian_stack(img1), gaussian_stack(img2)
    mask_stack = gaussian_stack(mask, ksize=1, dksize=100)
    lstack1, lstack2 = laplacian_stack(gstack1), laplacian_stack(gstack2)
    out_img = np.zeros(img1.shape, dtype=img1.dtype)
    for depth in range(GAUSSIAN_STACK_DEPTH):
        masked_img1, masked_img2 = lstack1[depth] * mask_stack[depth], lstack2[depth] * (1 - mask_stack[depth])
        blend_layer = masked_img1 + masked_img2
        out_img += blend_layer
    plt.show()
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
    return out_img


# project 1.1 & 1.2

img = plt.imread("media/cameraman.png")[:,:,0]
edge = get_edge(img, ksize=9, combine_filter=True, bin_edge=True, therehold=0.1)
plt.imsave("media/edge_smoothed_uncombined_filter.png", edge, cmap="gray")
edge = get_edge(img, ksize=9, combine_filter=False, bin_edge=True, therehold=0.1)
plt.imsave("media/edge_smoothed_combined_filter.png", edge, cmap="gray")
plt.imshow(edge, cmap="gray")
plt.show()

# project 2.1

# sharpen taj.jpg
img = plt.imread("media/taj.jpg") / 255
sharp_img = sharpen_img(img, alpha=1, ksize=7)
plt.subplot(121)
plt.imshow(img)
plt.title("original image")
plt.subplot(122)
plt.imshow(sharp_img)
plt.title("sharpened image")
plt.show()
plt.imsave("media/taj_sharpened.jpg", sharp_img)

# sharpen forest.jpg
img = plt.imread("media/forest.jpg") / 255
sharp_img = sharpen_img(img, alpha=1.5, ksize=9)
plt.subplot(121)
plt.imshow(img)
plt.title("original image")
plt.subplot(122)
plt.imshow(sharp_img)
plt.title("sharpened image")
plt.show()
plt.imsave("media/forest_sharpened.jpg", sharp_img)

# blur and sharpen tiger.png
img = plt.imread("media/tiger.png")
blurred_img = blur_img(img, ksize=7)
sharp_img = sharpen_img(blurred_img, alpha=2, ksize=15)
plt.subplot(311)
plt.imshow(img)
plt.title("original image")
plt.subplot(312)
plt.imshow(blurred_img)
plt.title("blurred image")
plt.subplot(313)
plt.imshow(sharp_img)
plt.title("sharpened image")
plt.show()
plt.imsave("media/tiger_blurred.png", blurred_img)
plt.imsave("media/tiger_sharpened.png", sharp_img)

# project 2.2

# generate spiderman hybrid image
high_freq_img = plt.imread("media/spiderman_high.jpg") / 255
low_freq_img = plt.imread("media/spiderman_low.jpg") / 255
hybrid_img_ = hybrid_img(low_freq_img, high_freq_img, low_freq_ksize=19, high_freq_ksize=13)
plt.imshow(hybrid_img_)
plt.show()
plt.imsave("media/spiderman_hybrid.jpg", hybrid_img_)


# generate wednesday hybrid image
high_freq_img = plt.imread("media/cartoon.jpg") / 255
low_freq_img = plt.imread("media/real.jpg") / 255
hybrid_img_ = hybrid_img(low_freq_img, high_freq_img, low_freq_ksize=19, high_freq_ksize=13, alpha=3)
plt.imshow(hybrid_img_)
plt.show()
plt.imsave("media/wednesday_hybrid.jpg", hybrid_img_)


# project 2.3 and 2.4.

# blend apple and orange into oraple

img1 = plt.imread("media/apple.png")
img2 = plt.imread("media/orange.png")
mask = plt.imread("media/oraple_mask.png")[:,:,:1]
out_img = blend(img1, img2, mask)
plt.subplot(131)
plt.imshow(img1)
plt.title("apple")
plt.subplot(132)
plt.imshow(out_img)
plt.title("oraple")
plt.subplot(133)
plt.imshow(img2)
plt.title("orange")
plt.show()
plt.imsave("media/oraple.png", out_img)

# blend earth and moon into moonearth

img1 = plt.imread("media/earth.jpg") / 255
img2 = plt.imread("media/moon.jpg") / 255
mask = plt.imread("media/moonearth_mask.png")[:,:,:1]
out_img = blend(img1, img2, mask)
plt.subplot(131)
plt.imshow(img1)
plt.title("earth")
plt.subplot(132)
plt.imshow(out_img)
plt.title("moonearth")
plt.subplot(133)
plt.imshow(img2)
plt.title("moon")
plt.show()
plt.imsave("media/moonearth.png", out_img)

# blend mike and kangaroo into a scornful kangaroo (mikaroo)

img1 = plt.imread("media/mike.jpg") / 255
img2 = plt.imread("media/kangaroo.jpg") / 255
mask = (plt.imread("media/mikaroo_mask.jpg") / 255)[:,:,:1]
out_img = blend(img1, img2, mask)
plt.subplot(131)
plt.imshow(img1)
plt.title("mike")
plt.subplot(132)
plt.imshow(out_img)
plt.title("mikaroo")
plt.subplot(133)
plt.imshow(img2)
plt.title("kangaroo")
plt.show()
plt.imsave("media/mikaroo.png", out_img)

# blend earth and moon into moonearth_ultra (use blending result from last time as mask.)

img1 = plt.imread("media/earth.jpg") / 255
img2 = plt.imread("media/moon.jpg") / 255
mask = plt.imread("media/moonearth.png")[:,:,:1]
out_img = blend(img1, img2, mask)
plt.subplot(131)
plt.imshow(img1)
plt.title("earth")
plt.subplot(132)
plt.imshow(out_img)
plt.title("moonearth ultra")
plt.subplot(133)
plt.imshow(img2)
plt.title("moon")
plt.show()
plt.imsave("media/moonearth_ultra.png", out_img)