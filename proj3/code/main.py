import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import Delaunay
import skimage as ski


NUM_POINTS1 = 100 # number of correspondences in task 1.
NUM_POINTS2 = 68 # number of correspondences in task 2.
NUM_FRAMES = 51 # number of frames in morphed image sequence.
NUM_IMGS = 12 # number of images in population.

def get_points(img_name: str, num_points: int) -> np.ndarray:
    """select and save correspondences to file."""
    img = plt.imread("media/" + img_name + ".png")
    plt.imshow(img)
    points = plt.ginput(n=num_points-4, timeout=0)
    edges = [(0, 0), (img.shape[1] - 1, 0), (0, img.shape[0] - 1), (img.shape[1] - 1, img.shape[0] - 1)]
    points.extend(edges)
    points = np.array(points)[:,::-1]
    with open("media/" + img_name + "_pt.pickle", "wb") as file:
        pickle.dump(points, file)
    return points

def read_points(img_name: str) -> np.ndarray:
    """read recorded correspondences from file."""
    with open("media/" + img_name + "_pt.pickle", "rb") as file:
        points = pickle.load(file)
    return points

def get_tri(points: np.ndarray, save_to = None) -> Delaunay:
    """compute triangulation of correspondences using Delaunay Triangulation."""
    tri = Delaunay(points)
    if save_to is not None:
        with open(save_to + "/tri.pickle", "wb") as file:
            pickle.dump(tri, file)
    return tri

def read_tri() -> Delaunay:
    """read recorded triangulation from file"""
    with open("media/tri.pickle", "rb") as file:
        tri = pickle.load(file)
    return tri

def show_points(img_name: str, points: np.ndarray, tri: Delaunay) -> None:
    """show image's correspondences and triangulation."""
    img = plt.imread("media/" + img_name + ".png")
    plt.imshow(img)
    plt.triplot(points[:,1], points[:,0], tri.simplices)
    plt.plot(points[:,1], points[:,0], 'o')
    plt.show()


def affine_mat(tri1_pts: list, tri2_pts: list) -> tuple[np.ndarray, np.ndarray]:
    """compute linear transformation that affine triangle 1 into triangle 2."""
    [(t1_y1, t1_x1), (t1_y2, t1_x2), (t1_y3, t1_x3)] = tri1_pts
    [(t2_y1, t2_x1), (t2_y2, t2_x2), (t2_y3, t2_x3)] = tri2_pts
    trans1 = np.array([[t1_y2 - t1_y1, t1_y3 - t1_y1], [t1_x2 - t1_x1, t1_x3 - t1_x1]])
    trans2 = np.array([[t2_y2 - t2_y1, t2_y3 - t2_y1], [t2_x2 - t2_x1, t2_x3 - t2_x1]])
    out_mat = trans2 @ np.linalg.inv(trans1)
    bias = np.array([t2_y1, t2_x1]) - out_mat @ np.array([t1_y1, t1_x1])
    return out_mat, bias

def snap(values: list, range: tuple) -> list:
    """snap values within range."""
    for value_idx, value in enumerate(values):
        if value < range[0]:    values[value_idx] = range[0]
        elif value > range[1]:  values[value_idx] = range[1]
    return values

def morph(img1: np.ndarray, # input image 1.
          img1_points: np.ndarray, # coordinates of points marked on img1.
          img2: np.ndarray, # input image 2.
          img2_points: np.ndarray, # coordinates of points marked on img2.
          triangulation: Delaunay, # triangulation of the points.
          warp_frac: float, # within range [0, 1]; output image's shape gradually changes from img1 to img2 as warp_frac increases.
          dissolve_frac: float # within range [0, 1]; output image's color gradually changes from img1 to img2 as color_frac increases.
    ) -> np.ndarray: # outputs the morphed image.
    """combine image by morphing image 1 and image 2."""
    morph_points = img1_points * (1 - warp_frac) + img2_points * warp_frac # compute the corresponding points on morph_img.
    mask = np.zeros(img1.shape[:2], dtype=int) - 1 # mask image records which triangle each pixel belongs.
    img1_affine_mats, img2_affine_mats = [], [] # transformation from triangles from morph_img to img1/img2.
    for tri_idx, tri_corners in enumerate(triangulation.simplices):
        # for each triangle on morph image, obtains all its pixels, and label the elements on mask.
        # for each triangle on morph_img/img1/img2, obtain its corners to compute affine matrix.
        morph_tri_coords = [morph_points[corner] for corner in tri_corners] # this triangle's corner coords on morph_img.
        img1_tri_coords = [img1_points[corner] for corner in tri_corners] # this triangle's corner coords on img1.
        img2_tri_coords = [img2_points[corner] for corner in tri_corners] # this triangle's corner coords on img2.
        morph_tri_rows = [coord[0] for coord in morph_tri_coords]
        morph_tri_cols = [coord[1] for coord in morph_tri_coords]
        tri_pixels = ski.draw.polygon(morph_tri_rows, morph_tri_cols) # obtain pixels within this triangle on morph_img.
        mask[tri_pixels] = tri_idx # label the pixels within this triangle using the triangle's index.
        img1_affine_mats.append(affine_mat(morph_tri_coords, img1_tri_coords)) # compute this triangle's transformation from morph_img to img1.
        img2_affine_mats.append(affine_mat(morph_tri_coords, img2_tri_coords)) # compute this triangle's transformation from morph_img to img2.
    morphed_img1, morphed_img2 = np.zeros(img1.shape), np.zeros(img2.shape)
    for (y, x), tri_idx in np.ndenumerate(mask):
        # obtain corresponding pixels in img1/img2:
        (T1, b1), (T2, b2) = img1_affine_mats[tri_idx], img2_affine_mats[tri_idx]
        (y1, x1) = (T1 @ np.array([y, x]) + b1).astype(int)
        (y2, x2) = (T2 @ np.array([y, x]) + b2).astype(int)
        # snap coordinates into range:
        [y1, y2] = snap([y1, y2], (0, mask.shape[0]-1))
        [x1, x2] = snap([x1, x2], (0, mask.shape[1]-1))
        # morph pixel:
        morphed_img1[y, x] = img1[y1, x1]
        morphed_img2[y, x] = img2[y2, x2]
    out_img = morphed_img1 * (1 - dissolve_frac) + morphed_img2 * dissolve_frac
    return out_img

def img_mean(path: str, num_imgs: int):
    """compute mean img from collected points.
       also returns the mean points."""
    # read imgs and their points:
    imgs, imgs_points = [], []
    for idx in range(num_imgs):
        img = plt.imread(path + str(idx) + ".png")
        imgs.append(img)
        with open(path + str(idx) + "_pt.pickle", "rb") as file:
            img_point = pickle.load(file)
            imgs_points.append(img_point)
    # get mean points:
    mean_points = np.array(imgs_points).mean(axis=0)
    # get triangulation:
    tri = get_tri(mean_points)
    # label each pixel with its triangle, and record the triangles' coordinates:
    mask = np.zeros(imgs[0].shape[:2], dtype=int) - 1 # mask image records which triangle each pixel belongs.
    mean_tris_coords = [] # record the corner coordinates for each triangle.
    for tri_idx, tri_corners in enumerate(tri.simplices):
        mean_tri_coords = [mean_points[corner] for corner in tri_corners] # this triangle's corner coords on morph_img.
        mean_tri_rows = [coord[0] for coord in mean_tri_coords]
        mean_tri_cols = [coord[1] for coord in mean_tri_coords]
        tri_pixels = ski.draw.polygon(mean_tri_rows, mean_tri_cols) # obtain pixels within this triangle on morph_img.
        mask[tri_pixels] = tri_idx # label the pixels within this triangle using the triangle's index.
        mean_tris_coords.append(mean_tri_coords)
    # affine each image's shape to the mean image:
    morphed_imgs = [] # record the morphed images.
    for img, img_points in zip(imgs, imgs_points):
        img_affine_mats = [] # record the affine matrix for each triangle.
        morphed_img = np.zeros(img.shape)
        for tri_idx, (tri_corners, mean_tri_coords) in enumerate(zip(tri.simplices, mean_tris_coords)):
            img_tri_coords = [img_points[corner] for corner in tri_corners]
            img_affine_mats.append(affine_mat(mean_tri_coords, img_tri_coords))
        for (y, x), tri_idx in np.ndenumerate(mask):
            # obtain corresponding pixels in img1/img2:
            (T, b) = img_affine_mats[tri_idx]
            (y_ori, x_ori) = (T @ np.array([y, x]) + b).astype(int)
            # snap coordinates into range:
            [y_ori] = snap([y_ori], (0, mask.shape[0]-1))
            [x_ori] = snap([x_ori], (0, mask.shape[1]-1))     
            # morph pixel:
            morphed_img[y, x] = img[y_ori, x_ori]
        morphed_imgs.append(morphed_img)
        # plt.imshow(morphed_img)
        # plt.show()
    mean_img = np.array(morphed_imgs).mean(axis=0)
    return mean_img, mean_points
    


# read images, their correspondences and triangulation from file:
# - to manually select your own correspondences, 
#   replace "read_points()" and "read_tri()" with "get_points()" and "get_tri()"
img1, img2 = plt.imread("media/elon_musk.png"), plt.imread("media/george_clooney.png")
img1_points, img2_points = read_points("elon_musk"), read_points("george_clooney")
tri = read_tri()
# show correspondences and triangulation:
show_points("elon_musk", img1_points, tri)
show_points("george_clooney", img2_points, tri)

# morph images (compute midway image):
out_img = morph(img1, img1_points, img2, img2_points, tri, warp_frac=0.5, dissolve_frac=0.5)
plt.imshow(out_img)
plt.show()
plt.imsave("media/midway.png", out_img)

# produce sequence of morphed images with different weight.
for frame in range(NUM_FRAMES):
    print(f"generating frame {frame} / {NUM_FRAMES}")
    morph_weight = frame / (NUM_FRAMES - 1)
    out_img = morph(img1, img1_points, img2, img2_points, tri, warp_frac=morph_weight, dissolve_frac=morph_weight)
    plt.imsave("media/morph_imgs/morph_img_" + str(morph_weight) + ".png", out_img)

# produce mean image of grim Brazilian male and smiling Brazilian male:
mean_grim_img, mean_grim_points = img_mean("media/boys_regular/", NUM_IMGS)
mean_smile_img, mean_smile_points = img_mean("media/boys_smile/", NUM_IMGS)
plt.imshow(mean_grim_img)
plt.show()
plt.imsave("media/mean_img_regular.png", mean_grim_img)
plt.imshow(mean_smile_img)
plt.show()
plt.imsave("media/mean_img_smile.png", mean_smile_img)

# stretch my image into brazilian style and smiling style.
me_img = plt.imread("media/me.png")
me_grim_points = read_points("me_grim")
me_smile_points = read_points("me_smile")
me_brazilian = morph(me_img, me_grim_points, mean_grim_img, mean_grim_points, get_tri(me_grim_points), warp_frac=1, dissolve_frac=0)
me_smile = morph(me_img, me_smile_points, mean_smile_img, mean_smile_points, get_tri(me_smile_points), warp_frac=1, dissolve_frac=0)
plt.imshow(me_brazilian)
plt.show()
plt.imshow(me_smile)
plt.show()
plt.imsave("media/me_morphed.png", me_brazilian)
plt.imsave("media/me_smile.png", me_smile)