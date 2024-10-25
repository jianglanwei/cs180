import numpy as np
from skimage.feature import corner_harris, peak_local_max
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from proj4a_func import compute_mat, perspective_trans, warp_img, alpha_blend, snap

# this is the implementation code for project 4b.

# param for harris corner:
NUM_PEAKS = 5000 # number of peaks.

# param for anms:
ANMS_LOCAL_RADIUS = 18 # radius of the local area.
# interest points need to be the local maximum to be retained.

# params for descriptor in feature matching:
PATCH_RADIUS = 20 # radius of descriptor's patches.
PATCH_STEP = 5 # step size of descriptor's patches.

# param for feature matching:
# if best match isn't significantly better than the second, drop it.
EXCEED_CONST = 0.6

# params for ransac:
NUM_RANSAC_ATTEMPTS = 1000 # number of attempts to randomly select matches.
ALIGN_THRESHOLD = 3 # when distance between correspondences is under threshold,
                    # they are considered aligned.


def get_harris_corners(im: np.ndarray, edge_discard: int = 20
                       ) -> tuple[np.ndarray, np.ndarray]:
    """ finds harris corners in the image. 
        - harris corners near the edge are discarded. 
        - a 2d array (h) containing the h value of every pixel is also returned.
        - adapted from sample code given by course staff."""
    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1, num_peaks=NUM_PEAKS)

    # discard points on edge
    mask = (coords[:, 0] > edge_discard) & \
           (coords[:, 0] < im.shape[0] - edge_discard) & \
           (coords[:, 1] > edge_discard) & \
           (coords[:, 1] < im.shape[1] - edge_discard)
    coords = coords[mask]
    return h, coords


def dist2(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """ calculates squared distance between two sets of points.
        - adapted from sample code given by course staff."""
    return (np.ones((c.shape[0], 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((x.shape[0], 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

def anms(points: np.ndarray, r: int = ANMS_LOCAL_RADIUS) -> np.ndarray:
    """conduct adaptive non-maximum suppression (anms) on coordinates.
       - return retained coordinates."""
    r2, dist2_mat = r ** 2, dist2(points, points)
    retained_pts = []
    for pt_idx in range(points.shape[0]):
        local_max = True
        for retained_pt_idx in retained_pts:
            if dist2_mat[pt_idx, retained_pt_idx] <= r2:
                local_max = False
                break
        if local_max:
            retained_pts.append(pt_idx)
    retained_points = points[retained_pts]
    return retained_points

def match(img1: np.ndarray, # input b&w image 1.
          img2: np.ndarray, # input b&w image 2.
          points1: np.ndarray, # interest points for image 1.
          points2: np.ndarray # intereset points for image 2.
    ) -> np.ndarray: # point/correspondences matches.
    """match interest points on two images."""
    patch_size = 2 * PATCH_RADIUS // PATCH_STEP
    patches1 = np.zeros((len(points1), patch_size, patch_size))
    patches2 = np.zeros((len(points2), patch_size, patch_size))
    for point_idx, point_coord in enumerate(points1):
        patches1[point_idx] = \
        img1[(point_coord[0]-PATCH_RADIUS):(point_coord[0]+PATCH_RADIUS):PATCH_STEP,
             (point_coord[1]-PATCH_RADIUS):(point_coord[1]+PATCH_RADIUS):PATCH_STEP]
    for point_idx, point_coord in enumerate(points2):
        patches2[point_idx] = \
        img2[(point_coord[0]-PATCH_RADIUS):(point_coord[0]+PATCH_RADIUS):PATCH_STEP,
             (point_coord[1]-PATCH_RADIUS):(point_coord[1]+PATCH_RADIUS):PATCH_STEP]
    norm_patches1 = (patches1 - patches1.mean(axis=(1, 2), keepdims=True)) \
                    / patches1.std(axis=(1, 2), keepdims=True)
    norm_patches2 = (patches2 - patches2.mean(axis=(1, 2), keepdims=True)) \
                    / patches2.std(axis=(1, 2), keepdims=True)
    matches = []
    for patch_idx, patch in enumerate(norm_patches1):
        patch = np.expand_dims(patch, axis=0)
        l2_distance = (patch * norm_patches2).sum(axis=(1, 2))
        first_choice, max_distance = l2_distance.argmax(), l2_distance.max()
        l2_distance[first_choice] = -np.inf
        second_max_distance = l2_distance.max()
        if max_distance * EXCEED_CONST > second_max_distance: 
            # best match is significantly better than the second.
            matches.append((points1[patch_idx], points2[first_choice]))
    return np.array(matches)

def ransac(matches: np.ndarray) -> np.ndarray:
    """use random sample consensus (ransac) to calculate transformation matrix
       that end up with least outliners."""
    img1_pts, img2_pts = matches[:, 0], matches[:, 1]
    choices = [np.random.choice(len(matches), 4, replace=False) 
               for _ in range(NUM_RANSAC_ATTEMPTS)]
    align_counts = np.zeros(NUM_RANSAC_ATTEMPTS)
    for attempt_idx, choice in enumerate(choices):
        img1_chosen_pts, img2_chosen_pts = img1_pts[choice], img2_pts[choice]
        mat = compute_mat(img1_chosen_pts, img2_chosen_pts)
        img1_pts_trans = perspective_trans(mat, img1_pts)
        align_loss = (img1_pts_trans - img2_pts).sum(axis=1) ** 2
        align_counts[attempt_idx] = (align_loss < ALIGN_THRESHOLD ** 2).sum()
    best_choice = choices[align_counts.argmax()]
    img1_chosen_pts, img2_chosen_pts = img1_pts[best_choice], img2_pts[best_choice]
    optimal_mat = compute_mat(img1_chosen_pts, img2_chosen_pts)
    return optimal_mat

def auto_mosaic(img1_name: str, img2_name: str) -> np.ndarray:
    """blend img1 and img2 into a mosaic image automatically.
       - img1 is the image going to be warpped."""
    img1 = plt.imread("media/" + img1_name + ".png")[:, :, :3] + 1e-4
    img2 = plt.imread("media/" + img2_name + ".png")[:, :, :3] + 1e-4
    _, img1_pts = get_harris_corners(img1[:,:,0])
    _, img2_pts = get_harris_corners(img2[:,:,0])
    img1_retained_pts, img2_retained_pts = anms(img1_pts), anms(img2_pts)
    matches = match(img1[:,:,0], img2[:,:,0], img1_retained_pts, img2_retained_pts)
    show_matches(img1, img2, img1_retained_pts, img2_retained_pts, matches)
    mat = ransac(matches)
    out_img1, (offset_y, offset_x)= warp_img(img1, mat, mosaic=True)
    out_img2 = np.zeros(out_img1.shape)
    out_img2[offset_y:offset_y+img2.shape[0], offset_x:offset_x+img2.shape[1]] = img2
    out_img_blend = alpha_blend(out_img1, out_img2)
    out_img_blend = snap(np.where(out_img_blend == 0, 1, out_img_blend), (0, 1))
    return out_img_blend

def show_matches(img1: np.ndarray, img2: np.ndarray, # input images.
                 img1_pts: np.ndarray, img2_pts: np.ndarray, # image interest points.
                 matches: np.ndarray): # interest-point-matches/correspondences.
    """show:
       - interest points on images (blue dots);
       - interest-point-matches/correspondences between images (red line/dot)."""
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1)
    ax1.plot(img1_pts.T[1], img1_pts.T[0], 'o', color='pink', markersize=2)
    ax1.plot(matches[:, 0, 1], matches[:, 0, 0], 'o', color='r', markersize=2)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2)
    ax2.plot(img2_pts.T[1], img2_pts.T[0], 'o', color='pink', markersize=2)
    ax2.plot(matches[:, 1, 1], matches[:, 1, 0], 'o', color='r', markersize=2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    for ((y1, x1), (y2, x2)) in matches:
        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", 
                              coordsB="data", axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
    plt.show()


# out_img = auto_mosaic("hub_left", "hub_right")
# plt.imshow(out_img) 
# plt.show()






# img2 = plt.imread("media/moffitt_right.png")
# h2, points2 = get_harris_corners(img2[:,:,0])
# retained_points2 = anms(points2)
# plt.imsave("media/harris_matrix.png", h2)
# plt.imshow(img2)
# plt.plot(points2[:, 1], points2[:, 0], 'o', color='b', markersize=2)
# plt.plot(retained_points2[:, 1], retained_points2[:, 0], 'o', color='pink', markersize=2)
# plt.show()


# matches = match(img1[:,:,0], img2[:,:,0], retained_points1, retained_points2)
# mat = ransac(matches)


# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)


# ax1.imshow(img1)
# ax1.plot(retained_points1.T[1], retained_points1.T[0], 'o', color='b', markersize=1)
# ax2.imshow(img2)
# ax2.plot(retained_points2.T[1], retained_points2.T[0], 'o', color='b', markersize=1)

# for (y1, x1), (y2, x2) in matches:
#     con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data", 
#                         axesA=ax2, axesB=ax1, color="red")
#     ax2.add_artist(con)
# plt.show()




# plt.subplot(121)
# plt.imshow(h)
# plt.subplot(122)
# plt.imshow(img)
# plt.plot(points.T[1], points.T[0], 'o', color='b', markersize=3)
# plt.plot(retained_points.T[1], retained_points.T[0], 'o', color='r', markersize=3)
# plt.show()
# print(h[points[0], points[1]])
