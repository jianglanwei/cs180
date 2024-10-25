from proj4a_func import *
from proj4b_func import *


# implementation code for project 4a.

# image rectification.

img_names = ["sheet", "sign"]

for img_name in img_names:
    out_img = rectification(img_name)
    plt.imshow(out_img)
    plt.show()
    plt.imsave("media/" + img_name + "_rectificated.png", out_img)

# image mosaic using manually selected correspondences.

img1_names = ["tower_top", "road_left", "room_left"]
img2_names = ["tower_bottom", "road_right", "room_right"]
out_img_names = ["tower", "road", "room"]

for img1_name, img2_name, out_img_name in zip(img1_names, img2_names, out_img_names):
    show_points(img1_name, img2_name)
    mosaic_img = mosaic(img1_name, img2_name, lr=LR)
    plt.imshow(mosaic_img)
    plt.show()
    plt.imsave("media/" + out_img_name + "_mosaic.png", mosaic_img)


# implementation code for project 4b.

img1_names = ["moffitt_left", "hub_left", "door_left"]
img2_names = ["moffitt_right", "hub_right", "door_right"]
out_img_names = ["moffitt", "hub", "door"]


for img1_name, img2_name, out_img_name in zip(img1_names, img2_names, out_img_names):
    # image mosaic using manual selected correspondences.
    show_points(img1_name, img2_name)
    mosaic_manual = mosaic(img1_name, img2_name)

    # image mosaic using automatically generated correspondences.
    mosaic_auto = auto_mosaic(img1_name, img2_name)

    # show mosaic image.
    ax1 = plt.subplot(211)
    ax1.imshow(mosaic_manual)
    ax1.set_title("mosaic image using manually selected correspondences.")
    ax2 = plt.subplot(212)
    ax2.imshow(mosaic_auto)
    ax2.set_title("mosaic image using auto genertated correspondences.")
    plt.show()

    # save masaic image.
    plt.imsave("media/" + out_img_name + "_mosaic_manual.png", mosaic_manual)
    plt.imsave("media/" + out_img_name + "_mosaic_auto.png", mosaic_auto)