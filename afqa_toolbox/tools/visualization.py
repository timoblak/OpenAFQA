import cv2
import numpy as np
import matplotlib.cm as cm


def visualize_orientation_field(image, orientation, blk_size=20):
    """Vizualized the orientation field of a friction ridge image

    :param image: A grayscale input image
    :param orientation: Orientation matrix where each element presents the orientation in radians
    :param blk_size: the size of block, showing rotation (can be set independant of orientation matrix of image size)
    :return: image with orientation visualization
    """
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape
    line_len = 0.8*blk_size
    new_shape = h//blk_size, w//blk_size

    field = cv2.resize(orientation, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)

    xoff = line_len / 2 * np.cos(field)
    yoff = line_len / 2 * np.sin(field)

    x, y = np.meshgrid(np.linspace(blk_size//2, w-blk_size//2, new_shape[1]), np.linspace(blk_size//2, h-blk_size//2, new_shape[0]))
    x -= xoff
    y -= yoff

    x = x.astype(int)
    y = y.astype(int)
    u = (x + xoff * 2).astype(int)
    v = (y + yoff * 2).astype(int)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            cv2.line(image_color, (x[i, j], y[i, j]), (u[i, j], v[i, j]), (0, 0, 255))
    return image_color


def visualize_minutiae(image, template, mask=None, min_quality=0, show_type=False, show_others=False):
    """
    Visualizes minutiae points in a template dictionary
    :param image: A grayscale input image
    :param template: A python dictionary of a minutia template file
    :return: a color image of visualized minutiae points
    """
    box_size = 5
    thickness = 1
    line_radius = 15

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for fp in template["fingerprints"]:
        for mnt in fp["minutiae"]:
            x, y = mnt["minx"], mnt["miny"]

            if mask is not None and mask[y, x] == 0:
                continue
            type, angle, quality = mnt["mintype"], mnt["minangle"], mnt["minquality"]

            if quality < min_quality:
                continue
            angle = (angle / 256) * 2 * np.pi

            val = mnt["minquality"]

            color = np.array(cm.jet(val)[:3])*255  # [0, 0, 255]

            # counterclockwise, starting from the right
            x_new = int(round(x + line_radius * np.cos(angle)))
            y_new = int(round(y - line_radius * np.sin(angle)))

            if type == 1:  #  Ending
                if show_type:
                    color = (0, 0, 255)
                cv2.rectangle(image_color, (x-box_size, y-box_size), (x+box_size, y+box_size), color, thickness)
                cv2.line(image_color, (x, y), (x_new, y_new), color, thickness)
            elif type == 2:  #  Bifurcaton
                if show_type:
                    color = (255, 0, 0)
                cv2.rectangle(image_color, (x-box_size, y-box_size), (x+box_size, y+box_size), color, thickness)
                cv2.line(image_color, (x, y), (x_new, y_new), color, thickness)
            elif show_others:
                if show_type:
                    color = (0, 255, 0)
                cv2.rectangle(image_color, (x-box_size, y-box_size), (x+box_size, y+box_size), color, thickness)
                cv2.line(image_color, (x, y), (x_new, y_new), color, thickness)
    return image_color