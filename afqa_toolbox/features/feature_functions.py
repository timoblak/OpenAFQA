from afqa_toolbox.tools import EPS
import cv2
import numpy as np


def slanted_block_properties(img_size, blk_size, v1sz_x, v1sz_y):
    """Calculates properties for sliding window processing

    :param img_size: Size of input image
    :param blk_size: Size of individual blocks
    :param v1sz_x: Width of slanted blocks
    :param v1sz_y: Height of slanted blocks
    :return: Offset based on block size in px, number of rows and cols of the resulting matrix
    """

    diag = np.ceil(np.sqrt((v1sz_x * v1sz_x) + (v1sz_y * v1sz_y)))
    diff = diag - blk_size
    blk_offset = int(np.ceil(diff / 2))

    map_rows = int(img_size[0] / blk_size)
    map_cols = int(img_size[1] / blk_size)
    return blk_offset, map_rows, map_cols


def block_properties(img_size, blk_size):
    """Calculates properties for sliding window processing

    :param img_size: Size of input image
    :param blk_size: Size of individual blocks
    :return: Number of rows and cols of the resulting matrix
    """
    map_rows = int(img_size[0] / blk_size)
    map_cols = int(img_size[1] / blk_size)
    return map_rows, map_cols


def get_rotated_block(block, orientation, pad=False):
    """Orient a local image patch based on the angle of principal axis.

    :param block: Local image patch
    :param orientation: Angle of principal axis
    :param pad: Optional padding
    :return: Rotated local image patch
    """

    c_block = block.shape[0] / 2  # flaat
    ic_block = block.shape[0] // 2  # int
    if c_block != ic_block:
        print("Warning: Wrong block size! Consider using even number.")

    if pad:
        inblock = cv2.copyMakeBorder(block, 2, 2, 2, 2, borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        inblock = block

    orient_degrees = np.rad2deg(orientation)

    # Subtract 1 to block size to calculate true (float) center, not center indices
    # size 3 -> indices [0, 1, 2] -> 3-1=2 -> center at 1
    # size 4 -> center [0, 1, 2, 3] -> 4-1=3 -> center at 1.5
    center = (inblock.shape[1]-1) / 2, (inblock.shape[0]-1) / 2

    rot_mat = cv2.getRotationMatrix2D(center, orient_degrees, 1)

    rotatedBlock = cv2.warpAffine(inblock, rot_mat, (inblock.shape[1], inblock.shape[1]), flags=cv2.INTER_NEAREST)
    return rotatedBlock


def covcoef(block, mode="diff_cv"):
    """Calculates image covariances of an image block

    :param block: Local patch from image
    :param mode: Type of algorithm for determining derivative [sobel_cv, diff_cv, diff_matlab]
    :return: Image covariances a, b and c
    """
    assert len(block.shape) == 2
    assert mode == "c_diff_cv" or mode == "c_diff_loop" or mode == "sobel_cv"

    # Central differences with filter2D (like matlab) #FASTEST OPTION
    if mode == "c_diff_cv":
        kernelx = np.array([[-0.5, 0, 0.5]])
        kernely = np.array([[-0.5], [0], [0.5]])

        # calculate central differences
        fx = cv2.filter2D(block, cv2.CV_64F, kernelx)
        fy = cv2.filter2D(block, cv2.CV_64F, kernely)

        # calculate forward differences for borders only
        fx[:, len(fx) - 1] = block[:, len(fx) - 1].astype(np.float64) - block[:, len(fx) - 2]
        fx[:, 0] = block[:, 1].astype(np.float64) - block[:, 0]

        fy[len(fy) - 1] = block[len(fy) - 1].astype(np.float64) - block[len(fy) - 2]
        fy[0] = block[1].astype(np.float64) - block[0]

    # Manual central differences (like matlab)
    elif mode == "c_diff_loop":
        image = block.astype(np.float64)
        fx, fy = np.zeros(image.shape, dtype=np.float64), np.zeros(image.shape, dtype=np.float64)
        nrows, ncols = image.shape

        fy[0] = image[1] - image[0]
        fy[nrows-1] = image[nrows - 1] - image[nrows - 2]
        for r in range(1, nrows - 1):
            fy[r] = (image[r + 1] - image[r - 1]) / 2.0

        fx[:, 0] = image[:, 1] - image[:, 0]
        fx[:, ncols - 1] = image[:, ncols - 1] - image[:, ncols - 2]
        for c in range(1, ncols - 1):
            fx[:, c] = (image[:, c + 1] - image[:, c - 1]) / 2.0

    # Using Sobel (a bit different from matlab)
    else:
        fx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
        fy = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)

    a = np.mean(fx * fx)
    b = np.mean(fy * fy)
    c = np.mean(fx * fy)

    return a, b, c


def orient(a, b, c):
    """Calculates principal axis of variation from image covariances of an image block

    :param a: Image covariance a
    :param b: Image covariance b
    :param c: Image covariance c
    :return: Estimated angle of principal axis
    """
    tmp = (a - b)
    denom = (c * c + tmp * tmp) + EPS
    sin2theta = c / denom
    cos2theta = tmp / denom
    return np.arctan2(sin2theta, cos2theta) / 2

