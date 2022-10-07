import numpy as np
import cv2


EPS = np.finfo(float).eps


def normed(image):
    """Returns a normalized float image between 0 and 1 from any matrix -  also works for matrices with nan values

    :param image: input image
    :return: normalized output image
    """
    if np.all(np.isnan(image)):
        return image
    img = image.copy()
    img -= img[~np.isnan(img)].min()
    img /= img[~np.isnan(img)].max() + 1e-7
    return img


def resized(image, factor):
    """Returns a resized image based on given factor

    :param image: input image
    :return: resized output image
    """
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
