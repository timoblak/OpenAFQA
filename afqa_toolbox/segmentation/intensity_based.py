import cv2
import numpy as np


def std_segmentation(image, blk_size, min_std=8):
    offset = blk_size//2
    bordered_image = cv2.copyMakeBorder(image, offset, offset, offset, offset, cv2.BORDER_REPLICATE)
    std_image = np.zeros(image.shape, dtype=int)
    for i in range(offset, image.shape[0] + offset):
        for j in range(offset, image.shape[1] + offset):
            patch = bordered_image[i-offset:i+offset, j-offset:j+offset]
            std_image[i-offset, j-offset] = 1 if patch.std() > min_std else 0
    return std_image


def hist_equalization_segmentation(image, threshold=127):
    image_copy = image.copy()
    image_eq = 255 - cv2.equalizeHist(image_copy)
    image_eq_gauss = cv2.GaussianBlur(image_eq, (17, 17), 8)

    image_eq_mask = np.zeros(image.shape, dtype=np.uint8)
    image_eq_mask[image_eq_gauss > threshold] = 255
    image_eq_mask = cv2.morphologyEx(image_eq_mask, cv2.MORPH_CLOSE,
                                     kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    image_eq_mask = cv2.morphologyEx(image_eq_mask, cv2.MORPH_OPEN,
                                     kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return image_eq_mask