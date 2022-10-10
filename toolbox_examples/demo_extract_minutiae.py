from afqa_toolbox.enhancement import dog_filter, GaborEnhancement
from afqa_toolbox.minutiae import FJFXWrapper, MinutiaeExtraction, MINDTCTWrapper
from afqa_toolbox.tools import visualize_minutiae, resized, normed
import time
import numpy as np
import cv2
import os

# Windows
FJFX_LIB_PATH = ""  # path to FJFX.dll (windows) or libFJFX.so (linux)
MINDTCT_BIN_PATH = ""  # path to mindtct executable
latent = ""  # Path to latent/fingermark image

image = cv2.imread(latent, 0)

# Get PPI from filename if part of the NIST SD 301/302 datasets, otherwise specify
ppi = int(latent.split("\\")[-1].split("_")[6].replace("PPI", ""))
# ppi =

cv2.namedWindow("window")
cv2.createTrackbar("c", "window", 11, 51, lambda x: x)

fjfx = FJFXWrapper(path_library=FJFX_LIB_PATH)
mindtct = MINDTCTWrapper(path_binary=MINDTCT_BIN_PATH)
cn_method = MinutiaeExtraction()

enh = GaborEnhancement()

# Resize image to match 500 ppi
image = cv2.resize(image, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_NEAREST)

while True:
    # blk_size = cv2.getTrackbarPos("blk", "window")
    weighing_factor = cv2.getTrackbarPos("c", "window") / 100

    # FJFX and MINDTCT receive as input an image with a slightly enhanced local contrast for best results
    imgcopy = image.copy()
    imgcopy[imgcopy == 255] = imgcopy[imgcopy < 255].mean()
    imgcopy = cv2.GaussianBlur(imgcopy, (11, 11), 0.8)
    img_enhanced = (normed(dog_filter(imgcopy, ksize=11, sigma=3.8)) * 255).astype(np.uint8)
    img_enhanced = cv2.addWeighted(image, weighing_factor, img_enhanced, 1 - weighing_factor, 0)

    t0 = time.time()
    minutiae_data_fjfx = fjfx.lib_wrapper(img_enhanced.copy(), 500)
    print("FJFX processing time:", time.time() - t0)

    t0 = time.time()
    minutiae_data_mindtct = mindtct.bin_wrapper(img_enhanced.copy(), contrast_enhancement=True)
    print("MINDTCT processing time:", time.time() - t0)

    # The implemented crossing number algorithm receives an image, enhanced with oriented Gabor filter and then binarized
    img_enhanced_gabor, binim = enh.enhance(image.copy())
    cv2.imshow("enhanced_image", normed(img_enhanced_gabor))
    cv2.imshow("enhanced_image_bin", binim.astype(float))
    t0 = time.time()
    minutiae_data_cn = cn_method.extract(binim.astype(np.uint8))
    print("CN method processing time:", time.time() - t0)

    if minutiae_data_fjfx:
        minu_data1 = visualize_minutiae(image.copy(), minutiae_data_fjfx, show_type=True, show_others=False)
        cv2.imshow("minutiae_data_fjfx", resized(minu_data1, 1))

    if minutiae_data_mindtct:
        minu_data1 = visualize_minutiae(image.copy(), minutiae_data_mindtct, show_type=True, show_others=False)
        cv2.imshow("minutiae_data_mindtct", resized(minu_data1, 1))

    if minutiae_data_cn:
        minu_data1 = visualize_minutiae(image.copy(), minutiae_data_cn, show_type=True, show_others=False)
        cv2.imshow("minutiae_data_cn", resized(minu_data1, 1))

    print("----------------")
    c = cv2.waitKey(1) & 0xFF
    if c == ord("1"):
        break
