from afqa_toolbox.enhancement import dog_filter, GaborEnhancement
from afqa_toolbox.minutiae import FJFXWrapper, MinutiaeExtraction, MINDTCTWrapper
from afqa_toolbox.tools import visualize_minutiae, resized, normed
import time
import numpy as np
import cv2
import os

# Windows
FJFX_LIB_PATH = "E:/Tim/doctorate/FingerJetFXOSE/FingerJetFXOSE/dist/Windows64/Release/FJFX.dll"
MINDTCT_BIN_PATH = "E:/Tim/doctorate/NBIS_5.0.0/mindtct/bin/mindtct"
dataset = "D:/NIST datasets/SD 302/sd302e/images/latent/png/"


cv2.namedWindow("window")
cv2.createTrackbar("c", "window", 11, 51, lambda x: x)

fjfx = FJFXWrapper(path_library=FJFX_LIB_PATH)
mindtct = MINDTCTWrapper(path_binary=MINDTCT_BIN_PATH)
cn_method = MinutiaeExtraction()

enh = GaborEnhancement()

i = 0
for fp_image in os.listdir(dataset):
    i += 1
    # Read image from dataset
    img = cv2.imread(dataset + fp_image, 0)
    
    # Get PPI from filename
    ppi = int(fp_image.split("\\")[-1].split("_")[6].replace("PPI", ""))

    # Resize image to match 500 ppi
    img = cv2.resize(img, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_NEAREST)

    while True:
        # blk_size = cv2.getTrackbarPos("blk", "window")
        f = cv2.getTrackbarPos("c", "window") / 100
        #sigmax = cv2.getTrackbarPos("gauss", "window") / 100
        #q = cv2.getTrackbarPos("quality", "window")

        # FJFX and MINDTCT receive as input an image with a slightly enhanced local contrast for best results
        imgcopy = img.copy()
        imgcopy[imgcopy == 255] = imgcopy[imgcopy < 255].mean()
        imgcopy = cv2.GaussianBlur(imgcopy, (11, 11), 0.8)
        img_enhanced = (normed(dog_filter(imgcopy, ksize=11, sigma=3.8)) * 255).astype(np.uint8)
        img_enhanced = cv2.addWeighted(img, f, img_enhanced, 1 - f, 0)


        t0 = time.time()
        minutiae_data_fjfx = fjfx.lib_wrapper(img_enhanced.copy(), 500)
        print("FJFX processing time:", time.time() - t0)

        t0 = time.time()
        minutiae_data_mindtct = mindtct.bin_wrapper(img_enhanced.copy(), contrast_enhancement=True)
        print("MINDTCT processing time:", time.time() - t0)

        # The implemented crossing number algorithm receives an image, enhanced with oriented Gabor filter and then binarized
        img_enhanced_gabor, binim = enh.enhance(img.copy())
        cv2.imshow("enhanced_image", normed(img_enhanced_gabor))
        cv2.imshow("enhanced_image_bin", binim.astype(float))
        t0 = time.time()
        minutiae_data_cn = cn_method.extract(binim.astype(np.uint8))
        print("CN method processing time:", time.time() - t0)

        if minutiae_data_fjfx:
            minu_data1 = visualize_minutiae(img.copy(), minutiae_data_fjfx, show_type=True, show_others=False)
            cv2.imshow("minutiae_data_fjfx", resized(minu_data1, 1))

        if minutiae_data_mindtct:
            minu_data1 = visualize_minutiae(img.copy(), minutiae_data_mindtct, show_type=True, show_others=False)
            cv2.imshow("minutiae_data_mindtct", resized(minu_data1, 1))

        if minutiae_data_cn:
            minu_data1 = visualize_minutiae(img.copy(), minutiae_data_cn, show_type=True, show_others=False)
            cv2.imshow("minutiae_data_cn", resized(minu_data1, 1))

        print("----------------")
        c = cv2.waitKey(1) & 0xFF
        if c == ord("d"):
            break
        elif c == ord("q"):
            exit()
