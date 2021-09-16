import cv2
import os
import numpy as np
from afqa_toolbox.enhancement import GaborEnhancement, dog_filter, magnitude_filter
from afqa_toolbox.tools import normed, resized, visualize_minutiae, create_minutiae_record


if __name__ == "__main__":
    DATASET = "D:/NIST datasets/SD 301/sd301b/images/latent/png/"

    cv2.namedWindow("image")
    cv2.createTrackbar("dog_ksize", "image", 5, 55, lambda x: x)
    cv2.createTrackbar("dog_sigma", "image", 100, 1000, lambda x: x)
    cv2.createTrackbar("magnitude_factor", "image", 50, 100, lambda x: x)
    enh = GaborEnhancement()

    for id_per in os.listdir(DATASET):
        while True:
            image = cv2.imread(DATASET + id_per, 0)
            ppi = int(id_per.split("\\")[-1].split("_")[6].replace("PPI", ""))
            image = cv2.resize(image, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_NEAREST)

            cv2.imshow("image", image)

            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            enhanced_gabor, binim = enh.enhance(image)
            cv2.imshow("gabor_enhanced", normed(enhanced_gabor))
            cv2.imshow("gabor_binary", binim.astype(float))

            ksize = cv2.getTrackbarPos("dog_ksize", "image")
            ksize = ksize+1 if ksize % 2 == 0 else ksize
            sigma = cv2.getTrackbarPos("dog_sigma", "image") / 100

            enhanced_dog = dog_filter(image, ksize, sigma, log_enhance=True)

            f = cv2.getTrackbarPos("magnitude_factor", "image") / 100
            enhanced_mag, _ = magnitude_filter(image, f)

            cv2.imshow("enhanced_mag", normed(enhanced_mag))
            cv2.imshow("enhanced_dog", normed(enhanced_dog))
            c = cv2.waitKey(5) & 0xFF
            if c == ord("q"):
                break
