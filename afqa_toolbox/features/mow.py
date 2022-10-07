from afqa_toolbox.features import block_properties
import numpy as np
import cv2


class FeatMOW:
    """Feature extraction for Mean Object Width"""
    def __init__(self, blk_size=32, foreground_ratio=0.8):
        """Initialize

        :param blk_size: Size of individual blocks
        :param foreground_ratio : Ratio of minimal mask pixels to determine foreground
        """
        self.blk_size = blk_size
        self.foreground_ratio = foreground_ratio

    def mow(self, image, maskim):
        """Divides the input image into individual blocks and calculates the MOW metric

        :param image: Input fingerprint image
        :param maskim: Input fingerprint segmentation mask
        :return: Resulting quality map in form of a matrix
        """
        rows, cols = image.shape

        map_rows, map_cols = block_properties(image.shape, self.blk_size)

        result = np.full((map_rows, map_cols), np.nan, dtype=np.float64)

        br, bc = 0, 0
        for r in range(0, rows - self.blk_size - 1, self.blk_size):
            for c in range(0, cols - self.blk_size - 1, self.blk_size):
                # print("Image from: ", r, min(r+blk_size, rows), c, min(c+blk_size, cols))
                patch = image[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]
                mask = maskim[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]

                # If the accepted number of pixels in block is foreground
                if mask.sum() / self.blk_size ** 2 > self.foreground_ratio:
                    result[br, bc] = self.mow_block(patch)
                bc += 1
            br += 1
            bc = 0

        return result

    @staticmethod
    def mow_block(block):
        """Calculates the MOW of a single image block

        :param block: An image block
        :return: calculated metric
        """
        ppi = 500
        block_area = block.shape[0] * block.shape[1]

        # We use an inverse image when calculating s3pg and bimodal separation, since this is how original method does it
        block_inverse = cv2.bitwise_not(block)

        # Apply adaptive thresholding on the ROI to calculate masks for ridges and valleys
        ridges = cv2.adaptiveThreshold(block_inverse, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)

        # fing external contours in image and fit ellipses to them
        #block_clr = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
        cnts, _ = cv2.findContours(ridges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        minor_axes = []
        for i, c in enumerate(cnts):
            # eliminate really small segments
            cnt_area = cv2.contourArea(c)
            cnt_perimeter = cv2.arcLength(c, True)
            if len(c) < 5 or cnt_perimeter == 0:
                continue

            circularity = 4 * np.pi * (cnt_area / (cnt_perimeter * cnt_perimeter))
            if block_area * 0.01 <= cnt_area <= block_area * 0.5 and 0.01 <= circularity <= 0.75:
                # save the minor axes of the ellipses
                ellipse = cv2.fitEllipse(c)
                #cv2.ellipse(block_clr, ellipse, (0, 0, 255), 1)
                minor_axes.append(ellipse[1][0])

        #cv2.imshow("block_clr", cv2.resize(block_clr, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

        # In what units are object widths measured?
        # The equation in ImageJ macro is: (mow_px/ppi)*100 - percentage of an inch?
        if len(minor_axes) > 0:
            return np.mean(minor_axes) / ppi * 100
        return np.nan
