from afqa_toolbox.features import block_properties
import numpy as np
import cv2


class FeatSEP:
    """Feature extraction for Bimodal Separation"""
    def __init__(self, blk_size=32, foreground_ratio=0.8):
        """Initialize

        :param blk_size: Size of individual blocks
        :param foreground_ratio : Ratio of minimal mask pixels to determine foreground
        """
        self.blk_size = blk_size
        self.foreground_ratio = foreground_ratio

    def sep(self, image, maskim):
        """Divides the input image into individual blocks and calculates the SEP metric

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
                    result[br, bc] = self.sep_block(patch)
                bc += 1
            br += 1
            bc = 0

        return result

    @staticmethod
    def sep_block(block):
        """Calculates the SEP of a single image block

        :param block: An image block
        :return: calculated metric
        """
        # We use an inverse image when calculating s3pg and bimodal separation, since this is how original method does it
        block_inverse = cv2.bitwise_not(block)

        # Apply adaptive thresholding on the ROI to calculate masks for ridges and valleys
        ridges = cv2.adaptiveThreshold(block_inverse, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 0)
        valleys = 1 - ridges

        block_float = block_inverse.astype(np.float)
        valley_px = block_float[valleys.astype(np.bool)]
        ridges_px = block_float[ridges.astype(np.bool)]

        # We calculate: (mean(S) - mean(B)) / 2 * (std(B) + std(S))
        return (ridges_px.mean() - valley_px.mean()) / (2 * (ridges_px.std() + valley_px.std()))
