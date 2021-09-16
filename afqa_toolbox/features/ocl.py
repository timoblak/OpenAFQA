from afqa_toolbox.features import slanted_block_properties, covcoef
import numpy as np
import cv2


class FeatOCL:
    """Feature extraction for Orientation Clarity Level"""
    def __init__(self, blk_size=32, v1sz_x=32, v1sz_y=16, foreground_ratio=0.8):
        """Initialize

        :param blk_size: Size of individual blocks
        :param v1sz_x: Width of slanted block
        :param v1sz_y: Height of slanted block
        """
        self.blk_size = blk_size
        self.v1sz_x = v1sz_x
        self.v1sz_y = v1sz_y
        self.foreground_ratio = foreground_ratio

    def ocl(self, image, maskim):
        """Divides the input image into individual blocks and calculates the OCL metric

        :param image: Input fingerprint image
        :param maskim: Input fingerprint segmentation mask
        :return: Resulting quality map in form of a matrix
        """
        rows, cols = image.shape

        blk_offset, map_rows, map_cols = slanted_block_properties(image.shape, self.blk_size, self.v1sz_x, self.v1sz_y)

        result = np.full((map_rows, map_cols), np.nan, dtype=np.float64)

        # COMPUTE LOCAL FEATURES with a sliding window
        br, bc = 0, 0
        b_image = cv2.copyMakeBorder(image, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT, value=0)
        b_maskim = cv2.copyMakeBorder(maskim, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT, value=0)
        for r in range(blk_offset, (blk_offset + rows) - self.blk_size - 1, self.blk_size):
            for c in range(blk_offset, (blk_offset + cols) - self.blk_size - 1, self.blk_size):
                patch = b_image[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]
                mask = b_maskim[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]

                # If the accepted number of pixels in block is foreground
                if mask.sum() / self.blk_size ** 2 > self.foreground_ratio:
                    cova, covb, covc = covcoef(patch, "c_diff_cv")
                    result[br, bc] = 1 - self.ocl_block(cova, covb, covc)
                bc += 1
            br += 1
            bc = 0
        return result

    @staticmethod
    def ocl_block(a, b, c):
        """Calculates the OCL metric based on covariances of a local image block

        :param a: Image covariance a
        :param b: Image covariance b
        :param c: Image covariance c
        :return: A scalar metric for the input block
        """
        eigvmax = ((a + b) + np.sqrt((a - b)*(a - b) + 4 * c * c)) / 2
        eigvmin = ((a + b) - np.sqrt((a - b)*(a - b) + 4 * c * c)) / 2
        return eigvmin / eigvmax  # [1(worst) - 0(best)]
