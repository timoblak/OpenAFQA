from afqa_toolbox.features import slanted_block_properties, covcoef, orient
import numpy as np
import cv2


class FeatOFL:
    """Feature extraction for Orientation Flow"""
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

        self.image, self.result, self.maskim = None, None, None

    def ofl(self, image, maskim):
        """Divides the input image into individual blocks and calculates the OFL metric

        :param image: Input fingerprint image
        :param maskim: Input fingerprint segmentation mask
        :return: Resulting quality map in form of a matrix
        """
        self.image = image
        self.maskim = maskim
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
                    result[br, bc] = orient(cova, covb, covc)
                bc += 1
            br += 1
            bc = 0
        self.result = self.ofl_blocks(result)
        return self.result

    @staticmethod
    def ofl_blocks(orientations, ang_min_deg=0):
        """Calculates the OFL metric based on precomputed block orientations

        :param orientations: Matrix of masked orientations, np.nan presents background
        :param ang_min_deg: Minimum angle determining change
        :return: A matrix of OFL letrics of same size that orientations
        """
        ang_diff = np.deg2rad(90 - ang_min_deg)
        ang_min = np.deg2rad(ang_min_deg)

        padded_orientation = cv2.copyMakeBorder(orientations, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=np.nan)

        loqall = np.full(orientations.shape, np.nan, dtype=np.float64)
        for i in range(1, orientations.shape[0] + 1):
            for j in range(1, orientations.shape[1] + 1):
                block_roi = padded_orientation[i - 1:i + 2, j - 1:j + 2]

                block_roi = block_roi[1, 1] - block_roi
                loq = np.abs(block_roi).sum() / 8

                if loq > ang_min and np.all(~np.isnan(block_roi)):
                    loqall[i - 1, j - 1] = (loq - ang_min) / ang_diff
        return loqall
