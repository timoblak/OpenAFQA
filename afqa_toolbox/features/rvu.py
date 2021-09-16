from afqa_toolbox.features import slanted_block_properties, covcoef, orient, get_rotated_block
import numpy as np
import cv2


class FeatRVU:
    """Feature extraction for Ridge Valley Uniformity"""
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

    def rvu(self, image, maskim):
        """Divides the input image into individual blocks and calculates the RVU metric

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
                if mask.sum()/self.blk_size**2 > self.foreground_ratio:
                    cova, covb, covc = covcoef(patch, "c_diff_cv")
                    orient_value = orient(cova, covb, covc)
                    blkwim = b_image[r - blk_offset:min(r + self.blk_size + blk_offset, rows),
                             c - blk_offset:min(c + self.blk_size + blk_offset, cols)]

                    result[br, bc] = self.rvu_block(blkwim, orient_value, self.v1sz_x, self.v1sz_y)

                bc += 1
            br += 1
            bc = 0
        return result

    @staticmethod
    def rvu_block(block, orientation, v1sz_x, v1sz_y, pad=False):
        """Calculates the RVU metric for a single image block

        :param block: An image block
        :param orientation: Angle of rotation of the friction ridge normal
        :param v1sz_x: Width of the slanted block size
        :param v1sz_y: Height of the slanted block size
        :param pad: Optinal padding when rotating block
        :return: A scalar metric for the input block
        """
        rows, cols = block.shape

        ic_block = rows // 2
        block_rotated = get_rotated_block(block, orientation, pad=pad)

        xoff, yoff = v1sz_x // 2, v1sz_y // 2

        block_cropped = block_rotated[ic_block - (yoff - 1) - 1:ic_block + yoff, ic_block - (xoff - 1) - 1:ic_block + xoff]

        # Get ridge-valley structure by using linear regression
        t = block_cropped.mean(axis=0, keepdims=True)

        x = np.arange(1, t.shape[1] + 1)  # From 1 to 31
        ones = np.ones(shape=(t.shape[1],))  # Only ones

        cat = np.stack([ones, x])
        dt1 = np.linalg.lstsq(cat.T, t.T, rcond=None)[0]
        dt = x * dt1[1] + dt1[0]
        ridval = (t < dt).T

        # Ridge-valley thickness
        change = np.bitwise_xor(ridval, np.roll(ridval, 1))
        change = change[1:]  # there can't be change in 1. element
        change_idx = np.where(change == 1)[0]  # Take indices of where change happens

        # If there were changes, we have a ridge-valley structure
        if change_idx.size > 0:

            ridval_complete = ridval[change_idx[0] + 1:change_idx[-1]]

            change_idx_complete = change_idx - change_idx[0]
            change_idx_complete = change_idx_complete[1:]

            if ridval_complete.size == 0:
                return np.nan

            # matlab: change_idx_complete(end: -1:2) = change_idx_complete(end: -1:2) - change_idx_complete(end - 1: -1:1);
            change_idx_complete[1:] = change_idx_complete[1:] - change_idx_complete[:-1]

            ratios = change_idx_complete[:-1] / change_idx_complete[1:]

            begrid = int(ridval_complete[0])
            ratios[begrid::2] = 1 / ratios[begrid::2]
            if ratios.size:
                return ratios.std()
        return np.nan


