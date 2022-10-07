from afqa_toolbox.features import slanted_block_properties, covcoef, orient, get_rotated_block
import numpy as np
import cv2


class FeatFDA:
    """Feature extraction for Frequency Domain Analysis"""
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

    def fda(self, image, maskim):
        """Divides the input image into individual blocks and calculates the FDA metric

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
                    result[br, bc] = self.fda_block(blkwim, orient_value, self.v1sz_x, self.v1sz_y, pad=False)

                bc += 1
            br += 1
            bc = 0
        return result

    @staticmethod
    def fda_block(block, orientation, v1sz_x, v1sz_y, pad=False):
        """Calculates the FDA metric for a single image block

        :param block: An image block
        :param orientation: Angle of rotation of the friction ridge normal
        :param v1sz_x: Width of the slanted block size
        :param v1sz_y: Height of the slanted block size
        :param pad: Optinal padding when rotating block
        :return: A scalar metric for the input block
        """
        rows, cols = block.shape

        block_rotated = get_rotated_block(block, orientation + (np.pi / 2), pad=pad)

        xoff, yoff = v1sz_x // 2, v1sz_y // 2

        ic_block = rows // 2  # int
        block_cropped = block_rotated[ic_block - (xoff - 1) - 1:ic_block + xoff, ic_block - (yoff - 1) - 1:ic_block + yoff]

        t = block_cropped.mean(axis=1)

        dft = np.fft.fft(t.T)

        abs_dft = np.absolute(dft[1:])

        #plt.figure(1)
        #plt.clf()
        #plt.plot(t)

        #plt.figure(2)
        #plt.clf()
        #plt.plot(np.arange(len(abs_dft)), abs_dft, "o")
        #plt.vlines(np.arange(len(abs_dft)), [0], abs_dft, linestyles='dotted', lw=2)
        #plt.pause(0.001)

        f_max_idx = np.argmax(abs_dft[:len(abs_dft)//2])
        iqm_denom = np.sum(abs_dft[:len(abs_dft)//2])

        #print(f_max_idx, abs_dft[f_max_idx], iqm_denom)
        #print(f_max_idx)
        if f_max_idx == 0:
            return np.nan

        return ((abs_dft[f_max_idx] +
                0.3 * (abs_dft[f_max_idx - 1] +
                       abs_dft[f_max_idx + 1])) / iqm_denom)
