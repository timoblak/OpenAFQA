from afqa_toolbox.features import slanted_block_properties, covcoef, orient, get_rotated_block
import numpy as np
import cv2


class FeatLCS:
    """Feature extraction for Local Clarity Score"""
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

    def lcs(self, image, maskim):
        """Divides the input image into individual blocks and calculates the LCS metric

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
                    orient_value = orient(cova, covb, covc)
                    blkwim = b_image[r - blk_offset:min(r + self.blk_size + blk_offset, rows),
                             c - blk_offset:min(c + self.blk_size + blk_offset, cols)]

                    result[br, bc] = self.lcs_block(blkwim, orient_value, self.v1sz_x, self.v1sz_y, pad=False)
                bc += 1
            br += 1
            bc = 0
        return result

    @staticmethod
    def lcs_block(block, orientation, v1sz_x, v1sz_y, pad=False):
        """Calculates the LCS metric for a single image block

        :param block: An image block
        :param orientation: Angle of rotation of the friction ridge normal
        :param v1sz_x: Width of the slanted block size
        :param v1sz_y: Height of the slanted block size
        :param pad: Optinal padding when rotating block
        :return: A scalar metric for the input block
        """
        rows, cols = block.shape

        ic_block = rows // 2  # int
        block_rotated = get_rotated_block(block, orientation, pad=pad)

        xoff, yoff = v1sz_x // 2, v1sz_y // 2
        block_cropped = block_rotated[ic_block - (yoff - 1) - 1:ic_block + yoff, ic_block - (xoff - 1) - 1:ic_block + xoff]

        # Get ridge-valley structure by using linear regression
        t = block_cropped.mean(axis=0, keepdims=True)

        x = np.arange(1, t.shape[1]+1)
        ones = np.ones(shape=(t.shape[1],))
        cat = np.stack([ones, x])
        dt1 = np.linalg.lstsq(cat.T, t.T, rcond=None)[0]
        dt = x * dt1[1] + dt1[0]

        ridval = (t < dt).T

        # Ridge-valley thickness
        begrid = ridval[0]
        change = np.bitwise_xor(ridval, np.roll(ridval, 1))

        change = change[1:]  # there can't be change in 1. element
        change_idx = np.where(change == 1)[0]  # Take indices of where change happens

        # If there were changes, we have a ridge-valley structure
        if change_idx.size > 0:
            change1r = np.roll(change_idx, 1)
            change1r[0] = 0
            w_rv = change_idx - change1r  #ridge and valley

            w_rmax125 = 5.0 # max ridge for 125 ppi scanner
            w_vmax125 = 5.0 # max valley for 125 ppi scanner
            w_rmin = 3.0
            w_rmax = 10.0
            w_vmin = 2.0
            w_vmax = 10.0

            ppi = 500

            if begrid:
                w_r = w_rv[0::2]  # matlab "odd" indeces
                w_v = w_rv[1::2]  # matlab "even" indeces
            else:
                w_v = w_rv[0::2]  # matlab "odd" indeces
                w_r = w_rv[1::2]  # matlab "even" indeces

            r_scale_norm = (ppi / 125.0) * w_rmax125
            v_scale_norm = (ppi / 125.0) * w_vmax125
            nw_r = w_r / r_scale_norm
            nw_v = w_v / v_scale_norm

            nw_rmin = w_rmin / r_scale_norm
            nw_rmax = w_rmax / r_scale_norm
            nw_vmin = w_vmin / v_scale_norm  # Changed to v_scale_norm
            nw_vmax = w_vmax / v_scale_norm  # Changed to v_scale_norm
        else:  # NOT ridge / valley structure, skip computation
            return np.nan

        # Clarity test
        mu_nw_r = nw_r.mean()
        mu_nw_v = nw_v.mean()

        # cv2.imshow("asd", cv2.resize(block_cropped, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(0)

        if nw_rmin <= mu_nw_r <= nw_rmax and nw_vmin <= mu_nw_v <= nw_vmax:
            # ridge region
            ridge_idxs = ridval.T[0]
            ridmat = block_cropped[:, ridge_idxs]
            dtridmat = np.matmul(np.ones(shape=ridmat.shape), np.diag(dt[ridge_idxs]))  # dt1 tresh for each coloum of ridmat
            ridbad = ridmat >= dtridmat

            # valley region
            valley_idxs = np.logical_not(ridval.T[0])
            valmat = block_cropped[:, valley_idxs]
            dtvalmat = np.matmul(np.ones(shape=valmat.shape), np.diag(dt[valley_idxs]))  # dt1 tresh for each coloum of ridmat
            valbad = valmat < dtvalmat

            alpha = ridbad.mean()
            beta = valbad.mean()
            lcs = 1 - ((alpha + beta) / 2)
            return lcs
        else:
            return 0


