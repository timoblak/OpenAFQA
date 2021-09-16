from afqa_toolbox.features import block_properties
import numpy as np
from skimage.feature import peak_local_max


class FeatSF:
    """Feature extraction for Spatial Frequency"""
    def __init__(self, blk_size=32, foreground_ratio=0.8):
        """Initialize

        :param blk_size: Size of individual blocks
        :param foreground_ratio : Ratio of minimal mask pixels to determine foreground
        """
        self.blk_size = blk_size
        self.foreground_ratio = foreground_ratio

    def sf(self, image, maskim):
        """Divides the input image into individual blocks and calculates the SF metric

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
                    result[br, bc] = self.sf_block(patch)
                bc += 1
            br += 1
            bc = 0

        return result

    @staticmethod
    def sf_block(block):
        """Calculates the SF of a single image block

        :param block: An image block
        :return: calculated metric
        """
        mm_per_block = 2.54

        # Calculate the power spectrum in shift to center
        f_response = np.fft.fft2(block)
        fs_response = np.fft.fftshift(f_response)
        power = np.log(1 + np.abs(fs_response))
        power_normed = power - power.min()
        power_normed /= power_normed.max()

        # Get 3 highest peaks; 1st should be the DC value (image mean) and 2nd and 3rd are ridge frequency
        peaks = peak_local_max(power_normed, num_peaks=3)  # min_distance=2)

        # Get values of peaks and sort in ascending order
        freq_vals = [(pt[0], pt[1], power_normed[pt[0], pt[1]]) for pt in peaks]
        freq_vals = sorted(freq_vals, key=lambda x: x[2])
        #power_normed_clr = cv2.cvtColor((power_normed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # If no secondary peaks are detected, the DC value is taken by default.
        # This corresponds to taking image mean is no other frequency is present

        peak_y, peak_x = block.shape[0]//2, block.shape[1]//2
        if len(freq_vals) > 1:
            peak_y, peak_x = freq_vals[-2][0], freq_vals[-2][1]

        # The distance of secondary peak from center (radius) is the frequency of ridges
        # This frequency is for a window of size 0.1 inch, so to get frequency/mm (as stated in paper),
        # we divide by 2.54

        # Visualize
        #power_normed_clr[peak_y, peak_x] = (0, 0, 255)
        #cv2.imshow("power-clr", cv2.resize(power_normed_clr, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        #cv2.waitKey(0)
        return np.sqrt((block.shape[0] / 2 - peak_y) ** 2 + (block.shape[1] / 2 - peak_x) ** 2) / mm_per_block




