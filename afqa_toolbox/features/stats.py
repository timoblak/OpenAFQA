from afqa_toolbox.features import block_properties
import numpy as np


class FeatStats:
    """Feature extraction for basic pixel statistics"""
    def __init__(self, blk_size=32, foreground_ratio=0.8):
        """Initialize

        :param blk_size: Size of individual blocks
        :param foreground_ratio : Ratio of minimal mask pixels to determine foreground
        """
        self.blk_size = blk_size
        self.foreground_ratio = foreground_ratio

    def stats(self, image, maskim):
        """Divides the input image into individual blocks and calculates the mean and std of pixels in block

        :param image: Input fingerprint image
        :param maskim: Input fingerprint segmentation mask
        :return: Resulting quality map in form of a matrix
        """
        rows, cols = image.shape

        map_rows, map_cols = block_properties(image.shape, self.blk_size)

        mean_per_block = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
        std_per_block = np.full((map_rows, map_cols), np.nan, dtype=np.float64)

        br, bc = 0, 0
        for r in range(0, rows - self.blk_size - 1, self.blk_size):
            for c in range(0, cols - self.blk_size - 1, self.blk_size):
                # print("Image from: ", r, min(r+blk_size, rows), c, min(c+blk_size, cols))
                patch = image[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]
                mask = maskim[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]

                # If the accepted number of pixels in block is foreground
                if mask.sum() / self.blk_size ** 2 > self.foreground_ratio:
                    mean, std = self.stats_block(patch)
                    mean_per_block[br, bc] = mean
                    std_per_block[br, bc] = std
                bc += 1
            br += 1
            bc = 0

        return mean_per_block, std_per_block

    @staticmethod
    def stats_block(block):
        """Calculates the mean and standard deviation of a single image block

        :param block: An image block
        :return: Mean and std of the input block
        """
        return block.mean(), block.std()
