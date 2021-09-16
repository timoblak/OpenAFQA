from afqa_toolbox.features import block_properties, covcoef, orient
import numpy as np
import cv2


def laplacian_of_gaussian_filter(sigma):
    """Creates a 2D LOG filter based on the input sigma value

    :param sigma: Sigma to paramtrize the Gaussian
    :return: 2D filter
    """
    n = np.ceil(sigma*6)
    y, x = np.ogrid[-n//2:n//2+1, -n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y)) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    return final_filter


class FeatRDI:
    """Feature extraction for Ridge Discontinuity Indicator"""
    def __init__(self, blk_size=32, foreground_ratio=0.8):
        """Initialize

        :param blk_size: Size of individual blocks
        :param foreground_ratio : Ratio of minimal mask pixels to determine foreground
        """
        self.blk_size = blk_size
        self.foreground_ratio = foreground_ratio

    def rdi(self, image, maskim):
        """Divides the input image into individual blocks and calculates the RDI metric

        :param image: Input fingerprint image
        :param maskim: Input fingerprint segmentation mask
        :return: Resulting quality map in form of a matrix
        """
        rows, cols = image.shape

        map_rows, map_cols = block_properties(image.shape, self.blk_size)

        result = np.full((map_rows, map_cols), np.nan, dtype=np.float64)

        # COMPUTE LOCAL FEATURES with a sliding window
        br, bc = 0, 0
        for r in range(0, rows - self.blk_size - 1, self.blk_size):
            for c in range(0, cols - self.blk_size - 1, self.blk_size):
                # print("Image from: ", r, min(r+blk_size, rows), c, min(c+blk_size, cols))
                patch = image[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]
                mask = maskim[r:min(r + self.blk_size, rows), c:min(c + self.blk_size, cols)]

                # If the accepted number of pixels in block is foreground
                if mask.sum()/self.blk_size**2 > self.foreground_ratio:
                    result[br, bc] = self.rdi_block(patch)

                bc += 1
            br += 1
            bc = 0
        return result

    @staticmethod
    def rdi_block(block):
        # Calculates response for one sigma on image block (for dot detection)
        filter_log = laplacian_of_gaussian_filter(1.6)  # 500 ppi
        block_response = cv2.filter2D(block.astype(np.float64), -1, filter_log)  # convolving imag
        lap_block = np.square(block_response)  # squaring the response
        return lap_block.max()


"""
lap_block1 = cv2.Laplacian(block, cv2.CV_64F, ksize=3, scale=2**(2+4-3*2))
    lap_block2 = cv2.Laplacian(block, cv2.CV_64F, ksize=5, scale=2**(2+4-5*2))
    lap_block3 = cv2.Laplacian(block, cv2.CV_64F, ksize=7, scale=2**(2+4-7*2))

 if whole_image:
        # Calculates response for different sigma on whole image (for scale detection)
        maxes = []
        imgs = []
        for sigma in np.arange(1, 8, 0.6):
            filter_log = laplacian_of_gaussian_filter(sigma)
            image = cv2.filter2D(block.astype(np.float64), -1, filter_log)  # convolving imag
            lap_block = np.square(image)  # squaring the response
            maxes.append(lap_block.max())
            imgs.append(lap_block)

        responses = np.hstack(imgs)
        suma = np.sum(imgs, axis=0)

        plt.subplot(2,1,1)
        plt.imshow(responses, vmin=0, vmax=300)
        plt.subplot(2, 1, 2)
        plt.imshow(suma)
        response = np.max(maxes)

"""
