from afqa_toolbox.features import block_properties
import numpy as np
import cv2


def gabor_filter(theta, freq, sigma, shen=False):
    """Produces a Gabor filter based on the provided parameters

    :param theta: The angle of the filter
    :param freq: The frequency of the filter
    :param sigma: The standard deviation of the gaussian envelope
    :param shen: Alternative definition of the Gabor filter by Shen et al.
    :return:
    """
    # define range (add small eps to also include the last index
    range = np.arange(-2.5 * sigma, 2.5 * sigma + 1e-5)

    [x, y] = np.meshgrid(range, range)

    # Shen et al. define the Gabor filter a bit differently
    if shen:
        x1 = x * np.cos(theta) + y * np.sin(theta)
        y1 = -x * np.sin(theta) + y * np.cos(theta)
    else:
        x1 = x * np.sin(theta) + y * np.cos(theta)
        y1 = x * np.cos(theta) - y * np.sin(theta)

    return np.exp((-1/2) * ((x1 * x1) / (sigma * sigma) + (y1 * y1) / (sigma * sigma))) * \
         np.exp(1j * 2 * np.pi * freq * x1)


class FeatGabor:
    """Filters the imput image with differently oriented Gabor filters"""
    def __init__(self, blk_size, sigma=6, freq=0.1, angle_num=8):
        # Default values are suitable for fingerprint image of 500 ppi
        self.blk_size = blk_size
        self.sigma = sigma
        self.freq = freq
        self.angle_num = angle_num

    def gabor_stds(self, image, smooth=False, shen=False):
        """Calculates the standard deviation of responses to differently oriented Gab filters

        :param image: Input image
        :param angle_num: The number of angles in half circle for which Gabor filters will be calculated
        :return:
        """

        h, w = image.shape

        img_float = image.astype(np.float64)/255
        gauss_kernel_1 = cv2.getGaussianKernel(7, 1)
        gauss_kernel_4 = cv2.getGaussianKernel(25, 4)
        gauss_image = cv2.sepFilter2D(img_float, cv2.CV_64F, gauss_kernel_1, gauss_kernel_1)

        img_detail = img_float - gauss_image

        gauss_responses = np.zeros(shape=(h, w, self.angle_num))
        for i, angle in enumerate(range(self.angle_num)):
            theta = (np.pi*angle) / self.angle_num
            gf = gabor_filter(theta, self.freq, self.sigma, shen)

            # Calculate the response of Gabor filters
            response = cv2.filter2D(img_detail, cv2.CV_64F, gf.real) + 1j * cv2.filter2D(img_detail, cv2.CV_64F, gf.imag)
            magnitude = np.abs(response)

            # Calc Gauss of the Gabor magnitudes for smoothing
            if smooth:
                gauss_responses[:, :, i] = cv2.sepFilter2D(magnitude, cv2.CV_64F, gauss_kernel_4, gauss_kernel_4)
            else:
                gauss_responses[:, :, i] = magnitude

        std_local = gauss_responses.std(axis=-1, ddof=1)

        rows, cols = block_properties(image.shape, self.blk_size)
        return cv2.resize(std_local, (cols, rows), interpolation=cv2.INTER_AREA)


