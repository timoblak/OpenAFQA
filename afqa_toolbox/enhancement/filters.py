import cv2
import numpy as np
from afqa_toolbox.tools import normed


def dog_filter(image, ksize, sigma, log_enhance=True):
    """Unsharp mesking  local contrast enhancement

    :param image: Input image
    :param ksize: Size of Gaussian kernel
    :param sigma: Sigma parameter of the Gaussian kernel
    :param log_enhance: Size of Gaussian kernel
    :return: Locally enhanced image
    """
    image_blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    contrast_enhanced = image.astype(float) - image_blurred

    if log_enhance:
        contrast_enhanced = np.sign(contrast_enhanced) * np.log(1 + np.abs(contrast_enhanced))

    return contrast_enhanced


def magnitude_filter(image, fac):
    """The frequencies with lowest magnitude are removed from the image

    :param image: Input image
    :param fac: Magnitude threshold. All frequencies with magnitude < fac will be removed.
    :return: Filtered image
    """
    f_response = np.fft.fft2(image)
    fs_response = np.fft.fftshift(f_response)
  
    power = np.log(1 + np.abs(fs_response))
    power = normed(power)

    fs_response[power < fac] = 0
    power[power < fac] = 0

    f_response = np.fft.ifftshift(fs_response)
    backprojection = np.fft.ifft2(f_response)

    return backprojection.real, power
    


