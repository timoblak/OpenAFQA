import numpy as np


def transform_data(m, theta):
    """Transforms 2D power spectrum to polar coordinate system"""
    thetadelta = np.pi / theta
    im_center = int(m.shape[0] // 2)

    angles = np.arange(0, np.pi, thetadelta+1e-5)
    radiuses = np.arange(0, im_center+1e-5)

    A, R = np.meshgrid(angles, radiuses)
    imx = R * np.cos(A) + im_center
    imy = R * np.sin(A) + im_center
    return m[imy.astype(int)-1, imx.astype(int)-1]


class FeatRPS:
    """Feature extraction for Radial Power Spectrum"""
    def __init__(self, blk_size=32, foreground_ratio=0.8):
        """Initialize

        :param blk_size: Size of individual blocks
        :param foreground_ratio : Ratio of minimal mask pixels to determine foreground
        """
        self.rad = 10
        self.theta = 100
        self.fmin = 0.06
        self.fmax = 0.18
        self.blk_size = blk_size
        self.foreground_ratio = foreground_ratio

    def rps(self, image):
        """Divides the input image into individual blocks and calculates the SF metric

        :param image: Input fingerprint image
        :param maskim: Input fingerprint segmentation mask
        :return: Resulting quality map in form of a matrix
        """
        r, c = image.shape  # r, c
        if r != c:
            d = max(r, c)

            imagepadded = np.ones(shape=(d, d), dtype=np.uint8) * 127
            cx = int(np.fix(d / 2 - c / 2))
            ry = int(np.fix(d / 2 - r / 2))
            imagepadded[ry:ry + r, cx:cx + c] = image
        else:
            imagepadded = image

        imdimension = max(imagepadded.shape)
        h = np.blackman(imagepadded.shape[0])

        filt = np.expand_dims(h, axis=1)
        filt = filt * filt.T

        imagewindowed = imagepadded * filt

        f_response = np.fft.fft2(imagewindowed)
        fs_response = np.fft.fftshift(f_response)

        power = np.log(1 + np.abs(fs_response))

        fsmin = np.max([np.floor(imdimension * self.fmin), 1])
        fsmax = np.min([np.ceil(imdimension * self.fmax), power.shape[0]])

        power_polar = transform_data(power, self.theta)

        roi_power_polar = power_polar[int(fsmin):int(fsmax) + 1]
        roi_power_sum = roi_power_polar.sum(axis=1)

        m = len(roi_power_sum) % self.rad
        if len(roi_power_sum) <= self.rad:
            radial_ps = roi_power_sum
        else:
            tmp = roi_power_sum[0:len(roi_power_sum) - m]
            radial_ps = np.reshape(tmp, newshape=(self.rad, int((len(roi_power_sum) - m) / self.rad))).sum(axis=1)
        return radial_ps.max()
