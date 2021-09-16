from features import FeatStats, FeatGabor
import numpy as np
import cv2


def gabor_quality(self, image, thresh_poor=0.1, thresh_good=0.25, thresh_dry=180, thresh_smudge=170):
    """ Global Gabor qualities, proposed by Shen et al.

    :param image: Input fingerprint image
    :param thresh_poor: Lower bound for std, determining poor quality blocks
    :param thresh_good: Lower bound for std, determining good quality blocks
    :param thresh_dry: Lower bound for mean, determining dry blocks
    :param thresh_smudge: Upper bound for mean, determining smudged blocks
    :return: qi, si, di - overall quality of fingerprint, "smudginess", "dryness"
    """
    # sigma = 4
    # freq = 0.12
    # block_size = 30

    stds = self.gabor_stds(image, shen=True)
    means, _ = FeatStats(self.blk_size).stats(image, np.ones(shape=image.shape, dtype=int))

    foreground = np.zeros(shape=stds.shape, dtype=int)
    good = np.zeros(shape=stds.shape, dtype=int)
    dry = np.zeros(shape=stds.shape, dtype=int)
    smudged = np.zeros(shape=stds.shape, dtype=int)

    foreground[stds > thresh_poor] = 1
    good[stds > thresh_good] = 1
    poor = foreground - good

    dry[means >= thresh_dry] = 1
    dry = cv2.bitwise_and(dry, poor)
    smudged[means < thresh_smudge] = 1
    smudged = cv2.bitwise_and(smudged, poor)

    # stdsn = stds - stds.min()
    # stdsn = stdsn / stdsn.min()
    # cv2.imshow("gshs", cv2.resize(stdsn, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    # cv2.imshow("gshs_fg", cv2.resize(foreground.astype(float), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    # cv2.imshow("gshs_good", cv2.resize(good.astype(float), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    # cv2.imshow("gshs_poor", cv2.resize(poor.astype(float), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    # cv2.imshow("gshs_dry", cv2.resize(dry.astype(float), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    # cv2.imshow("gshs_smud", cv2.resize(smudged.astype(float), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    # cv2.waitKey(0)

    if foreground.sum():
        qi = 1 - (poor.sum() / foreground.sum())
        si = smudged.sum() / foreground.sum()
        di = dry.sum() / foreground.sum()
        return qi, si, di
    return 0, 0, 0
