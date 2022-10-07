from afqa_toolbox.features import FeatStats, FeatGabor
import numpy as np
import cv2


class GaborQuality:
    """ Global Gabor qualities, proposed by Shen et al. """

    def __init__(self, blk_size, sigma=6, freq=0.1, angle_num=8):
        self.blk_size = blk_size
        self.sigma = sigma
        self.freq = freq
        self.angle_num = angle_num

    def predict(self, image, thresh_poor=0.1, thresh_good=0.25, thresh_dry=180, thresh_smudge=170):
        """
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

        stds = FeatGabor(self.blk_size, self.sigma, self.freq, self.angle_num).gabor_stds(image, shen=True)
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

        if foreground.sum():
            qi = 1 - (poor.sum() / foreground.sum())
            si = smudged.sum() / foreground.sum()
            di = dry.sum() / foreground.sum()
            return qi, si, di
        return 0, 0, 0


if __name__ == "__main__":
    BLK_SIZE = 32
    latent = "/home/oper/data/SD302/images/latent/png/00002344_4G_X_206_IN_D800_1118PPI_16BPC_1CH_LP14_1.png"
    image = cv2.imread(latent, 0)
    ppi = int(latent.split("\\")[-1].split("_")[6].replace("PPI", ""))

    # Resize image to match 500 ppi
    image = cv2.resize(image, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_NEAREST)
    #color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    q = GaborQuality(BLK_SIZE)

    qi, si, di = q.predict(image)
    print(qi, si, di)


