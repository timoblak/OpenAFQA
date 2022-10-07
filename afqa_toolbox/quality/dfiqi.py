from afqa_toolbox.features import FeatS3PG, FeatSEP, FeatMOW, FeatACUT, FeatSF
import numpy as np
import time
import cv2


class DFIQI:
    # Input parameters for scoring functions as given in paper
    PARAMS = {
        "s3pg": (51.408, 4.134),
        "sep": (0.843, 0.147),
        "acut": (6.869, 0.532),
        "mow": (1.383, 0.391),
        "sp": (2.078, 0.397)
    }

    VALUE_COEFFS = [
        # Intercept, LQSsum, nFEAT
        [0, 0, 0],  # NV
        [-1.735636, 0.2769346, -0.05051632],  # VEO
        [-6.041873, 0.7257197, 0.49457174]  # VID
    ]

    COMPLEXITY_COEFFS = [
        # Intercept, LQSsum, nFEAT
        [3.325, -0.459, -0.100],  # Highly complex
        [0, 0, 0],  # Complex
        [-1.781, -0.025, 0.741]  # Non-complex
    ]

    DIFFICULTY_COEFFS = [
        # Intercept, LQSsum, nFEAT
        [0, 0, 0],  # High
        [-1.896, 0.125, 0.289],  # Medium
        [-3.071, -0.004, 0.965]  # Low
    ]

    YELLOWTH = 0.35
    REDTH = 0.20

    GCOEFF = 1
    YCOEFF = 2 / 3
    RCOEFF = 0

    def __init__(self):
        pass

    @staticmethod
    def _multinomial_logreg(feats, coeffs):
        assert len(feats) == len(coeffs[0])
        # First calculate the denominator in the equation
        denom = 0
        for coeff in coeffs:
            denom += np.exp(np.dot(feats, coeff))

        # Calculate class probabilities
        class_probs = []
        for coeff in coeffs:
            class_probs.append(np.exp(np.dot(feats, coeff)) / denom)
        return class_probs

    def value(self, sumLQS, nMINUT):
        class_p = self._multinomial_logreg([1, nMINUT, sumLQS], self.VALUE_COEFFS)
        return class_p[2] - class_p[0]

    def complexity(self, sumLQS, nMINUT):
        class_p = self._multinomial_logreg([1, nMINUT, sumLQS], self.COMPLEXITY_COEFFS)
        return class_p[2] - class_p[0]

    def difficulty(self, sumLQS, nMINUT):
        class_p = self._multinomial_logreg([1, nMINUT, sumLQS], self.DIFFICULTY_COEFFS)
        return class_p[2] - class_p[0]

    @staticmethod
    def _scoring_function_fx(measure, mean, stdev):
        return np.exp(-((measure - mean) ** 2) / (2 * stdev ** 2))

    @staticmethod
    def _scoring_function_gx(measure, mean, scale):
        return 1 / (1 + np.exp(- (measure - mean) / scale))

    def _local_quality_score(self, block):
        # ############ Calculate Signal Percent Pixels Per Grid (s3pg) ############
        # The percent of ridges in the ROI
        s3pg = FeatS3PG.s3pg_block(block)
        s3pg_fx = self._scoring_function_fx(s3pg, self.PARAMS["s3pg"][0], self.PARAMS["s3pg"][1])

        # ############ Calculate Bimodal Separation (sep) ############
        sep = FeatSEP.sep_block(block)
        sep_fx = self._scoring_function_fx(sep, self.PARAMS["sep"][0], self.PARAMS["sep"][1])

        # ############ Calculate Acutance (acut) ############
        acut = FeatACUT.acut_block(block)
        acut_fx = self._scoring_function_gx(acut, self.PARAMS["acut"][0], self.PARAMS["acut"][1])

        # ############ Calculate Mean Object Width (mow) ############
        mow = FeatMOW.mow_block(block)
        mow_fx = self._scoring_function_fx(mow, self.PARAMS["mow"][0], self.PARAMS["mow"][1])

        # ############ Calculate Spatial Frequency (sf) ############
        sf = FeatSF.sf_block(block)
        sf_fx = self._scoring_function_fx(sf, self.PARAMS["sp"][0], self.PARAMS["sp"][1])

        return (s3pg, sep, acut, mow, sf), (s3pg_fx, sep_fx, acut_fx, mow_fx, sf_fx)

    def predict(self, minutiae_data, return_lqs=True):
        lqs = []
        for minutia in minutiae_data:
            x, y = minutia
            h, w = image.shape[:2]
            delta = 25
            _, measure_fx = self._local_quality_score(image[max(y - delta, 0):min(y + delta, h), max(x - delta, 0):min(x + delta, w)])
            lqs.append(np.mean(measure_fx))

        v = self.value(np.sum(lqs), len(lqs))
        c = self.complexity(np.sum(lqs), len(lqs))
        d = self.difficulty(np.sum(lqs), len(lqs))

        # Returns global quality metrics
        if return_lqs:
            return v, d, c, lqs
        return v, d, c


def click_events(event, x, y, flags, param):
    global minutiae_data, display_resize

    if event == cv2.EVENT_LBUTTONUP:
        minutiae_data.append((int(x/display_resize), int(y/display_resize)))
    elif event == cv2.EVENT_RBUTTONUP:
        del minutiae_data[-1]


if __name__ == "__main__":
    latent = "/home/oper/data/SD302/images/latent/png/00002344_4G_X_206_IN_D800_1118PPI_16BPC_1CH_LP14_1.png"

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_events)
    cv2.createTrackbar("c", "image", 20, 40, lambda x: x)
    cv2.createTrackbar("x", "image", 200, 400, lambda x: x)
    cv2.createTrackbar("y", "image", 120, 290, lambda x: x)

    image = cv2.imread(latent, 0)
    ppi = int(latent.split("\\")[-1].split("_")[6].replace("PPI", ""))

    # Resize image to match 500 ppi
    image = cv2.resize(image, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_NEAREST)
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    minutiae_data = [
        [191, 313],
        [113, 291]
    ]
    q = DFIQI()
    display_resize = 2

    while True:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print("---------------------------")

        t0 = time.time()
        v, d, c, lqs = q.predict(minutiae_data, return_lqs=True)
        total_time = time.time() - t0

        feat_qual = 0
        for minutia, local_quality in zip(minutiae_data, lqs):
            x, y = minutia
            if local_quality >= q.YELLOWTH:
                m_clr = (0, 255, 0)
                feat_qual += q.GCOEFF
            elif q.REDTH <= local_quality < q.YELLOWTH:
                m_clr = (0, 255, 255)
                feat_qual += q.YCOEFF
            else:
                m_clr = (0, 0, 255)
                feat_qual += q.RCOEFF

            cv2.circle(color_img, (x, y), 3, m_clr, -1, lineType=cv2.LINE_AA)
            cv2.rectangle(color_img, (x - 25, y - 25), (x + 25, y + 25), (0, 255, 0), 1)

        print("LQSs: ", lqs, np.sum(lqs), len(lqs))
        print("V: ", v)
        print("C: ", c)
        print("D: ", d)

        print("FeatQual: ", feat_qual)
        print("Time: ", time.time() - t0)

        color_img = cv2.copyMakeBorder(color_img, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=0)
        txt = "V: " + str(round(v, 2)) + " | C: " + str(round(c, 2)) + "| D: " + str(round(d, 2))

        cv2.putText(color_img, txt, (10, color_img.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1,
                    lineType=cv2.LINE_AA)

        cv2.imshow("image", cv2.resize(color_img, None, fx=display_resize, fy=display_resize, interpolation=cv2.INTER_NEAREST))

        c = cv2.waitKey(1) & 0xff
        if c == ord("q"):
            break
        elif c == ord("p"):
            cv2.waitKey(0)

