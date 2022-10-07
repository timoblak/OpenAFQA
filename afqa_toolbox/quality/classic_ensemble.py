import pickle
import sys

import cv2
import numpy as np
from afqa_toolbox import features as feat
from afqa_toolbox.segmentation import hist_equalization_segmentation
from afqa_toolbox.enhancement import dog_filter
from afqa_toolbox.minutiae import FJFXWrapper
import importlib.resources


class ClassicEnsemble:
    """ Implementation of the classic ensemble approach from article:

        T. Oblak, R. Haraksim, P. Peer, L. Beslay.
        Fingermark quality assessment framework with classic and deep learning ensemble resources.
        Knowledge-Based Systems, Volume 250, 2022

    """
    RESOURCES_PATH = "afqa_toolbox.resources"

    DEFAULT_CLASSIC_QUALITY_MODEL_PATH = "models_randomforest.pkl"
    DEFAULT_FUSION_MODEL_PATH = "pca_fusion_model.pkl"
    DEFAULT_MINDET_PATHS_WIN_PATH = "FJFX.dll"
    DEFAULT_MINDET_PATHS_LIN_PATH = "libFJFX.so"
    DEFAULT_HISTOGRAM_EDGES_PATH = "precalculated_edges.pkl"

    # Feature extraction sliding window settings
    BLK_SIZE = 32
    SL_BLK_SIZE_X = 32
    SL_BLK_SIZE_Y = 16

    # The processed block will be considered as foreground if STD is more that the threshold
    MIN_STD = 8

    # Each block will be processed only if the ratio between foreground and background pixels is larger than this threshold
    ACCEPTED_FG_RATIO = 0.5

    def __init__(self, ensemble_models_path=None, pca_coeffs_path=None, bin_edges_path=None, mindet_lib_path=None):
        # Load ensemble
        ensemble_models_path = self.check_default_location(ensemble_models_path, self.DEFAULT_CLASSIC_QUALITY_MODEL_PATH)
        pca_coeffs_path = self.check_default_location(pca_coeffs_path, self.DEFAULT_FUSION_MODEL_PATH)
        bin_edges_path = self.check_default_location(bin_edges_path, self.DEFAULT_HISTOGRAM_EDGES_PATH)

        if sys.platform.startswith("win32"):
            mindet_lib_path = self.check_default_location(mindet_lib_path, self.DEFAULT_MINDET_PATHS_WIN_PATH)
        elif sys.platform.startswith("linux"):
            mindet_lib_path = self.check_default_location(mindet_lib_path, self.DEFAULT_MINDET_PATHS_LIN_PATH)

        with open(ensemble_models_path, "rb") as handle:
            self.ensemble_models = pickle.load(handle)

        with open(pca_coeffs_path, "rb") as handle:
            self.pca_coeffs = pickle.load(handle)

        with open(bin_edges_path, "rb") as handle:
            self.bin_edges = pickle.load(handle)

        self.minutiae_detector = FJFXWrapper(path_library=mindet_lib_path)

        self.background_color = True

    def check_default_location(self, given_path, default_path):
        if given_path is None:
            if importlib.resources.is_resource(self.RESOURCES_PATH, default_path):
                with importlib.resources.path(self.RESOURCES_PATH, default_path) as path:
                    return path
            raise FileNotFoundError("The path of one of the the required external files (" + default_path +
                                    ") was neither passed to the constructor nor was it found in the default resources location (" +
                                    self.RESOURCES_PATH + ")")
        return given_path

    def feature_vector(self, input_image):
        """ Preprocessin, feature extraction, and feature vector creation for a given fingermark image
        :param input_image: Grayscale image
        :return: A vector of features extracted from the fingermark image
        """
        if len(input_image.shape) != 2 or input_image.dtype != np.uint8:
            raise TypeError("The input image is expected to be a 2D array in 8 bit grayscale color.")

        if self.background_color:
            input_image[input_image == 255] = int(input_image.mean())

        # Get block properties for sliding window processing
        rows, cols = input_image.shape
        blk_offset, map_rows, map_cols = feat.slanted_block_properties(input_image.shape, self.BLK_SIZE, self.SL_BLK_SIZE_X, self.SL_BLK_SIZE_Y)

        # Initialize feature maps
        results = np.full((map_rows, map_cols, 16), np.nan, dtype=np.float64)

        # Segment the input image
        segmentation_mask = hist_equalization_segmentation(input_image, threshold=127)
        cv2.bitwise_and(input_image, input_image, mask=segmentation_mask)
        maskim = segmentation_mask / 255

        results[:, :, 0] = cv2.resize(maskim, dsize=(map_cols, map_rows), interpolation=cv2.INTER_NEAREST)

        # ### COMPUTE LOCAL FEATURES with a sliding window ###
        br, bc = 0, 0
        b_image = cv2.copyMakeBorder(input_image, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT,
                                     value=0)
        b_maskim = cv2.copyMakeBorder(maskim, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT,
                                      value=0)
        for r in range(blk_offset, (blk_offset + rows) - self.BLK_SIZE - 1, self.BLK_SIZE):
            for c in range(blk_offset, (blk_offset + cols) - self.BLK_SIZE - 1, self.BLK_SIZE):
                patch = b_image[r:min(r + self.BLK_SIZE, rows), c:min(c + self.BLK_SIZE, cols)]

                # If the accepted number of pixels in block is foreground
                if results[br, bc, 0] == 1:
                    cova, covb, covc = feat.covcoef(patch, "c_diff_cv")
                    orient_value = feat.orient(cova, covb, covc)
                    results[br, bc, 1] = patch.mean()
                    results[br, bc, 2] = patch.std()
                    results[br, bc, 3] = orient_value

                    results[br, bc, 4] = 1 - feat.FeatOCL.ocl_block(cova, covb, covc)
                    results[br, bc, 5] = feat.FeatRDI.rdi_block(patch)

                    results[br, bc, 6] = feat.FeatS3PG.s3pg_block(patch)
                    results[br, bc, 7] = feat.FeatSEP.sep_block(patch)
                    results[br, bc, 8] = feat.FeatMOW.mow_block(patch)
                    results[br, bc, 9] = feat.FeatACUT.acut_block(patch)
                    results[br, bc, 10] = feat.FeatSF.sf_block(patch)

                    # For these features, a block with an additional offset/border is extracted
                    # This is done since the block is rotated within these algorithms
                    blkwim = b_image[r - blk_offset:min(r + self.BLK_SIZE + blk_offset, rows),
                             c - blk_offset:min(c + self.BLK_SIZE + blk_offset, cols)]

                    results[br, bc, 11] = feat.FeatFDA.fda_block(blkwim, orient_value, self.SL_BLK_SIZE_X, self.SL_BLK_SIZE_Y)
                    results[br, bc, 12] = feat.FeatLCS.lcs_block(blkwim, orient_value, self.SL_BLK_SIZE_X, self.SL_BLK_SIZE_Y)
                    results[br, bc, 13] = feat.FeatRVU.rvu_block(blkwim, orient_value, self.SL_BLK_SIZE_X, self.SL_BLK_SIZE_Y)

                bc += 1
            br += 1
            bc = 0

        # ### COMPUTE GLOBAL FEATURES ###
        results[:, :, 14] = feat.FeatOFL.ofl_blocks(results[:, :, 2])
        results[:, :, 15] = feat.FeatGabor(self.BLK_SIZE, angle_num=8).gabor_stds(input_image)

        # ### COMPUTE MINUTIAE FEATURES ###
        # First enhance the ridge structure using a Difference of Gaussians filter
        image_copy = input_image.copy()
        image_copy = cv2.GaussianBlur(image_copy, (11, 11), 0.8)
        img1 = 255 - (dog_filter(image_copy, ksize=11, sigma=3.8) * 255).astype(np.uint8)
        img1 = cv2.addWeighted(input_image, 0.5, img1, 1 - 0.5, 0)

        # Calculate minutiae data
        minutiae_data = self.minutiae_detector.lib_wrapper(img1, 500)

        # Only use minutiae, detected within the segmented foreground
        ok_minutiae = []
        if len(minutiae_data["fingerprints"]) > 0:
            for md in minutiae_data["fingerprints"][0]["minutiae"]:
                if maskim[md["miny"], md["minx"]] == 1:
                    ok_minutiae.append(md)

        # ###################################################################
        # ##################### FEATURE VECTOR CREATION #####################
        feature_vector = []
        # Iterate through the feature maps but skip the segmented mask
        for i in range(1, 16):
            dat = results[:, :, i][np.bitwise_and(~np.isnan(results[:, :, i]), np.isfinite(results[:, :, i]))]

            self.bin_edges["feature_maps"][i][0] = -np.inf
            self.bin_edges["feature_maps"][i][-1] = np.inf

            hist = np.histogram(dat, bins=self.bin_edges["feature_maps"][i])
            if len(dat) > 0:
                fv = [dat.mean(), dat.std(), *hist[0]]
            else:
                fv = [0, 0, *hist[0]]
            feature_vector.extend(fv)

        minutiae = []
        centroid = np.array([0, 0])
        for md in ok_minutiae:
            minutiae.append(md["minquality"])
            centroid[0] += md["miny"]
            centroid[1] += md["minx"]

        hist = np.histogram(minutiae, bins=self.bin_edges["minutiae"])

        mean_quality = 0 if len(minutiae) == 0 else np.mean(minutiae)
        feature_vector.extend([mean_quality, hist[0].sum(), *hist[0]])

        return feature_vector

    def predict_ensemble(self, feature_vector):
        """ Uses the pre-trained ensemble models to predict the quality of an input fingermark image from a feature vector
        :param feature_vector: A vector of features extracted from the fingermark image
        :return: A dictionary of predictions from the quality assessment ensemble
        """
        ensemble_predictions = {}
        for model in self.ensemble_models:
            self.ensemble_models[model].verbose = False
            q = self.ensemble_models[model].predict([feature_vector])[0]
            ensemble_predictions[model] = q
        return ensemble_predictions

    def fusion(self, ensemble_predictions):
        """ Fuses the individual ensemble predictions into a single quality value
        :param ensemble_predictions: A dictionary of predictions from the quality assessment ensemble
        :return: Fused value
        """
        vfq = int(ensemble_predictions["vfq"])
        lqm = int(ensemble_predictions["lqm"])
        mor = int(ensemble_predictions["mor"])
        nfq = int(ensemble_predictions["nfq"])

        pca_transform = ((self.pca_coeffs["model"].transform([[nfq, vfq, lqm, mor]]) - self.pca_coeffs["min"]) / (self.pca_coeffs["max"] - self.pca_coeffs["min"]))[0][0]

        fusion_quality = int(np.clip(pca_transform, a_min=0, a_max=100) * 100)
        return fusion_quality



