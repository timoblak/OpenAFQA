import cv2
import numpy as np
from time import time
import os
import pickle
from afqa_toolbox.features import FeatFDA, FeatLCS, FeatOCL, FeatRVU, FeatOFL, FeatGabor, \
    FeatRDI, covcoef, orient, slanted_block_properties, FeatS3PG, FeatSEP, FeatMOW, FeatACUT, FeatSF
from afqa_toolbox.tools import visualize_minutiae, normed, resized
from afqa_toolbox.enhancement import dog_filter
from afqa_toolbox.minutiae.fjfx import FJFXWrapper
from afqa_toolbox.segmentation import hist_equalization_segmentation


IMAGE = "D:/NIST datasets/SD 302/sd302e/images/latent/png/00002304_1C_R_L01_BP_S22_1200PPI_8BPC_1CH_LP03_1.png"
MODEL = "./trained_models/rf_model_vfq.pkl"

# Initialize minutiae detector
minutiae_detect = FJFXWrapper(path_library="../../afqa_toolbox/minutiae/FJFX.dll")
# Load precomputed histogram bin edges for computing fixed feature vector
with open("./precalculated_edges.pkl", "rb") as handle:
    bin_edges = pickle.load(handle)
# Load trained quality assessment model
with open(MODEL, "rb") as handle:
    qa_model = pickle.load(handle)

# ### DEFINE CONSTANTS ###
# Sliding window properties
BLK_SIZE = 32
SL_BLK_SIZE_X = 32
SL_BLK_SIZE_Y = 16
# Block will be processed if the ratio between foreground and background pixels is larger than this threshold
ACCEPTED_FG_RATIO = 0.5
# The processed block will be considered as foreground if STD is more that the threshold
MIN_STD = 8
# Set if PPI is know in advance otherwise it is read from filename (if using NIST SD301 or SD302 fingermark datasets)
PRESET_PPI = None
# Set background color to mean value
BACKGROUND_MEAN = True
# Intermediate results are shown
SHOW = True

# ##############################################################
# ##################### FEATURE EXTRACTION #####################
image = cv2.imread(IMAGE, 0)
t0 = time()

# Determine PPI
if PRESET_PPI:
    ppi = PRESET_PPI
else:
    # PPI is read from filename in case NIST SD301 or SD302 fingermark datasets
    ppi = int(IMAGE.split("\\")[-1].split("_")[6].replace("PPI", ""))

# resize the fingermarks to 500 ppi
if ppi != 500:
    print("Resizing... ", ppi, 500 / ppi)
    image = cv2.resize(image, None, fx=500 / ppi, fy=500 / ppi, interpolation=cv2.INTER_NEAREST)

if BACKGROUND_MEAN:
    image[image == 255] = int(image.mean())

# Get block properties for sliding window processing
rows, cols = image.shape
blk_offset, map_rows, map_cols = slanted_block_properties(image.shape, BLK_SIZE, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)

# Initialize feature maps
results = np.full((map_rows, map_cols, 16), np.nan, dtype=np.float64)

# Segment the input image
segmentation_mask = hist_equalization_segmentation(image, threshold=127)
cv2.bitwise_and(image, image, mask=segmentation_mask)
maskim = segmentation_mask / 255

results[:, :, 0] = cv2.resize(maskim, dsize=(map_cols, map_rows), interpolation=cv2.INTER_NEAREST)

print("Computing local features..")
# ### COMPUTE LOCAL FEATURES with a sliding window ###
br, bc = 0, 0
b_image = cv2.copyMakeBorder(image, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT, value=0)
b_maskim = cv2.copyMakeBorder(maskim, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT, value=0)
for r in range(blk_offset, (blk_offset + rows) - BLK_SIZE - 1, BLK_SIZE):
    for c in range(blk_offset, (blk_offset + cols) - BLK_SIZE - 1, BLK_SIZE):

        patch = b_image[r:min(r + BLK_SIZE, rows), c:min(c + BLK_SIZE, cols)]
        mask = b_maskim[r:min(r + BLK_SIZE, rows), c:min(c + BLK_SIZE, cols)]

        # If the accepted number of pixels in block is foreground
        if results[br, bc, 0] == 1:  # mask.sum()/(BLK_SIZE**2) >= ACCEPTED_FG_RATIO:

            cova, covb, covc = covcoef(patch, "c_diff_cv")
            orient_value = orient(cova, covb, covc)
            results[br, bc, 1] = patch.mean()
            results[br, bc, 2] = patch.std()
            results[br, bc, 3] = orient_value

            results[br, bc, 4] = 1 - FeatOCL.ocl_block(cova, covb, covc)
            results[br, bc, 5] = FeatRDI.rdi_block(patch)

            results[br, bc, 6] = FeatS3PG.s3pg_block(patch)
            results[br, bc, 7] = FeatSEP.sep_block(patch)
            results[br, bc, 8] = FeatMOW.mow_block(patch)
            results[br, bc, 9] = FeatACUT.acut_block(patch)
            results[br, bc, 10] = FeatSF.sf_block(patch)
            # For these features, a block with an additional offset/border is extracted
            # This is done since the block is rotated within these algorithms

            blkwim = b_image[r - blk_offset:min(r + BLK_SIZE + blk_offset, rows),
                     c - blk_offset:min(c + BLK_SIZE + blk_offset, cols)]

            results[br, bc, 11] = FeatFDA.fda_block(blkwim, orient_value, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)
            results[br, bc, 12] = FeatLCS.lcs_block(blkwim, orient_value, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)
            results[br, bc, 13] = FeatRVU.rvu_block(blkwim, orient_value, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)

        bc += 1
    br += 1
    bc = 0

print("Computing global features..")
# ### COMPUTE GLOBAL FEATURES ###
results[:, :, 14] = FeatOFL.ofl_blocks(results[:, :, 2])
results[:, :, 15] = FeatGabor(BLK_SIZE, angle_num=8).gabor_stds(image)

print("Computing minutiae features..")
# ### COMPUTE MINUTIAE FEATURES ###

# First enhance the ridge structure using a Difference of Gaussians filter
image_copy = image.copy()
image_copy = cv2.GaussianBlur(image_copy, (11, 11), 0.8)
img1 = 255 - (dog_filter(image_copy, ksize=11, sigma=3.8) * 255).astype(np.uint8)
img1 = cv2.addWeighted(image, 0.5, img1, 1 - 0.5, 0)

# Calculate minutiae data
minutiae_data = minutiae_detect.lib_wrapper(img1, 500)

# Only use minutiae, detected within the segmented foreground
ok_minutiae = []
if len(minutiae_data["fingerprints"]) > 0:
    maskim_clr = cv2.cvtColor((maskim * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # print(len(minutiae_data["fingerprints"][0]["minutiae"]))
    for md in minutiae_data["fingerprints"][0]["minutiae"]:
        if maskim[md["miny"], md["minx"]] == 1:
            ok_minutiae.append(md)
            cv2.circle(maskim_clr, (md["minx"], md["miny"]), 4, (255, 0, 0), -1)
        else:
            cv2.circle(maskim_clr, (md["minx"], md["miny"]), 4, (0, 0, 255), -1)

    if SHOW:
        cv2.imshow("minutiae", resized(maskim_clr, 0.5))
        minu_visualized = visualize_minutiae(image.copy(), minutiae_data, mask=maskim, min_quality=0, show_others=False)
        cv2.imshow("minutiae_local_threshold", resized(minu_visualized, 1))

if SHOW:
    cv2.imshow("image", resized(image, 0.5))
    cv2.imshow("mask", cv2.resize(results[:, :, 0], None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("mean", cv2.resize(normed(results[:, :, 1]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("std", cv2.resize(normed(results[:, :, 2]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("orient", cv2.resize(normed(results[:, :, 3]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("ocl", cv2.resize(normed(results[:, :, 4]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("rdi", cv2.resize(normed(results[:, :, 5]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("s3pg", cv2.resize(normed(results[:, :, 6]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("pes", cv2.resize(normed(results[:, :, 7]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("mow", cv2.resize(normed(results[:, :, 8]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("acut", cv2.resize(normed(results[:, :, 9]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("sf", cv2.resize(normed(results[:, :, 10]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("fda", cv2.resize(normed(results[:, :, 11]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("lcs", cv2.resize(normed(results[:, :, 12]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("rvu", cv2.resize(normed(results[:, :, 13]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("ofl", cv2.resize(normed(results[:, :, 14]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    cv2.imshow("gab", cv2.resize(normed(results[:, :, 15]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
    c = cv2.waitKey(0) & 0xff


# ###################################################################
# ##################### FEATURE VECTOR CREATION #####################
feature_vector = []
# Iterate through the feature maps but skip the segmented mask
for i in range(1, 16):
    dat = results[:, :, i][np.bitwise_and(~np.isnan(results[:, :, i]), np.isfinite(results[:, :, i]))]

    print(dat)
    bin_edges["feature_maps"][i][0] = -np.inf
    bin_edges["feature_maps"][i][-1] = np.inf

    hist = np.histogram(dat, bins=bin_edges["feature_maps"][i])
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

hist = np.histogram(minutiae, bins=bin_edges["minutiae"])

mean_quality = 0 if len(minutiae) == 0 else np.mean(minutiae)
feature_vector.extend([mean_quality, hist[0].sum(), *hist[0]])

fingermark_quality = qa_model.predict([feature_vector])[0]

print("------------- DONE ----------------")
print("Feature vector:", feature_vector)
print("Compute time (preprocessing + feature extraction + inference): ", time() - t0, "seconds")
print("===== Predicted quality of input fingermark based on model " + MODEL + ": " + str(fingermark_quality) + "/100 =====")