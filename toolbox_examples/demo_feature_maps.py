import cv2
import numpy as np
from afqa_toolbox.features import FeatFDA, FeatLCS, FeatOCL, FeatRVU, FeatOFL, FeatStats, FeatGabor, \
    FeatRDI, covcoef, orient, slanted_block_properties, FeatS3PG, FeatSEP, FeatMOW, FeatACUT, FeatSF
from afqa_toolbox.tools import normed
from time import time


# Fingerprint
latent = "D:/NIST datasets/SD 302/sd302e/images/latent/png/00002344_4G_X_206_IN_D800_1118PPI_16BPC_1CH_LP14_1.png"
image = cv2.imread(latent, 0)

# Determine PPI
ppi = int(latent.split("\\")[-1].split("_")[6].replace("PPI", ""))
#ppi = 500
if ppi != 500:
    image = cv2.resize(image, None, fx=500/ppi, fy=500/ppi, interpolation=cv2.INTER_NEAREST)
cv2.imshow("image", cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST))

# Determine constants
BLK_SIZE = 32
SL_BLK_SIZE_X = 32
SL_BLK_SIZE_Y = 16
ACCEPTED_FG_RATIO = 0.8

# Get block properties for sliding window processing
rows, cols = image.shape
blk_offset, map_rows, map_cols = slanted_block_properties(image.shape, BLK_SIZE, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)

# Get basic statistics
maskim = np.ones(shape=image.shape, dtype=int)
means, stds = FeatStats(BLK_SIZE).stats(image, maskim)

t0 = time()
# Determine ROI (based on local pixel STD
min_std = 13
_, maskim = cv2.threshold(stds.copy(), min_std, 1, cv2.THRESH_BINARY)
maskim = cv2.resize(maskim, (cols, rows), interpolation=cv2.INTER_NEAREST).astype(int)

# Prepare matrices for metrics
fdas = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
lcss = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
ocls = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
rvus = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
rdis = np.full((map_rows, map_cols), np.nan, dtype=np.float64)

s3pgs = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
seps = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
mows = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
acuts = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
sfs = np.full((map_rows, map_cols), np.nan, dtype=np.float64)

orientations = np.full((map_rows, map_cols), np.nan, dtype=np.float64)
mask_seg = np.zeros(shape=(map_rows, map_cols), dtype=int)

# COMPUTE LOCAL FEATURES with a sliding window

br, bc = 0, 0
b_image = cv2.copyMakeBorder(image, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT, value=0)
b_maskim = cv2.copyMakeBorder(maskim, blk_offset, blk_offset, blk_offset, blk_offset, cv2.BORDER_CONSTANT, value=0)
for r in range(blk_offset, (blk_offset + rows) - BLK_SIZE - 1, BLK_SIZE):
    for c in range(blk_offset, (blk_offset + cols) - BLK_SIZE - 1, BLK_SIZE):

        patch = b_image[r:min(r + BLK_SIZE, rows), c:min(c + BLK_SIZE, cols)]
        mask = b_maskim[r:min(r + BLK_SIZE, rows), c:min(c + BLK_SIZE, cols)]

        # If the accepted number of pixels in block is foreground
        if mask.sum()/BLK_SIZE**2 > ACCEPTED_FG_RATIO:

            cova, covb, covc = covcoef(patch, "c_diff_cv")
            orient_value = orient(cova, covb, covc)
            orientations[br, bc] = orient_value
            mask_seg[br, bc] = 1

            ocls[br, bc] = 1 - FeatOCL.ocl_block(cova, covb, covc)
            rdis[br, bc] = FeatRDI.rdi_block(patch)

            s3pgs[br, bc] = FeatS3PG.s3pg_block(patch)
            seps[br, bc] = FeatSEP.sep_block(patch)
            mows[br, bc] = FeatMOW.mow_block(patch)
            acuts[br, bc] = FeatACUT.acut_block(patch)
            sfs[br, bc] = FeatSF.sf_block(patch)

            # For these features, a block with an additional offset/border is extracted
            # This is done since the block is rotated within these algorithms
            blkwim = b_image[r - blk_offset:min(r + BLK_SIZE + blk_offset, rows),
                             c - blk_offset:min(c + BLK_SIZE + blk_offset, cols)]

            fdas[br, bc] = FeatFDA.fda_block(blkwim, orient_value, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)

            lcss[br, bc] = FeatLCS.lcs_block(blkwim, orient_value, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)

            rvus[br, bc] = FeatRVU.rvu_block(blkwim, orient_value, SL_BLK_SIZE_X, SL_BLK_SIZE_Y)

        bc += 1
    br += 1
    bc = 0


# COMPUTE GLOBAL FEATURES
gab_feat = FeatGabor(BLK_SIZE, angle_num=8)
ofls = FeatOFL.ofl_blocks(orientations)

gabs = gab_feat.gabor_stds(image)

print("Compute time: ", time() - t0, "s")
cv2.imshow("fdas", cv2.resize(normed(fdas), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("lcss", cv2.resize(normed(lcss), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("ocls", cv2.resize(normed(ocls), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("ofls", cv2.resize(normed(ofls), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("rvus", cv2.resize(normed(rvus), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("gabs", cv2.resize(normed(gabs), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("rdis", cv2.resize(normed(rdis), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))

cv2.imshow("s3pgs", cv2.resize(normed(s3pgs), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("seps", cv2.resize(normed(seps), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("mows", cv2.resize(normed(mows), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("acuts", cv2.resize(normed(acuts), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("sfs", cv2.resize(normed(sfs), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("means", cv2.resize(normed(means), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
cv2.imshow("stds", cv2.resize(normed(stds), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))

print("Global features")
print("FDA:", fdas[~np.isnan(fdas)].mean(), fdas[~np.isnan(fdas)].std())
print("LCS:", lcss[~np.isnan(lcss)].mean(), lcss[~np.isnan(lcss)].std())
print("OCL:", ocls[~np.isnan(ocls)].mean(), ocls[~np.isnan(ocls)].std())
print("OFL:", 1 - ofls[~np.isnan(ofls)].mean(), 1 - ofls[~np.isnan(ofls)].std())
print("RVU:", rvus[~np.isnan(rvus)].mean(), rvus[~np.isnan(rvus)].std())
print("STD:", stds[~np.isnan(stds)].mean(), stds[~np.isnan(stds)].std())
print("MU:", means[~np.isnan(means)].mean(), means[~np.isnan(means)].std())
print("GAB:", gabs[~np.isnan(fdas)].mean(), gabs[~np.isnan(fdas)].std())
print("RDI:", rdis[~np.isnan(rdis)].mean(), rdis[~np.isnan(rdis)].std())

cv2.waitKey(0)

