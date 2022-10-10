import cv2
import numpy as np
from afqa_toolbox.features import FeatFDA, FeatLCS, FeatOCL, FeatRVU, FeatOFL, FeatStats, FeatGabor, \
    FeatRDI, slanted_block_properties, FeatS3PG, FeatSEP, FeatMOW, FeatACUT, FeatSF
from afqa_toolbox.tools import normed
from time import time


latent = ""  # Path to latent/fingermark image
image = cv2.imread(latent, 0)

# Get PPI from filename if part of the NIST SD 301/302 datasets, otherwise specify
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
fdas = FeatFDA(blk_size=BLK_SIZE, v1sz_x=SL_BLK_SIZE_X, v1sz_y=SL_BLK_SIZE_Y).fda(image, maskim)
lcss = FeatLCS(blk_size=BLK_SIZE, v1sz_x=SL_BLK_SIZE_X, v1sz_y=SL_BLK_SIZE_Y).lcs(image, maskim)
ocls = FeatOCL(blk_size=BLK_SIZE, v1sz_x=SL_BLK_SIZE_X, v1sz_y=SL_BLK_SIZE_Y).ocl(image, maskim)
ofls = FeatOFL(blk_size=BLK_SIZE, v1sz_x=SL_BLK_SIZE_X, v1sz_y=SL_BLK_SIZE_Y).ofl(image, maskim)
rvus = FeatRVU(blk_size=BLK_SIZE, v1sz_x=SL_BLK_SIZE_X, v1sz_y=SL_BLK_SIZE_Y).rvu(image, maskim)

rdis = FeatRDI(BLK_SIZE).rdi(image, maskim)
gabs = FeatGabor(BLK_SIZE).gabor_stds(image)

s3pgs = FeatS3PG(BLK_SIZE).s3pg(image, maskim)
seps = FeatSEP(BLK_SIZE).sep(image, maskim)
mows = FeatMOW(BLK_SIZE).mow(image, maskim)
acuts = FeatACUT(BLK_SIZE).acut(image, maskim)
sfs = FeatSF(BLK_SIZE).sf(image, maskim)

print("Compute time: ", time() - t0)

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
#print("GSH:", gsh)

cv2.waitKey(0)
#plt.show()
