import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# The code for training the fusion model

LABELS = "qualities.pkl"
# LABELS is the location of a dictionary, where ground truth scores for each fingermark image are stored:
# {
#   "fingermark1_img_name.png":
#   {
#       "nfiq2": 13,
#       "vfq": 44,
#       "lqm": 63,
#       "morpho": 24
#   },
#   "fingermark2_img_name.png": { ... },
#   ...
# }

DATASET = "feature_vector_dataset/train/"
# DATASET is the location of saved feature vectors
# The feature vectors can be generated with "calculate_feature_vectors.py"

pca = make_pipeline(PCA(n_components=1))

with open(LABELS, "rb") as handle:
    labels = pickle.load(handle)

filelist = os.listdir(DATASET)
scores = np.zeros((len(filelist), 4))
for i, filename in enumerate(filelist):
    label = filename.split(".")[0]
    for j, metric in enumerate(["nfiq2", "vfq", "lqm", "morpho"]):
        scores[i, j] = labels[label][metric] if metric in labels[label] else 0

y = pca.fit_transform(scores)
min_y, max_y = np.min(y), np.max(y)


for i in range(10):
    name = filelist[i]
    raw_pca_score = y[i]

    final_quality = (raw_pca_score - min_y) / (max_y - min_y) * 100
    print(name, scores[i], raw_pca_score, final_quality)

with open("pca_fusion_model.pkl", "wb") as handle:
    fusion_model = {"model": pca, "min": min_y, "max": max_y}
    pickle.dump(fusion_model, handle)
