# IMPORT CODE
import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
from IPython import display
from PIL import Image
from sklearn.inspection import permutation_importance
import pickle


def load_data(location, labels, metric):
    fv_list = os.listdir(location)
    fv_nb = len(fv_list)

    with open(location + fv_list[0], "rb") as handle:
        fv0 = pickle.load(handle)

    with open(labels, "rb") as handle:
        labels = pickle.load(handle)

    print("Number of vectors: " + str(fv_nb) + " of length " + str(len(fv0)))

    data_x = np.zeros(shape=(fv_nb, len(fv0)))
    data_y = np.zeros(shape=(fv_nb,))
    for i, fv_filename in enumerate(fv_list):
        with open(location + fv_filename, "rb") as handle:
            fv = pickle.load(handle)

        fid = fv_filename.split(".")[0]

        data_y[i] = 0
        if metric in labels[fid]:
            data_y[i] = labels[fid][metric]

        data_x[i] = fv

    return data_x, data_y, fv_list


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

#VFQ
regr_vfq = RandomForestRegressor(n_estimators=750, max_depth=150, min_samples_split=2, min_samples_leaf=1, random_state=0, verbose=True, n_jobs=8)
regr_nfq = RandomForestRegressor(n_estimators=750, max_depth=150, min_samples_split=2, min_samples_leaf=1, random_state=0, verbose=True, n_jobs=8)
regr_lqm = RandomForestRegressor(n_estimators=750, max_depth=150, min_samples_split=2, min_samples_leaf=1, random_state=0, verbose=True, n_jobs=8)
regr_mor = RandomForestRegressor(n_estimators=750, max_depth=150, min_samples_split=2, min_samples_leaf=1, random_state=0, verbose=True, n_jobs=8)

metric = "nfiq2"
train_x, train_y, train_names = load_data(DATA_FOLDER + "train/", LABELS, metric)
test_x, test_y, test_names = load_data(DATA_FOLDER + "test/", LABELS, metric)
regr_nfq.fit(train_x, train_y)
predictions = np.clip(regr_nfq.predict(test_x), 0, 100)
print("-------- NFQ trained --------")
print("test MSE", mean_squared_error(test_y, predictions))
print("test MAE", mean_absolute_error(test_y, predictions))
print("test R2", r2_score(test_y, predictions))

results_classic = {"names": test_names, "nfq": (test_y, predictions)}

metric = "vfq"
train_x, train_y, train_names = load_data(DATA_FOLDER + "train/", LABELS, metric)
test_x, test_y, test_names = load_data(DATA_FOLDER + "test/", LABELS, metric)
regr_vfq.fit(train_x, train_y)
predictions = np.clip(regr_vfq.predict(test_x), 0, 100)
print("-------- VFQ trained --------")
print("test MSE", mean_squared_error(test_y, predictions))
print("test MAE", mean_absolute_error(test_y, predictions))
print("test R2", r2_score(test_y, predictions))
results_classic["vfq"] = (test_y, predictions)

metric = "lqm"
train_x, train_y, train_names = load_data(DATA_FOLDER + "train/", LABELS, metric)
test_x, test_y, test_names = load_data(DATA_FOLDER + "test/", LABELS, metric)
regr_lqm.fit(train_x, train_y)
predictions = np.clip(regr_lqm.predict(test_x), 0, 100)
print("-------- LQM trained --------")
print("test MSE", mean_squared_error(test_y, predictions))
print("test MAE", mean_absolute_error(test_y, predictions))
print("test R2", r2_score(test_y, predictions))
results_classic["lqm"] = (test_y, predictions)


metric = "morpho"
train_x, train_y, train_names = load_data(DATA_FOLDER + "train/", LABELS, metric)
test_x, test_y, test_names = load_data(DATA_FOLDER + "test/", LABELS, metric)
regr_mor.fit(train_x, train_y)
predictions = np.clip(regr_mor.predict(test_x), 0, 100)
print("-------- MOR trained --------")
print("test MSE", mean_squared_error(test_y, predictions))
print("test MAE", mean_absolute_error(test_y, predictions))
print("test R2", r2_score(test_y, predictions))
results_classic["mor"] = (test_y, predictions)


with open("trianed_classic_models.pkl", "wb") as handle:
    pickle.dump({"vfq": regr_vfq, "nfq": regr_nfq, "lqm": regr_lqm, "mor": regr_mor}, handle)

with open("results_classic.pkl", "wb") as handle:
    pickle.dump(results_classic, handle)
