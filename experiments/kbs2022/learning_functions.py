from matplotlib import pyplot as plt 
import numpy as np
import pickle
import torch
import cv2
import os


def load_data(location, labels_loc, metric):
    image_list = os.listdir(location)
    image_list = [f for f in image_list if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]

    if labels_loc is None:
        return [(image_filename, None) for image_filename in image_list]

    print("Number of images: " + str(len(image_list)))
    with open(labels_loc, "rb") as handle:
        labels = pickle.load(handle)

    data = []
    for i, image_filename in enumerate(image_list):
        fid = image_filename.split(".")[0]

        quality = 0
        if metric in labels[fid]:
            quality = [labels[fid][metric]]
        if metric == "all":
            quality = [0, 0, 0, 0]
            if "vfq" in labels[fid]:
                quality[0] = labels[fid]["vfq"]
            if "nfiq2" in labels[fid]:
                quality[1] = labels[fid]["nfiq2"]
            if "lqm" in labels[fid]:
                quality[2] = labels[fid]["lqm"]
            if "morpho" in labels[fid]:
                quality[3] = labels[fid]["morpho"]
        data.append((image_filename, quality))
    return data


class FingermarkDataset(torch.utils.data.Dataset):
    def __init__(self, db_location, labels, imsize):
        self.db_location = db_location
        self.imsize = imsize
        self.y = [x[1] for x in labels]
        self.x_ids = [x[0] for x in labels]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        im_name = self.x_ids[index]
        sample_location = self.db_location + im_name
        x = cv2.imread(sample_location, 0)
        x = cv2.resize(x, (self.imsize, self.imsize), interpolation=cv2.INTER_NEAREST)
        x = x.astype(np.float32) / 255
        x = np.expand_dims(x, 0)
        if self.y[index] is None:
            return x

        y = np.array([*self.y[index]], dtype=np.float32)
        return x, y