import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import pickle
import importlib.resources
import os.path
import time

class RegressionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(RegressionHead, self).__init__()
        self.out_layer = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x):
        x = self.out_layer(x)
        return x


class AFQAModel(nn.Module):
    def __init__(self, outputs, fcn=256, pretrained=True):
        super(AFQAModel, self).__init__()
        # Parameters
        self.outputs = outputs
        self.fcn = fcn

        self.encoder = models.densenet121(pretrained)

        # Pretrained weights to grayscale
        self.encoder.features[0].weight = nn.Parameter(torch.sum(self.encoder.features[0].weight, dim=1, keepdim=True))

        self.encoder.classifier = nn.Sequential(
            nn.Linear(1024, self.fcn), nn.LeakyReLU(),
        )
        self.output = RegressionHead(self.fcn, self.outputs)

    def forward(self, x):
        # Graph
        features_vector = self.encoder(x)
        out = self.output(features_vector)
        return out

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])


class DeepEnsemble:
    """ Implementation of the deep learning ensemble approach from article:

        T. Oblak, R. Haraksim, P. Peer, L. Beslay.
        Fingermark quality assessment framework with classic and deep learning ensemble resources.
        Knowledge-Based Systems, Volume 250, 2022
    """
    RESOURCES_PATH = "afqa_toolbox.resources"
    DEFAULT_FUSION_MODEL_PATH = "pca_fusion_model.pkl"
    DEFAULT_DEEP_QUALITY_MODEL_PATH = "model_densenet121.pt"
    MODEL_NAMES = ["vfq", "nfq", "lqm", "mor"]

    def __init__(self, deep_model_path=None, pca_coeffs_path=None, device="cpu"):
        """Initialize

        :param deep_model_path: Size of individual blocks
        :param pca_coeffs_path: Width of slanted block
        :param device: Width of slanted block
        :return:
        """
        self.device = device
        self.imsize = 512

        # Check
        pca_coeffs_path = self.check_default_location(pca_coeffs_path, self.DEFAULT_FUSION_MODEL_PATH)
        deep_model_path = self.check_default_location(deep_model_path, self.DEFAULT_DEEP_QUALITY_MODEL_PATH)

        with open(pca_coeffs_path, "rb") as handle:
            self.pca_coeffs = pickle.load(handle)

        self.deep_model = AFQAModel(outputs=4, fcn=512).to(device)
        self.deep_model.load_weights(deep_model_path)

    def check_default_location(self, given_path, default_path):
        if given_path is None:
            if importlib.resources.is_resource(self.RESOURCES_PATH, default_path):
                with importlib.resources.path(self.RESOURCES_PATH, default_path) as path:
                    return path
            raise FileNotFoundError("The path of one of the the required external files (" + default_path +
                                    ") was neither passed to the constructor nor was it found in the default resources location (" + self.RESOURCES_PATH + ")")
        return given_path

    def predict_ensemble(self, input_image):
        """ Uses the pre-trained ensemble models to predict the quality of an input fingermark image
        :param input_image: Grayscale image
        :return: A dictionary of predictions from the quality assessment ensemble
        """
        if len(input_image.shape) != 2 or input_image.dtype != np.uint8:
            raise TypeError("The input image is expected to be a 2D array in 8 bit grayscale color.")

        x = cv2.resize(input_image, (self.imsize, self.imsize), interpolation=cv2.INTER_NEAREST)
        x = x.astype(np.float32) / 255
        x = np.expand_dims(x, 0)
        x = np.expand_dims(x, 0)

        pred_labels = self.deep_model(torch.from_numpy(x).to(self.device))
        predictions = pred_labels.squeeze(1).detach().cpu().numpy()[0]

        ensemble_predictions = {}
        for model, prediction in zip(self.MODEL_NAMES, predictions):
            ensemble_predictions[model] = prediction
        return ensemble_predictions

    def fusion(self, ensemble_predictions):
        """ Fuses the individual ensemble predictions into a single quality value
        :param ensemble_predictions: A dictionary of predictions from the quality assessment ensemble
        :return: Fused value
        """
        vfq = int(ensemble_predictions["vfq"])
        nfq = int(ensemble_predictions["nfq"])
        lqm = int(ensemble_predictions["lqm"])
        mor = int(ensemble_predictions["mor"])

        pca_transform = ((self.pca_coeffs["model"].transform([[nfq, vfq, lqm, mor]]) - self.pca_coeffs["min"]) / (
                    self.pca_coeffs["max"] - self.pca_coeffs["min"]))[0][0]

        fusion_quality = int(np.clip(pca_transform, a_min=0, a_max=100) * 100)
        return fusion_quality





