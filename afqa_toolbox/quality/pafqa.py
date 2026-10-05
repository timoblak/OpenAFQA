import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import importlib.resources


class ClassificationHead(nn.Module):
    def __init__(self, in_features, outputs):
        super(ClassificationHead, self).__init__()
        self.out_layer = nn.Sequential(nn.Linear(in_features, outputs))

    def forward(self, x):
        x = self.out_layer(x)
        return torch.log_softmax(x, dim=-1)


class PAFQAModel(nn.Module):
    def __init__(self, outputs=100, fcn=256):
        super(PAFQAModel, self).__init__()
        self.outputs = outputs
        self.fcn = fcn

        self.encoder = models.resnet34(weights=None)

        # Grayscale input
        self.encoder.conv1.weight = nn.Parameter(torch.sum(self.encoder.conv1.weight, dim=1, keepdim=True))

        self.encoder.fc = nn.Sequential(
            nn.Linear(512, self.fcn), nn.LeakyReLU(),
            nn.Linear(self.fcn, self.fcn), nn.LeakyReLU()
        )
        self.output = ClassificationHead(self.fcn, self.outputs)

    def forward(self, x):
        x = self.encoder(x)
        return self.output(x)

    def load_weights(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])


class PAFQA:
    """ Implementation of the probabilistic fingermark quality assessment (pAFQA) approach from article:

        T. Oblak, R. Haraksim, L. Beslay, P. Peer.
        Probabilistic Fingermark Quality Assessment with Quality Region Localisation.
        Sensors, Volume 23, Issue 8, 2023

    The model predicts a probability distribution over 100 quality bins. The final quality is the expected value of
    this distribution, while its standard deviation can be used as a measure of the model's uncertainty.
    """
    RESOURCES_PATH = "afqa_toolbox.resources"
    DEFAULT_MODEL_PATH = "model_pafqa_resnet34.pt"
    QUALITY_BINS = np.arange(1, 101, dtype=np.float64)

    def __init__(self, model_path=None, device="cpu"):
        """Initialize

        :param model_path: Path to the pretrained pAFQA checkpoint (default: resources location)
        :param device: Torch device on which the model is executed (e.g. "cpu" or "cuda:0")
        """
        self.device = device
        self.imsize = 512

        model_path = self.check_default_location(model_path, self.DEFAULT_MODEL_PATH)

        self.model = PAFQAModel(outputs=len(self.QUALITY_BINS)).to(device)
        self.model.load_weights(model_path, device=device)
        self.model.eval()

    def check_default_location(self, given_path, default_path):
        if given_path is None:
            if importlib.resources.is_resource(self.RESOURCES_PATH, default_path):
                with importlib.resources.path(self.RESOURCES_PATH, default_path) as path:
                    return path
            raise FileNotFoundError("The path of one of the the required external files (" + default_path +
                                    ") was neither passed to the constructor nor was it found in the default resources location (" + self.RESOURCES_PATH + ")")
        return given_path

    def preprocess(self, input_image):
        """ Pads the image to a square shape with a white background and resizes it to the model input size
        :param input_image: Grayscale image
        :return: Input tensor of shape [1, 1, imsize, imsize]
        """
        if len(input_image.shape) != 2 or input_image.dtype != np.uint8:
            raise TypeError("The input image is expected to be a 2D array in 8 bit grayscale color.")

        h, w = input_image.shape
        diff = abs(h - w)
        db1, db2 = diff // 2, diff - diff // 2
        if h > w:
            x = cv2.copyMakeBorder(input_image, 0, 0, db1, db2, cv2.BORDER_CONSTANT, value=255)
        else:
            x = cv2.copyMakeBorder(input_image, db1, db2, 0, 0, cv2.BORDER_CONSTANT, value=255)

        x = cv2.resize(x, (self.imsize, self.imsize), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) / 255
        return torch.from_numpy(x)[None, None]

    def predict_distribution(self, input_image):
        """ Predicts the quality probability distribution of an input fingermark image
        :param input_image: Grayscale image
        :return: Array of 100 probabilities, corresponding to quality values 1-100
        """
        with torch.no_grad():
            log_probs = self.model(self.preprocess(input_image).to(self.device))
        return np.exp(log_probs[0].cpu().numpy().astype(np.float64))

    def predict(self, input_image, return_uncertainty=False):
        """ Predicts the quality of an input fingermark image
        :param input_image: Grayscale image
        :param return_uncertainty: Whether to also return the standard deviation of the predicted distribution
        :return: Quality value in range [1, 100] (and optionally the uncertainty of the prediction)
        """
        probabilities = self.predict_distribution(input_image)
        quality = float(np.sum(probabilities * self.QUALITY_BINS))
        if return_uncertainty:
            uncertainty = float(np.sqrt(np.sum(probabilities * (self.QUALITY_BINS - quality) ** 2)))
            return quality, uncertainty
        return quality
