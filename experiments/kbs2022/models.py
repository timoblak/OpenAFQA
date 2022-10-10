import torch
import torch.nn as nn
from torchvision import models


def r2_score_torch(target, output):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class RegressionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(RegressionHead, self).__init__()
        self.out_layer = nn.Sequential(
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        x = self.out_layer(x)
        return x


class AFQAModel(nn.Module):
    def __init__(self, outputs, fcn=256, dropout=0, pretrained=True):
        super(AFQAModel, self).__init__()
        # Parameters
        self.outputs = outputs
        self.fcn = fcn
        self.dropout = dropout
        # Convolutions

        #self.encoder = models.resnet18(pretrained)
        #self.encoder = models.resnet34(pretrained)
        #self.encoder = models.resnet50(pretrained)
        # change output size to 2048

        #self.encoder = models.resnext50_32x4d(pretrained)
        # change output size to 2048

        #self.encoder = models.inception_v3(pretrained, transform_input=False, aux_logits=False)
        # change conv1 to Conv2d_1a_3x3.conv
        # change output size to 2048

        self.encoder = models.densenet121(pretrained)
        # change conv1 to features[0]
        # change fc to classifier
        # change output size to 1024

        #self.encoder = models.efficientnet_b0(pretrained)
        # change conv1 to features[0][0]
        # change fc to classifier
        # nn.Dropout(p=0.2) before Linear
        # change output size to 1280

        #self.encoder = models.efficientnet_b1(pretrained)
        # change conv1 to features[0][0]
        # change fc to classifier
        # nn.Dropout(p=0.3) before Linear
        # change output size to 1280
        
        #self.encoder = models.efficientnet_b2(pretrained)
        # change conv1 to features[0][0]
        # change fc to classifier
        # nn.Dropout(p=0.3) before Linear
        # change output size to 1536

        #self.encoder = models.efficientnet_b3(pretrained)
        # change conv1 to features[0][0]
        # change fc to classifier
        # nn.Dropout(p=0.3) before Linear
        # change output size to 1536

        #self.encoder = models.efficientnet_b4(pretrained)
        # change conv1 to features[0][0]
        # change fc to classifier
        # nn.Dropout(p=0.3) before Linear
        # change output size to 1536


        # Pretrained to grayscale
        self.encoder.features[0].weight = nn.Parameter(torch.sum(self.encoder.features[0].weight, dim=1, keepdim=True))
        
        #self.encoder.fc = nn.Identity()
        
        # !!PUT ACTIVATION LAYER BEFORE LAST LAYER!!

        self.encoder.classifier = nn.Sequential(
            #nn.Dropout(p=0.2),
            nn.Linear(1024, self.fcn), nn.LeakyReLU(),
            #nn.Linear(self.fcn, 256), nn.LeakyReLU(),
            #nn.Dropout(p=0.5)
        )

        #self.output = RegressionHead(self.fcn, self.outputs)
        self.output = RegressionHead(self.fcn, self.outputs)

    def forward(self, x):
        # Graph
        features_vector = self.encoder(x)
        out = self.output(features_vector)
        return out, features_vector


def save_model(path, epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_model(path, model, optimizer=None):
    print("Loading model: " + path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, model, optimizer, loss