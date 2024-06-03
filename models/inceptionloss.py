import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional


class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.aux_logits = False
        self.model.eval()

    def forward(self, X):
        x_transform = torchvision.transforms.functional.resize(X, [299, 299])
        return self.model(x_transform)