from torch import nn
from torchvision import models
import torch.nn.functional as F

class PretrainModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        model_name = kwargs.get("model_name", "resnet18")
        freeze_weight = kwargs.get("freeze_weight", False)
        pretrained = kwargs.get("pretrained", True)

        if model_name == "resnet18":
            self.net = models.resnet18(pretrained=pretrained)
        elif model_name == "vgg16":
            self.net = models.vgg16(pretrained=pretrained)
        else:
            raise Exception("Model unsupported: {}".format(model_name))

        if freeze_weight:
            for param in self.net.parameters():
                param.requires_grad = False

        self.net.fc = nn.Linear(2048, 3)

    def forward(self, x):
        return self.net(x)

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)
