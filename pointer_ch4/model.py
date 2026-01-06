from torch import nn
from torchvision import models

def build_model(num_classes=2, freeze_backbone=True):
    transfer_model = models.resnet50(weights='DEFAULT')

    if freeze_backbone:
        for param in transfer_model.parameters():
            param.requires_grad = False

    transfer_model.fc = nn.Sequential(
        nn.Linear(transfer_model.fc.in_features, 500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(500, num_classes)
    )

    return transfer_model