from torch import nn
from torchvision import models
import torch

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transfer_model = transfer_model().to(device)

    return transfer_model