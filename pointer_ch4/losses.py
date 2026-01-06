import torch.nn as nn

def get_loss(name, **kwargs):
    if name == "ce":
        return nn.CrossEntropyLoss(**kwargs)

    elif name == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)

    else:
        raise ValueError(f"Unknown loss: {name}")