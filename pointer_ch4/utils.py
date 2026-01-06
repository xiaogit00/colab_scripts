import torch.optim as optim

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer