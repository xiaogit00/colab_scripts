import torch.optim as optim
import torch

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return optimizer

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device