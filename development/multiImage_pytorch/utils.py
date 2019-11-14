import torch

def gamma_decode(images):
    return torch.pow(images, 2.2)

def gamma_encode(images):
    return torch.pow(images, 1.0/2.2)