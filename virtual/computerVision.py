import torch
from torch import nn

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt

# Setup training data
train_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
    target_transform=None
)
# Setup testing data
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
    target_transform=None
)   

image, label = train_data[0]

print(f"image shape: {image.shape}")
plt.imshow(image.squeeze())  # Colour channels x height x width 
plt.title(label);