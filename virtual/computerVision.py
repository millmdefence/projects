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

class_names = train_data.classes

print(f"image shape: {image.shape}")
plt.imshow(image.squeeze())  # Colour channels x height x width 
plt.title(label);

### 2. Prepare the data loader

from torch.utils.data import DataLoader

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, # dataset to turn into iterable
    batch_size=BATCH_SIZE, # how many samples per batch? 
    shuffle=True # shuffle data every epoch?
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False # don't necessarily have to shuffle the testing data
)

# Let's check out what we've created
print(f"Dataloaders: {train_dataloader, test_dataloader}") 
print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
train_features_batch.shape, train_labels_batch.shape

## Build the Baseline model
# Need to flatten the model because its using a linear layer and linear layers only accept vectors as input.
# Create a flatten layer
flatten_model = nn.Flatten() # all nn modules function as a model (can do a forward pass)

# Get a single sample
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x) # perform forward pass

# Print out what happened
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")


from torch import nn
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential( # move through layer by layer 
            nn.Flatten(), # neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units), # in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape) # subsequent layers must line up with the previous layer's output (hidden_units) 
        )
    
    def forward(self, x): # Forward pass defines forward computation in the model.
        return self.layer_stack(x) # take an input x and pass it through the layer stack. 
    
torch.manual_seed(42)

# setup model with input parameters. 
model_0 = FashionMNISTModelV0(input_shape=784, # 28*28 pixels
                              hidden_units=10, # number of hidden units in the hidden layer (arbitrary)
                              output_shape=len(class_names) # number of output units (number of classes in the dataset)
).to("cpu")

model_0

### 3.1 Setup a loss function, optimizer and evaluation metrics
# Loss function = using Cross entropy loss because it's a multi-class classification problem.
# Optimizer = using Stochastic Gradient Descent (SGD) because it's a common optimization algorithm for training neural networks.
# Evaluation metric = using accuracy because it's a common metric for classification problems.

import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download.")
else:
    print("Downloading helper_functions.py...")
    url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py"
    response = requests.get(url)
    with open("helper_functions.py", "wb") as f:
        f.write(response.content)

from helper_functions import accuracy_fn

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss() # for multi-class classification problems
optimizer = torch.optim.SGD(params=model_0.parameters(), # which parameters to optimize
                            lr=0.1) # learning rate