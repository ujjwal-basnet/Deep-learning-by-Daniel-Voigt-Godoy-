# Common Libraries
import random 
import numpy as np
import torch

# Image Augmentation 
from PIL import Image 
from image_classification import generate_dataset 

# Previously Defined Functions and Classes
from v1 import StepByStep 
from torch.utils.data import Dataset , DataLoader

# Data Utilities
from helpers import index_splitter, make_balanced_sampler
from torchvision.transforms.v2 import Compose,Normalize

#PyTorch Optimization and Loss Functions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


#data geneartion 


images, labels= generate_dataset(img_size=10, n_images=1000, binary=False, seed=17)

############## data preparation ######################### 
class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y= y
        self.transform= transform

    def __getitem__(self, index):
        x= self.x[index]

        if self.transform:
            x=self.transform(x)

        return x, self.y[index] 

    def __len__(self):
        return len(self.x)
    

####################### data preparationn ############################################
from helpers import index_splitter, make_balanced_sampler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
import torch

# Convert numpy arrays to PyTorch tensors and normalize pixel values
x_tensor = torch.as_tensor(images / 255.0).float()
y_tensor = torch.as_tensor(labels).long()

# Check tensor shapes
print(f"x_tensor shape: {x_tensor.shape}")
print(f"y_tensor shape: {y_tensor.shape}")

# Split indices into training and validation sets
train_idx, val_idx = index_splitter(len(x_tensor), [80, 20])

# Ensure indices are valid
print(f"train_idx length: {len(train_idx)}")
print(f"val_idx length: {len(val_idx)}")

# Apply the split to tensors
x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

# Define normalization transforms for training and validation sets
transform = Compose([Normalize(mean=(0.5,), std=(0.5,))])

# Create datasets with transformations applied
train_dataset = TransformedTensorDataset(x_train_tensor, y_train_tensor, transform=transform)
val_dataset = TransformedTensorDataset(x_val_tensor, y_val_tensor, transform=transform)

# Create a weighted random sampler to handle class imbalance
sampler = make_balanced_sampler(y_train_tensor)

# Create data loaders for training and validation sets
train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)



####################### model configuration ##########################################################
torch.manual_seed(42)
model_cnn1= nn.Sequential()



# Featurizer
# Block 1: 1@10x10 -> n_channels@8x8 -> n_channels@4x4
n_channels= 1
model_cnn1.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3))
model_cnn1.add_module('relu1', nn.ReLU())
model_cnn1.add_module('maxp1', nn.MaxPool2d(kernel_size=2))    
# Flattening: n_channels * 4 * 4
model_cnn1.add_module('flatten', nn.Flatten())

## classification 
#hidden layer
model_cnn1.add_module('fc1', nn.Linear(in_features=n_channels*4*4, out_features=10))
model_cnn1.add_module('relu2', nn.ReLU())
#output layer
model_cnn1.add_module('fc2', nn.Linear(in_features=10, out_features=3))



lr= 0.01
multi_loss_fn= nn.CrossEntropyLoss(reduction='mean')
optimizer_cnn1= optim.SGD(model_cnn1.parameters(),lr=lr)



################# model training #########################
sbs_cnn1 = StepByStep(model_cnn1, multi_loss_fn,optimizer_cnn1)
sbs_cnn1.set_loaders(train_loader, val_loader) 

sbs_cnn1.train(20)

fig= sbs_cnn1.plot_losses()