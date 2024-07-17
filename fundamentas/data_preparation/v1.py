import torch 
import numpy as np
from torch.utils.data import TensorDataset , DataLoader

x_train = np.linspace(1 , 10 , 100)
x_train_tensor = torch.as_tensor(x_train).float().view(-1,1)
y_train = x_train *2 + 1 
y_train_tensor = torch.as_tensor(y_train).float().view(-1 , 1)

#build dataset
train_data = TensorDataset(x_train_tensor , y_train_tensor)


train_loader = DataLoader(
    dataset = train_data , 
    batch_size = 16 , 
    shuffle = True
)
