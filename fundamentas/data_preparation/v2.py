import torch 
import numpy as np 
from torch.utils.data import TensorDataset , DataLoader , random_split 

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_tensor = torch.as_tensor(X).float().view(-1, 1)
y_tensor = torch.as_tensor(y).float().view(-1 , 1)

#build 
dataset = TensorDataset(x_tensor , y_tensor)

#perform split
ratio = 0.8 
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train 
train_data , val_data = random_split(dataset , [n_train,n_val])


#build a loader each set
train_loader = DataLoader(
    dataset = train_data ,  
    batch_size = 16 , 
    shuffle = True
)

val_loader = DataLoader(
    dataset = val_data , 
    batch_size = 16 )
