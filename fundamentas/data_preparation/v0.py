
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train = np.linspace(1, 10, 100)
x_train_tensor = torch.as_tensor(x_train, device=device, dtype=torch.float).view(-1 ,1 )
y_train = x_train * 2 + 1 
y_train_tensor = torch.as_tensor(y_train , device = device  , dtype = torch.float).view(-1 , 1 )
