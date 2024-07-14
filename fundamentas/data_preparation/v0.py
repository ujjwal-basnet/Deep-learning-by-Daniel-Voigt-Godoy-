# data preparation 
# There hasn't been much data preparation up to this point.

import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x_train = np.linspace(1, 10, 100)
x_train_tensor = torch.as_tensor(x_train, device=device, dtype=torch.float).view(-1 ,1 )
y_train_tensor = x_train_tensor * 2 + 1 
