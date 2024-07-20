import torch 
import torch.nn as nn
import torch.optim as optim 
lr = 0.01
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1,1))
optimizer = optim.SGD(model.parameters() , lr = lr )
loss_fn  =  nn.MSELoss(reduction = 'mean')
