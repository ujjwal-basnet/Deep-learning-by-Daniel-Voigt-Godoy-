import torch 
import torch.nn as nn
import torch.optim as optim 

#define model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.01
#creae a model and sent to device
model = nn.Sequential(nn.Linear(1,1)).to(device)

optimizer = optim.SGD(model.parameters(), lr = lr)

#define mse
loss_fn = nn.MSELoss(reduction= 'mean')

def make_train_step(model , loss_fn , optimizer):
    def train_step(x , y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat , y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    return train_step

train_step_fn = make_train_step(model , loss_fn , optimizer)
        
#create a tranning step for our function
train_step_fn = make_train_step(model , loss_fn , optimizer)
