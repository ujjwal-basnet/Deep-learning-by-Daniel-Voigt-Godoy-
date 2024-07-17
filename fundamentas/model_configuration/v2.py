import torch 
import torch.optim as optim 
import torch.nn as nn 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.01 #set learning rate 
model = nn.Sequential(nn.Linear(1,1).to(device))
optimizer = optim.SGD(model.parameters() , lr = lr)
loss_fn = nn.MSELoss(reduction= 'mean')


def make_train_step_fn(model , loss_fn , optimizer):
    def train_step(x ,y):
        model.train()
        yhat = model(x)
        loss = loss_fn(yhat , y)
        loss.backward() #calcualte gradients 
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

train_step_fn = make_train_step_fn(model , loss_fn , optimizer)

def make_val_step_fn(model , loss_fn):
    def perform_val_step(x ,y ):
        model.eval() #model evaluation 
        yhat = model(x)
        loss = loss_fn(yhat , y)
        return loss.item()
    return perform_val_step


val_step_fn = make_val_step_fn(model , loss_fn)
