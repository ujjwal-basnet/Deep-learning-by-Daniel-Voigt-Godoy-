import torch 
import torch.optim as optim 
import torch.nn as nn 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.optim as optim

def make_train_step_fn(model , optimizer , loss_fn):
    def train_step(x ,y ):
        model.train() 
        yhat = model(x)
        loss = loss_fn(yhat , y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    
    return train_step 

def make_val_step_fn(model , loss_fn):
    def perform_val_step(x ,y ):
        model.eval()
        yhat = model(x)
        loss = loss_fn(yhat , y)
        return loss.item()
    return perform_val_step
        

#set lr rate 
lr = 0.01
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(1 ,1 ).to(device))
optimizer = optim.SGD(model.parameters() , lr = 0.01)
loss_fn = nn.MSELoss()

train_step_fn = make_train_step_fn(model ,optimizer , loss_fn)
val_step_fn = make_val_step_fn(model , loss_fn)

#create a summary writter to interface with tensorboard 
writer = SummaryWriter('runs/simple_linear_regression')
#fetch single mini batch so we can use add_graph
x_dummy , y_dummy = next(iter(train_loader))
x_dummy = x_dummy.view(-1 , 1)
y_dummy = y_dummy.view(-1,1)
writer.add_graph(model , x_dummy.to(device))
