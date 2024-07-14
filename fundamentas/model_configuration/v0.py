
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.01 
torch.manual_seed(42)
#create a model and send to device
model = nn.Sequential(nn.Linear(1 , 1)).to(device)

## define sgd optimizer to udpate the parameters 
optimizer = torch.optim.SGD(model.parameters() , lr = lr)

#define mse as loss function 
loss_fn = nn.MSELoss(reduction= 'mean')
