
torch.manual_seed(42)
n_epochs = 100 
for epoch in range(n_epochs):
    #set the model to train
    model.train()

    #compute the model prediction
    yhat = model(x_train_tensor)

    #step-2 loss 
    loss = loss_fn(yhat , y_train_tensor)

    #compute gradients 
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
