n_epochs = 200 
losses = [] 


def mini_batch(device , train_loader , train_step_fn):
    mini_batch_losses = [] 
    for x_batch , y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = train_step_fn(x_batch , y_batch)
        mini_batch_losses.append(mini_batch_loss)
    loss =np.mean(mini_batch_loss)
    return loss 



for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader , train_step_fn)
    losses.append(loss)
