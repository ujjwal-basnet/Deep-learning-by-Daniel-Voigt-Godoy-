def mini_batch(device , train_loader , train_step_fn):
    mini_batch_losses = []
    for x_batch , y_batch in train_loader:
        x_batch = x_batch.to(device).view(-1 , 1)
        y_batch = y_batch.to(device).view(-1 , 1)

        mini_batch_loss = train_step_fn(x_batch , y_batch)
        mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_loss)
    return loss 

n_epcohs = 200 
losses = []
val_losses = [] 
for epcoh in range(n_epcohs):
    #inner loop 
    loss = mini_batch(device ,  train_loader , train_step_fn)
    losses.append(loss)

    with torch.no_grad():
        val_loss = mini_batch(device , val_loader , val_step_fn)
        val_losses.append(val_loss)
