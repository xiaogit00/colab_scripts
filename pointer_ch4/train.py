import torch
import torch.nn.functional as F

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0 # each epoch has their own training loss.. 
        valid_loss = 0.0 # ..and validation loss
        model.train() # set it to training  mode
        for batch in train_loader: # to find how many batches, train_data_loader.__len__()
            optimizer.zero_grad() # what this does beneath the hood is that it loops through all the params of model and sets them to grad 0
            inputs, targets = batch # inputs is matrix of (B, C, W, H), targets (B, 1)
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0) # loss for this batch * batch_size
            
        # by this step, training_loss is the total loss of entire epoch
        training_loss /= len(train_loader.dataset) # divided by the num data points to find unit loss 
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0) #validation loss for this batch
            preds = torch.argmax(output, dim=1)
            targets = targets.view(-1).long()
            correct = (preds == targets) #compare all prediction in batch to ground truth
            num_correct += torch.sum(correct).item() # num correct in batch
            num_examples += correct.shape[0] #examples in batch
        #at this point, valid_loss is total loss of entire epoch
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples))
