import torch
import torch.nn.functional as F

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        
        # Training phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            
            # Ensure targets are the right shape for loss function
            if len(targets.shape) > 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1)  # Remove extra dimension if present
            
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * inputs.size(0)
        
        training_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        num_correct = 0
        num_examples = 0
        
        with torch.no_grad():  # CRITICAL: Disable gradient computation
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                
                # Consistent target reshaping
                if len(targets.shape) > 1 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                
                output = model(inputs)
                loss = loss_fn(output, targets)
                valid_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(output, dim=1)
                correct = (preds == targets)
                num_correct += correct.sum().item()
                num_examples += targets.size(0)
        
        valid_loss /= len(val_loader.dataset)
        accuracy = num_correct / num_examples
        
        print(f'Epoch: {epoch}, Training Loss: {training_loss:.4f}, '
              f'Validation Loss: {valid_loss:.4f}, accuracy = {accuracy:.4f}')