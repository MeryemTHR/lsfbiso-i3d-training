import torch
import gc

def train_model(
    model, criterion, optimizer, lr_scheduler, loader, device, batch_size, cumulation=1
):
    # Enable gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    epoch_loss = 0.0
    accuracy = 0
    total_samples = 0
    
    model.train().to(device)
    batch_idx = 0
    
    # Zero gradients at the beginning
    optimizer.zero_grad()
    
    for data in loader:
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")
        batch_idx += 1
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            # Clear cache to free memory
            torch.cuda.empty_cache()
            gc.collect()

        X, y = data
        current_batch_size = X.size(0)
        total_samples += current_batch_size
        
        # Ensure input is float32
        X = X.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Free memory
        del data
        
        # Use mixed precision for forward pass
        with torch.cuda.amp.autocast():
            output = model(X)
            
            # Handle I3D temporal dimension - average across time
            if output.dim() == 3:  # [batch_size, num_classes, time]
                # Average over the temporal dimension to get [batch_size, num_classes]
                output = output.mean(dim=2)
            
            loss = criterion(output, y)
        
        # Scale the loss and do backward pass
        scaler.scale(loss).backward()
        epoch_loss += loss.item() * current_batch_size
        
        _, preds = torch.max(output, 1)
        accuracy += torch.sum(preds == y.data)
        
        # Free memory
        del X, y, output, loss, preds
        
        # Cumulated gradient
        if batch_idx % cumulation == 0 or batch_idx == len(loader):
            # Update with scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
    
    epoch_loss = epoch_loss / total_samples
    train_acc = accuracy.double() / total_samples

    return epoch_loss, train_acc


def eval_model(model, criterion, loader, device, batch_size):
    eval_loss = 0
    eval_acc = 0
    total_samples = 0

    model.eval().to(device)
    batch_idx = 0
    
    for data in loader:
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")
        batch_idx += 1
        
        # Clear cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        X, y = data
        current_batch_size = X.size(0)
        total_samples += current_batch_size
        
        # Ensure input is float32
        X = X.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Free memory
        del data
        
        with torch.no_grad():
            # Use mixed precision
            with torch.cuda.amp.autocast():
                output = model(X)
                
                # Handle I3D temporal dimension - average across time
                if output.dim() == 3:  # [batch_size, num_classes, time]
                    # Average over the temporal dimension to get [batch_size, num_classes]
                    output = output.mean(dim=2)
                    
                loss = criterion(output, y)
                
            eval_loss += loss.item() * current_batch_size

            _, preds = torch.max(output, 1)
            eval_acc += torch.sum(preds == y.data)
            
        # Free memory
        del X, y, output, loss, preds
            
    eval_loss = eval_loss / total_samples
    eval_acc = eval_acc.double() / total_samples
    return eval_loss, eval_acc
