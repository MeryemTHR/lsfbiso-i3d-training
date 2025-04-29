import torch


def train_model(
    model, criterion, optimizer, lr_scheduler, loader, device, batch_size, cumulation=1
):

    epoch_loss = 0.0
    accuracy = 0

    model.train().to(device)
    batch_idx = 0
    for data in loader:
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")
        batch_idx += 1

        X, y = data
        # Ensure input is float32
        X = X.to(device, dtype=torch.float32)
        y = y.to(device)

        output = model(X)
        
        # Check if we need to reshape the output or target
        if output.dim() == 2 and y.dim() == 1:
            # This is standard classification - output is [batch_size, num_classes], target is [batch_size]
            loss = criterion(output, y)
            _, preds = torch.max(output, 1)
            correct = (preds == y).sum()
        elif output.dim() > 2:
            # Handle the case for 3D or higher dimensional output
            b, c, *rest = output.size()
            output_reshaped = output.view(b, c, -1).mean(dim=2)  # Average across spatial dimensions
            loss = criterion(output_reshaped, y)
            _, preds = torch.max(output_reshaped, 1)
            correct = (preds == y).sum()
        else:
            # Unknown case, print shapes for debugging
            print(f"DEBUG - Output shape: {output.shape}, Target shape: {y.shape}")
            raise ValueError(f"Unexpected tensor shapes: output {output.shape}, target {y.shape}")
        
        loss.backward()
        epoch_loss += loss.item()

        # Cumulated gradient
        if batch_idx % cumulation == 0 or batch_idx == len(loader):
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

        accuracy += correct

    epoch_loss = epoch_loss / len(loader)
    train_acc = accuracy.double() / (len(loader) * batch_size)

    return epoch_loss, train_acc


def eval_model(model, criterion, loader, device, batch_size):
    eval_loss = 0
    eval_acc = 0

    model.eval().to(device)
    batch_idx = 0

    for data in loader:
        print(f"\rBatch : {batch_idx+1} / {len(loader)}", end="\r")
        batch_idx += 1

        X, y = data
        # Ensure input is float32
        X = X.to(device, dtype=torch.float32)
        y = y.to(device)

        with torch.set_grad_enabled(False):
            output = model(X)
            
            # Check if we need to reshape the output or target
            if output.dim() == 2 and y.dim() == 1:
                # This is standard classification - output is [batch_size, num_classes], target is [batch_size]
                loss = criterion(output, y)
                _, preds = torch.max(output, 1)
                correct = (preds == y).sum()
            elif output.dim() > 2:
                # Handle the case for 3D or higher dimensional output
                b, c, *rest = output.size()
                output_reshaped = output.view(b, c, -1).mean(dim=2)  # Average across spatial dimensions
                loss = criterion(output_reshaped, y)
                _, preds = torch.max(output_reshaped, 1)
                correct = (preds == y).sum()
            else:
                # Unknown case, print shapes for debugging
                print(f"DEBUG - Output shape: {output.shape}, Target shape: {y.shape}")
                raise ValueError(f"Unexpected tensor shapes: output {output.shape}, target {y.shape}")
                
            eval_loss += loss.item()
            eval_acc += correct

    eval_loss = eval_loss / len(loader)
    eval_acc = eval_acc.double() / (len(loader) * batch_size)
    return eval_loss, eval_acc
