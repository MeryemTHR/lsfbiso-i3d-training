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
        
        # Handle I3D temporal dimension - average across time
        if output.dim() == 3:  # [batch_size, num_classes, time]
            # Average over the temporal dimension to get [batch_size, num_classes]
            output = output.mean(dim=2)
        
        loss = criterion(output, y)
        loss.backward()
        epoch_loss += loss.item()

        _, preds = torch.max(output, 1)
        accuracy += torch.sum(preds == y.data)

        # Cumulated gradient
        if batch_idx % cumulation == 0 or batch_idx == len(loader):
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

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
            
            # Handle I3D temporal dimension - average across time
            if output.dim() == 3:  # [batch_size, num_classes, time]
                # Average over the temporal dimension to get [batch_size, num_classes]
                output = output.mean(dim=2)
                
            loss = criterion(output, y)
            eval_loss += loss.item()

            _, preds = torch.max(output, 1)
            eval_acc += torch.sum(preds == y.data)

    eval_loss = eval_loss / len(loader)
    eval_acc = eval_acc.double() / (len(loader) * batch_size)
    return eval_loss, eval_acc
