import torch
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary


def has_mps():
    return torch.backends.mps.is_available()

def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def imshow(img, title):
    img = denormalize(img)
    npimg = img.numpy()
    fig = plt.figure(figsize=(15, 7))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def show_model_summary(model, input_size, device):
    print(summary(model, input_size, device=device))

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
    single_img = False
    if tensor.ndimension() == 3:
        single_img = True
        tensor = tensor[None, :, :, :]
    
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')
    
    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    ret = tensor.mul(std).add(mean)
    return ret[0] if single_img else ret


def count(output, target):
    with torch.no_grad():
        predict = torch.argmax(output, 1)
        correct = (predict == target).sum().item()
        return correct
    

def train_model(model, optimizer, loss_function, train_loader, val_loader, epochs, device):
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    for epoch in range(epochs):
        # Train and evaluate on the training set
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            yhat = model(X)
            loss = loss_function(yhat, y)

            # Backward anmd optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += count(yhat, y)
            train_total += y.size(0)
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader.dataset)
        history['acc'].append(train_acc)
        history['loss'].append(train_loss)

        # Evaluate on validation set
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                yhat = model(X)
                loss = loss_function(yhat, y)

                val_loss += loss.item() * X.size(0)
                val_correct += count(yhat, y)
                val_total += y.size(0)
            
            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader.dataset)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(loss.item())

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Val Loss: {:.4f}, Val Accuracy: {:.4f}%'
              .format(epoch+1, epochs, train_loss, train_acc, val_loss, val_acc))
    
    return history

def test_model(model, test_loader, device):
    test_correct = 0
    test_total = 0
    model.eval()

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            yhat = model(X)

            test_correct += count(yhat, y)
            test_total += y.size(0)

    test_acc = 100 * test_correct / test_total 
    return test_acc