import random, torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

def count(output, target):
    with torch.no_grad():
        predict = torch.argmax(output, 1)
        correct = (predict == target).sum().item()
        return correct
    
# CIFAR-10
    
# Returns data loaders for training, validation, and test data
def load_cifar10(data_dir, batch_size, random_seed=32, val_size=0.1, shuffle=True, test=False):
    all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                              std=[0.2023, 0.1994, 0.2010])])
    # Return test loader
    if test:
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=all_transforms, download=True)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader
    # Return train and val loaders
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=all_transforms, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=all_transforms, download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    return (train_loader, val_loader)

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