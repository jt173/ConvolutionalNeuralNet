import torch 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from util import load_cifar10, train_model, test_model

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(2021)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.Dropout2d(0.05),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=2)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=2)
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding='valid'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=2304, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear_layers(out)
        return out
    
# Variables
batch_size = 16
num_classes = 10
learning_rate = 0.001
num_epochs = 10

# Load data
train_loader, val_loader = load_cifar10(data_dir='./data', batch_size=batch_size)
test_loader = load_cifar10(data_dir='./data', batch_size=batch_size, test=True)

model = CNN(num_classes).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

history = train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device)
test_acc = test_model(model, test_loader, device)
print("Accuracy with spatial dropout: {:.4f}%".format(test_acc))