import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloaders import load_cifar10
from util import train_model, test_model 

# Device config
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(2021)

class CNN(nn.Module):
    # Srivastava et. al, 2014
    def __init__(self, num_classes, dropout=False):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv2
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Conv3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),            
        )
        if dropout:
            self.linear_layers = nn.Sequential(
                nn.Linear(in_features=1024, out_features=2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2048, out_features=2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2048, out_features=num_classes),
            )
        else:
            self.linear_layers = nn.Sequential(
                nn.Linear(in_features=1024, out_features=2048),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=2048),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=num_classes),
            )    
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.linear_layers(out)
        return out
    
    
# Variables
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10

# Load data
train_loader, val_loader = load_cifar10(data_dir='./data', batch_size=batch_size)
test_loader = load_cifar10(data_dir='./data', batch_size=batch_size, test=True)

# No Dropout

model = CNN(num_classes).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

history = train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device)
test_acc = test_model(model, test_loader, device)
print("Accuracy without dropout: {:.4f}%".format(test_acc))

# Dropout

dropout = True
model = CNN(num_classes, dropout).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

dropout_history = train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device)
test_acc = test_model(model, test_loader, device)
print("Accuracy with dropout: {:.4f}%".format(test_acc))

# Plotting
acc = history['acc']
val_acc = history['val_acc']
d_acc = dropout_history['acc']
d_val_acc = dropout_history['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Accuracy without dropout')
plt.plot(epochs, val_acc, label='Valdiation acc without dropout')
plt.plot(epochs, d_acc, label='Accuracy with dropout')
plt.plot(epochs, d_val_acc, label='Validation acc with dropout')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([50,100])
plt.legend(loc='lower right')
plt.title("Dropout Classification Accuracy Comparison")
plt.show()


