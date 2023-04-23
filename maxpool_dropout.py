import torch
import torch.nn as nn

from util import load_cifar10, train_model, test_model

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(2021)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0,7),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=num_classes)
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear_layers(out)
        return out

# Variables
batch_size = 100
num_classes = 10
learning_rate = 0.001
num_epochs = 25

# Load data
train_loader, val_loader = load_cifar10(data_dir='./data', batch_size=batch_size)
test_loader = load_cifar10(data_dir='./data', batch_size=batch_size, test=True)

model = CNN(num_classes).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)

history = train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device)
test_acc = test_model(model, test_loader, device)
print("Accuracy with maxpool dropout: {:.4f}%".format(test_acc))


