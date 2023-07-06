import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloaders import load_cifar10, load_cifar100, load_mnist, ImageNetConfig, ImagenetData
from util import train_model, test_model

# Device config
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class VGG16(nn.Module):
    def __init__(self, num_classes, in_channels=3, fc=False, maxpool=False, spatial=False):
        super(VGG16, self).__init__()

        # Conv1
        if spatial:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )
        elif maxpool:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        # Conv2
        if spatial:
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        elif maxpool:
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        # Conv3
        if spatial:
            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(256),
                nn.ReLU(),  
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)                
            )
        elif maxpool:
            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),  
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.conv3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),  
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # Conv4
        if spatial:
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        elif maxpool:
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # Conv5
        if spatial:
            self.conv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(0.5),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        elif maxpool:
            self.conv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else:
            self.conv5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        
        # FC
        if fc:
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 4096),
                nn.ReLU())
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5), 
                nn.Linear(4096, 4096),
                nn.ReLU())
        else:
            self.fc = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU())
            self.fc1 = nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU())
            
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out
    

def load_model(device, num_classes, in_channels, dropout_method=None):
    if dropout_method:
        if dropout_method == 'fc':
            model = VGG16(num_classes, in_channels, fc=True).to(device)
            return model
        elif dropout_method == 'mp':
            model = VGG16(num_classes, in_channels, maxpool=True).to(device)
            return model
        elif dropout_method == 'sp':
            model = VGG16(num_classes, in_channels, spatial=True).to(device)
            return model
    else:
        model = VGG16(num_classes, in_channels).to(device)
        return model
    
def main():
    # Setting up the model
    num_classes = 200
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.005
    in_channels = 3

    # Load data

    cutout = False

    train_loader, val_loader = load_cifar100(data_dir='./data', batch_size=batch_size, cutout=cutout)
    test_loader = load_cifar100(data_dir='./data', batch_size=batch_size, test=True)

    # Dropout method
    dropout_method = 'fc'
    model = load_model(device, num_classes, in_channels, dropout_method)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    history = train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device)
    test_acc = test_model(model, test_loader, device)
    print("Accuracy: {:.4f}%".format(test_acc))


    # Plotting
    acc = history['acc']
    val_acc = history['val_acc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label='Accuracy')
    plt.plot(epochs, val_acc, label='Valdiation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([50,100])
    plt.legend(loc='lower right')
    plt.title("Dropout Classification Accuracy")
    plt.show()


if __name__ == '__main__':
    main()