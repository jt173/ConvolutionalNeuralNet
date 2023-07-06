import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloaders import load_mnist
from util import train_model, test_model 

# Device config
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')



class CNN(nn.Module):
    def __init__(self, num_classes, in_channels, fc=False, mp=False, spatial=False):
        super(CNN, self).__init__()
        if mp:
            self.conv_layers = nn.Sequential(
                # Conv1
                nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=3),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Conv2
                nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2),
                # Conv3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=2),            
            )
        elif spatial:
            self.conv_layers = nn.Sequential(
                # Conv1
                nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=3),
                nn.Dropout2d(0.5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Conv2
                nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
                nn.Dropout2d(0.5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                # Conv3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.Dropout2d(0.5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),            
            )
        else:
            self.conv_layers = nn.Sequential(
                # Conv1
                nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Conv2
                nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                # Conv3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )

        if fc:
            self.linear_layers = nn.Sequential(
                # 256 for MNIST
                nn.Linear(in_features=256, out_features=2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2048, out_features=2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2048, out_features=num_classes),
            )
        else:
            self.linear_layers = nn.Sequential(
                nn.Linear(in_features=256, out_features=2048),
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
    
def main():
    # Variables
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 15
    in_channels=1

    # Load data
    train_loader, val_loader = load_mnist(data_dir='./data', batch_size=batch_size)
    test_loader = load_mnist(data_dir='./data', batch_size=batch_size, test=True)

    # Load model
    dropout_method = 'no_dropout'
    cutout = False

    if dropout_method == 'no_dropout':
        model = CNN(num_classes, in_channels).to(device)
    elif dropout_method == 'dropout':
        fc = True
        model = CNN(num_classes, in_channels, fc=fc).to(device)
    elif dropout_method == 'max_pool':
        mp = True
        model = CNN(num_classes, in_channels, mp=mp).to(device)
    elif dropout_method == 'spatial':
        spatial = True
        model = CNN(num_classes, in_channels, spatial=spatial).to(device)
    elif dropout_method == 'cutout':
        cutout = True
        model = CNN(num_classes, in_channels).to(device)
        train_loader, _ = load_mnist(data_dir='./data', batch_size=batch_size, cutout=cutout)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Train
    history = train_model(model, optimizer, loss_function, train_loader, val_loader, num_epochs, device)
    test_acc = test_model(model, test_loader, device)
    print(f'Accuracy with {dropout_method}: {test_acc}')
    
    # Plotting
    acc = history['acc']
    val_acc = history['val_acc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label=f'Accuracy with {dropout_method}')
    plt.plot(epochs, val_acc, label=f'Valdiation acc with {dropout_method}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([50,100])
    plt.legend(loc='lower right')
    plt.title("Dropout Classification Accuracy Comparison")
    plt.show()

if __name__ == '__main__':
    main()

