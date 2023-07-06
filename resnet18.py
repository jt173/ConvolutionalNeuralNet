import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataloaders import load_cifar10, load_cifar100, ImageNetConfig, ImagenetData
from util import train_model, test_model 

# Device config
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, spatial=False):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.spatial = spatial

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.spatial:
            out = F.dropout2d(out)
    
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.spatial:
            out = F.dropout2d(out)
        
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)

        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes, image_channels=3, dropout=False, spatial=False, mp_drop=False):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dropout
        # Using default p=0.5
        self.dropout = dropout
        self.spatial=spatial
        self.mp_drop = mp_drop

        # Resnet layers
        self.layer1 = self.make_layer(64, 64, self.spatial, stride=1)
        self.layer2 = self.make_layer(64, 128, self.spatial, stride=2)
        self.layer3 = self.make_layer(128, 256, self.spatial, stride=2)
        self.layer4 = self.make_layer(256, 512, self.spatial, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.dropout:
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes))
        else:
            self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, spatial, stride):
        downsample = None
        if stride != 1:
            downsample = self.downsample(in_channels, out_channels)
        return nn.Sequential(
            Block(in_channels, out_channels, downsample=downsample, stride=stride, spatial=spatial),
            Block(out_channels, out_channels, spatial=spatial)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        if self.spatial:
            out = F.dropout2d(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        if self.mp_drop:
            out = F.dropout(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
    
    def downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
def main():
    # Variables
    batch_size = 64
    num_classes = 200
    learning_rate = 0.001
    num_epochs = 15

    # Load data
    args = ImageNetConfig()
    data = ImagenetData(args)

    # No Dropout
    model = ResNet18(num_classes).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = train_model(model, optimizer, loss_function, data.train_loader, data.test_loader, num_epochs, device)
    test_acc = test_model(model, data.test_loader, device)
    print("Accuracy without dropout {:.4f}".format(test_acc))

    # Fully connected dropout
    dropout = True
    model = ResNet18(num_classes, dropout=dropout).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

    fc_history = train_model(model, optimizer, loss_function, data.train_loader, data.test_loader, num_epochs, device)
    test_acc = test_model(model, data.test_loader, device)
    print("Accuracy with fully-connected dropout {:.4f}".format(test_acc))

    # Max-pooling dropout
    mp_dropout = True
    model = ResNet18(num_classes, mp_drop=mp_dropout).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

    mp_history = train_model(model, optimizer, loss_function, data.train_loader, data.test_loader, num_epochs, device)
    test_acc = test_model(model, data.test_loader, device)
    print("Accuracy with max-pooling dropout {:.4f}".format(test_acc))

    # Spatial dropout
    spatial = True
    model = ResNet18(num_classes, spatial=spatial).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

    sp_history = train_model(model, optimizer, loss_function, data.train_loader, data.test_loader, num_epochs, device)
    test_acc = test_model(model, data.test_loader, device)
    print("Accuracy with spatial dropout {:.4f}".format(test_acc))

    # Cutout
    cutout = True
    args = ImageNetConfig(cutout)
    data = ImagenetData(args)

    model = ResNet18(num_classes).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

    c_history = train_model(model, optimizer, loss_function, data.train_loader, data.test_loader, num_epochs, device)
    test_acc = test_model(model, data.test_loader, device)
    print("Accuracy with cutout {:.4f}".format(test_acc))

    # Plotting
    acc = history['acc']
    val_acc = history['val_acc']
    fc_acc = fc_history['acc']
    fc_val_acc = fc_history['val_acc']
    mp_acc = mp_history['acc']
    mp_val_acc = mp_history['val_acc']
    sp_acc = sp_history['acc']
    sp_val_acc = sp_history['val_acc']
    c_acc = c_history['acc']
    c_val_acc = c_history['val_acc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label='Accuracy without dropout')
    plt.plot(epochs, val_acc, label='Valdiation acc without dropout')
    plt.plot(epochs, fc_acc, label='Accuracy with FC')
    plt.plot(epochs, fc_val_acc, label='Validation acc with FC')
    plt.plot(epochs, mp_acc, label='Accuracy with MP')
    plt.plot(epochs, mp_val_acc, label='Validation acc with MP')
    plt.plot(epochs, sp_acc, label="Accuracy with SP")
    plt.plot(epochs, sp_val_acc, label='Validation acc with SP')
    plt.plot(epochs, c_acc, label='Accuacy with cutout')
    plt.plot(epochs, c_val_acc, label='Validation acc with cutout')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([50,100])
    plt.legend(loc='lower right')
    plt.title("Dropout Classification Accuracy Comparison")
    plt.show()

if __name__ == '__main__':
    main()