import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils import data
from util import has_mps, imshow


# --- Imagenet --- 


def imagenet_transforms(cutout=False, is_train=False):
    # Mean and std of train dataset
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    transforms_list = []
    # Use data agug only for train data
    if is_train:
        if cutout:
            transforms_list.extend([
                Cutout(0.5, 8)
            ])
        transforms_list.extend([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip()
        ])
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transforms.Compose(transforms_list)


class ImagenetData(object):
    classes = ['%s' % i for i in range(200)]

    def __init__(self, args):
        super(ImagenetData, self).__init__()
        self.batch_size_mps = args.batch_size_mps
        self.batch_size_cpu = args.batch_size_cpu
        self.num_workers = args.num_workers
        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.cutout = args.cutout
        self.load()

    def transforms(self):
        # Data transformations
        train_transform = imagenet_transforms(self.cutout, is_train=True)
        test_transform = imagenet_transforms(is_train=False)
        return train_transform, test_transform
    
    def dataset(self):
        # Get data transforms
        train_transform, test_transform = self.transforms()

        # Dataset and train/test split
        train_set = torchvision.datasets.ImageFolder(root=self.train_data_path,
                                                     transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root=self.test_data_path,
                                                    transform=test_transform)
        return train_set, test_set
    
    def load(self):
        # Get train and test data
        train_set, test_set = self.dataset()

        # Dataloader arguments & test/train dataloaders
        dataloader_args = dict(
            shuffle=True,
            batch_size = self.batch_size_cpu
        )
        if has_mps():
            dataloader_args.update(
                batch_size = self.batch_size_mps,
                num_workers = self.num_workers,
                pin_memory = True
            )
        self.train_loader = data.DataLoader(train_set, **dataloader_args)
        self.test_loader = data.DataLoader(test_set, **dataloader_args)
    
    def show_samples(self):
        # Get some random training images
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        index = []
        num_img = min(len(self.classes), 10)
        for i in range(num_img):
            for j in range(len(labels)):
                if labels[j] == i:
                    index.append(j)
                    break
        if len(index) < num_img:
            for j in range(len(labels)):
                if len(index) == num_img:
                    break
                if j not in index:
                    index.append(j)
        imshow(torchvision.utils.make_grid(images[index], nrow=num_img, scale_each=True), "Sample train data")


class ImageNetConfig(object):
    def __init__(self, cutout=False):
        super(ImageNetConfig, self).__init__()
        self.seed = 1
        self.batch_size_mps = 64
        self.batch_size_cpu = 64
        self.num_workers = 4
        self.cutout = cutout

        self.train_data_path = 'tiny-imagenet-200/new_train'
        self.test_data_path = 'tiny-imagenet-200/new_test'




# --- CIFAR-10 ---
    
# Returns data loaders for training, validation, and test data
def load_cifar10(data_dir, batch_size, random_seed=32, val_size=0.1, shuffle=True, test=False, cutout=False):
    transforms_list = []
    if test:
        transforms_list.extend([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                    std=[0.2023, 0.1994, 0.2010])])
    else:
        if cutout:
            transforms_list.extend([Cutout(0.5, 8)])
        transforms_list.extend([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                     std=[0.2023, 0.1994, 0.2010])])
    all_transforms = transforms.Compose(transforms_list)
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


# --- CIFAR-100 ---
def load_cifar100(data_dir, batch_size, random_seed=32, val_size=0.1, shuffle=True, test=False, cutout=False):
    transforms_list = []
    if test:
        transforms_list.extend([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5701, 0.4867, 0.4408],
                                                    std=[0.2675, 0.2565, 0.2761])])
    else:
        if cutout:
            transforms_list.extend([Cutout(0.5, 8)])
        transforms_list.extend([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5701, 0.4867, 0.4408],
                                                    std=[0.2675, 0.2565, 0.2761])])
    all_transforms = transforms.Compose(transforms_list)
    # Return test data
    if test:
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=all_transforms, download=True)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader
    # Return train and val loaders
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, transform=all_transforms, download=True)
    val_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, transform=all_transforms, download=True)

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


# --- MNIST ---

# Returns data loaders for training, validation, and test data
def load_mnist(data_dir, batch_size, random_seed=32, val_size=0.1, shuffle=True, test=False, cutout=False):
    transforms_list = []
    if test:
        transforms_list.extend([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081))])
    else:
        if cutout:
            transforms_list.extend([Cutout(0.5, 8)])
        transforms_list.extend([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081))])
    all_transforms = transforms.Compose(transforms_list)
    
    # Return the test loader
    if test:
        test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, transform=all_transforms, download=True)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader
    # Return training and val loaders
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=all_transforms, download=True)
    val_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, transform=all_transforms, download=True)

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


# Cutout Augmentation

def cutout(img, pad_size, replace):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    center_h, center_w = torch.randint(high=h, size=(1,)), torch.randint(high=w, size=(1,))
    low_h, high_h = torch.clamp(center_h-pad_size, 0, h).item(), torch.clamp(center_h+pad_size, 0, h).item()
    low_w, high_w = torch.clamp(center_w-pad_size, 0, w).item(), torch.clamp(center_w+pad_size, 0, w).item()
    cutout_img = img.clone()
    cutout_img[:, low_h:high_h, low_w:high_w] = replace
    return F.to_pil_image(cutout_img)

class Cutout(torch.nn.Module):
    def __init__(self, p, pad_size, replace=128):
        super().__init__()
        self.p = p
        self.pad_size = int(pad_size)
        self.replace = replace

    def forward(self, image):
        if torch.rand(1) < self.p:
            cutout_image = cutout(image, self.pad_size, self.replace)
            return cutout_image
        else:
            return image
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, pad_size={1})".format(self.p, self.pad_size)