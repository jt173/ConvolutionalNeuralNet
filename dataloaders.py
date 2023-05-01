import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils import data
from util import has_mps, imshow


# --- Imagenet --- 


def imagenet_transforms(is_train=False):
    # Mean and std of train dataset
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    transforms_list = []
    # Use data agug only for train data
    if is_train:
        transforms_list.extend([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip()
        ])
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #if is_train:
    #    transforms_list.extend([
    #        transforms.RandomErasing(0.25)
    #    ])
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
        self.load()

    def transforms(self):
        # Data transformations
        train_transform = imagenet_transforms(is_train=True)
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
    def __init__(self):
        super(ImageNetConfig, self).__init__()
        self.seed = 1
        self.batch_size_mps = 128
        self.batch_size_cpu = 128
        self.num_workers = 4

        self.train_data_path = 'tiny-imagenet-200/new_train'
        self.test_data_path = 'tiny-imagenet-200/new_test'




# --- CIFAR-10 ---
    
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


# --- CIFAR-100 ---
def load_cifar100(data_dir, batch_size, random_seed=32, val_size=0.1, shuffle=True, test=False):
    all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5701, 0.4867, 0.4408],
                                                              std=[0.2675, 0.2565, 0.2761])])
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
def load_mnist(data_dir, batch_size, random_seed=32, val_size=0.1, shuffle=True, test=False):
    all_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081))])
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



def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image
        
        h, w = image.shape[:2]
        
        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset
        
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout