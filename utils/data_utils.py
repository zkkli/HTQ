import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def getTrainData(dataset='imagenet',
                 path='/dataset/imagenet/',
                 batch_size=32,
                 for_inception=False):
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        traindir = path + 'train'
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=12, pin_memory=True)
        return train_loader

    elif dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10('_dataset', True, train_transform, download=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=12, pin_memory=True)
        return train_loader


def getTestData(dataset='imagenet',
                path='/dataset/imagenet/',
                batch_size=128,
                for_inception=False):
    if dataset == 'imagenet':
        input_size = 299 if for_inception else 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        test_dataset = datasets.ImageFolder(
            path + 'val',
            transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=32)
        return test_loader

    elif dataset == 'cifar10':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        test_dataset = datasets.CIFAR10('_dataset', False, test_transform)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=32)
        return test_loader
