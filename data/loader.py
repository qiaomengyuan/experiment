# =============================================================================
# data/loader.py   # CIFAR-10数据加载
# =============================================================================
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(config):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=False, download=False, transform=transform_test
    )

    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, trainset
