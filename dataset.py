import torch
from torchvision import datasets, transforms

class Dataset():
    def __init__(self, data+dir, batchsize):
        data_transforms = [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]
        train_transforms = [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(100),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]

        train_dir = data_dir + "/train"
        valid_dir = data_dir + "/valid"
        test_dir = data_dir + "/test"
        dataset_training = datasets.ImageFolder(train_dir, transforms.Compose(train_transforms))
        dataset_validation = datasets.ImageFolder(
            valid_dir, transforms.Compose(data_transforms)
        )
        dataset_testing = datasets.ImageFolder(test_dir, transforms.Compose(data_transforms))

        self.training = torch.utils.data.DataLoader(dataset_training, batch_size)
        self.validation = torch.utils.data.DataLoader(dataset_validation, batch_size)
        self.testing = torch.utils.data.DataLoader(dataset_testing, batch_size)
