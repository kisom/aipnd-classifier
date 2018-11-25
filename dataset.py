import os
import torch
from torchvision import datasets, transforms

class Dataset():
    def __init__(self, data_dir, batchsize):
        data_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        train_transforms = [
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]

        train_dir = os.path.join(data_dir, "train")
        valid_dir = os.path.join(data_dir, "valid")
        test_dir = os.path.join(data_dir, "test")
        dataset_training = datasets.ImageFolder(train_dir, transforms.Compose(train_transforms))
        dataset_validation = datasets.ImageFolder(
            valid_dir, transforms.Compose(data_transforms)
        )
        dataset_testing = datasets.ImageFolder(test_dir, transforms.Compose(data_transforms))

        self.training = torch.utils.data.DataLoader(dataset_training, batchsize*2)
        self.validation = torch.utils.data.DataLoader(dataset_validation, batchsize)
        self.testing = torch.utils.data.DataLoader(dataset_testing, batchsize)
