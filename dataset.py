"""
dataset.py defines a container for a training dataset.
"""
import os
import torch
from torchvision import datasets, transforms


class Dataset:
    """
    Dataset encapsulations training, validation, and testing datasets
    from a single top-level directory.
    """

    def __init__(self, data_dir, batchsize):
        test_transforms = [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        train_transforms = [
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        validate_transforms = [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.batchsize = batchsize
        self.datadir = data_dir

        train_dir = os.path.join(data_dir, "train")
        valid_dir = os.path.join(data_dir, "valid")
        test_dir = os.path.join(data_dir, "test")
        dataset_training = datasets.ImageFolder(
            train_dir, transforms.Compose(train_transforms)
        )
        dataset_validation = datasets.ImageFolder(
            valid_dir, transforms.Compose(validate_transforms)
        )
        dataset_testing = datasets.ImageFolder(
            test_dir, transforms.Compose(test_transforms)
        )

        self.class_to_idx = dataset_training.class_to_idx
        self.training = torch.utils.data.DataLoader(
            dataset_training, batchsize * 2, shuffle=True
        )
        self.validation = torch.utils.data.DataLoader(
            dataset_validation, batchsize, shuffle=True
        )
        self.testing = torch.utils.data.DataLoader(
            dataset_testing, batchsize, shuffle=True
        )

    def __repr__(self):
        return "dataset(data_dir={}, batchsize={})".format(self.datadir, self.batchsize)

    def training_set(self):
        "Returns the training dataset."
        return self.training

    def validation_set(self):
        "Returns the validation dataset."
        return self.validation

    def testing_set(self):
        "Returns the testing dataset."
        return self.testing
