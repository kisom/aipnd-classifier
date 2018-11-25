#!/usr/bin/env python3

import json
import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, models, transforms

import network as net
import sys

data_dir = "flowers"
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"

batch_size = 32

# Data preprocessing.
#
# The testing and validation sets use a standard crop→tensor flow, while
# the training sets use augmented transforms that add additional randomness
# into the training data. This is currently the same set recommended earlier
# in the nanodegree.
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

dataset_training = datasets.ImageFolder(train_dir, transforms.Compose(train_transforms))
dataset_validation = datasets.ImageFolder(
    valid_dir, transforms.Compose(data_transforms)
)
dataset_testing = datasets.ImageFolder(test_dir, transforms.Compose(data_transforms))

dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size)
dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size)
dataloader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size)

with open("cat_to_name.json", "r") as f:
    cat_to_name = json.load(f)

device = 'cpu'
if torch.cuda.is_available():
    print('GPU online.')
    device = 'cuda'

model = None

if len(sys.argv) > 1:
    model = net.model_checkpoint_load(sys.argv[1])
else:
    hyper_params = net.default_hyper_parameters(len(cat_to_name))
    model = net.load_model(hyper_params)
    net.do_deep_learning(model, dataloader_training, dataloader_validation, 50)

if net.check_accuracy(model, dataloader_testing, device, 'testing') > 0.7:
    net.model_checkpoint_write(model, 'checkpoint.dat')
