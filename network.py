#!/usr/bin/env python3

import datetime
import json
import pickle
import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, models, transforms


def default_hyper_parameters(n_cats):
    return {
        "architecture": models.vgg11_bn,
        "epochs": 6,
        "criterion": nn.CrossEntropyLoss,
        "optimizer": optim.Adam,
        "learning_rate": 0.01,
        "classifier": [
            nn.Linear(25088, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, n_cats),
            # nn.LogSoftmax(dim=1),
        ],
    }


def load_model(hyper_parameters):
    model = hyper_parameters["architecture"](pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(*hyper_parameters["classifier"])
    model.criterion  = hyper_parameters["criterion"]()
    model.optimizer  = hyper_parameters["optimizer"](model.classifier.parameters(), lr=hyper_parameters['learning_rate'])
    model.hyper_parameters = hyper_parameters
    return model


def do_deep_learning(model, trainloader, validateloader, print_every):
    print("starting deep learning cycle.")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Only train the classifier parameters, feature parameters are frozen.
    epochs = model.hyper_parameters["epochs"]

    print_every = print_every
    steps = 0

    training_started = datetime.datetime.now()
 
    model.to(device) # I consider the time it takes to move a model to the GPU
                     # as part of the time to train.

    for e in range(epochs):
        print("starting epoch:", e)
        running_loss = 0
        epoch_started = datetime.datetime.now()
        tdelta = datetime.datetime.now()
        for (inputs, labels) in iter(trainloader):
            model.train()
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            model.optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print(
                    "{} {:6}|".format(datetime.datetime.now() - tdelta, steps),
                    "epoch: {}/{}... ".format(e + 1, epochs),
                    "loss: {:.4f}".format(running_loss / print_every),
                )

                tdelta = datetime.datetime.now()
                running_loss = 0

        check_accuracy(model, validateloader, device, 'validation')
        print("epoch completed in: {}".format(datetime.datetime.now() - epoch_started))
        print("-" * 72)
    print("training completed in {}".format(datetime.datetime.now() - training_started))

def check_accuracy(model, dataloader, device, datalabel):
    model.eval()
    max_fn = torch.max
    # if torch.cuda.is_available():
    #     max_fn = torch.cuda.max
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = max_fn(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "{} accuracy over {} test images: {:0.4}% ({}/{})".format(
            datalabel, total, (100 * correct / total), correct, total,
        )
    )

    return correct / total


def model_checkpoint_write(model, path):
    with open(path, 'wb') as checkpoint:
        checkpoint.write(pickle.dumps(model))


def model_checkpoint_load(path):
    with open(path, 'rb') as checkpoint:
        return pickle.loads(checkpoint.read())