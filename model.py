"""
model.py contains code for building models from torchvision models, and defines
a class that acts as a container.
"""

import logging
import pickle

from torch import nn
from torch import optim
from torchvision import models

# Resources:
#   + https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch


def _build_classifier(hp):
    classifier = []
    layers = hp["layers"]

    layer0 = [
        nn.Linear(hp["ninputs"], layers[0]),
        nn.ReLU(),
        nn.Dropout(p=hp["dropout"]),
    ]
    classifier.extend(layer0)

    for i in range(1, len(layers)):
        layer = [nn.Linear(layers[i - 1], layers[i], bias=True), nn.ReLU()]
        classifier.extend(layer)

    classifier.extend(
        [nn.Linear(layers[-1], hp["nfeatures"], bias=True), nn.LogSoftmax(dim=1)]
    )
    return nn.Sequential(*classifier)


DEFAULT_HYPER_PARAMETERS = {
    "architecture": "vgg16_bn",
    "criterion": "NLLLoss",  # Note that the classifier uses LogSoftmax for its final layer.
    "dropout": 0.5,
    "epochs": 6,
    "layers": [4096, 4096],
    "learning_rate": 0.001,
    "nfeatures": 102,
    "ninputs": 32,
    "optimizer": "Adam",
}

MODEL_NINPUTS = {
    "alexnet": 9216,
    "densenet121": 1024,
    "squeezenet": 106496,
    "vgg11_bn": 25088,
    "vgg13_bn": 25088,
    "vgg16_bn": 25088,
    "vgg19_bn": 25088,
    "vgg11": 25088,
    "vgg13": 25088,
    "vgg16": 25088,
    "vgg19": 25088,
}

CLASSIFIER_MODELS = [
    "densenet121",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
]


class Model:
    """A Model is a container for a neural network."""

    def __init__(self, hp):
        """
        Initialise a new model. The hyperparameters should be
        a dictionary with the following keys:
            "architecture": a string containing the name of a torchvision model
            "criterion": a string containing the criterion function name
            "dropout": a float containing the dropout rate, expressed between 0 and 1
            "epochs": an integer containing the number of training epochs
            "layers": a list of integers, containing the sizes of the hidden layers
            "learning_rate": a float containing the learning rate for the optimizer
            "nfeatures": an integer that is the number of features
            "ninputs": an integer that is the number of inputs to the hidden layers
            "optimizer": a string containing the optimizer function

        Note that the final hidden layer will have a LogSoftmax put on it, which may
        affect the choice of criterion.
        """

        self.hyper_params = hp

        self.network = getattr(models, hp["architecture"])(pretrained=True)
        for param in self.network.parameters():
            if param.requires_grad:
                param.requires_grad = False

        if hp["architecture"] in MODEL_NINPUTS:
            hp["ninputs"] = MODEL_NINPUTS[hp["architecture"]]

        if hp["architecture"] in CLASSIFIER_MODELS:
            self.network.classifier = _build_classifier(hp)
            self.optimizer = getattr(optim, hp["optimizer"])(
                self.network.classifier.parameters(), lr=hp["learning_rate"]
            )
        else:
            self.network.fc = _build_classifier(hp)
            self.optimizer = getattr(optim, hp["optimizer"])(
                self.network.fc.parameters(), lr=hp["learning_rate"]
            )

        self.criterion = getattr(nn, hp["criterion"])()

    def save(self, path):
        """
        save writes the model to disk such that it can be restored later. This
        is useful for checkpointing after training.
        """

        with open(path, "wb") as out:
            out.write(pickle.dumps(self))

    @classmethod
    def load(cls, path):
        """
        load restores a checkpointed Model from disk.
        """

        with open(path, "rb") as input_file:
            model = pickle.loads(input_file.read())
        return model

    def train(self):
        """"
        train places the model in training mode.
        """
        self.network.train()
        logging.info("network is in training mode")

    def eval(self):
        """
        eval places the model in evaluation mode.
        """
        self.network.eval()
        logging.info("network is in evaluation mode")

    def forward(self, inputs):
        """
        forward runs the inputs through the network.
        """
        return self.network.forward(inputs)

    def backprop(self, inputs, labels):
        """
        backprop runs a forward pass through the network and backpropagates
        updates.
        """
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return (outputs, loss.item())
