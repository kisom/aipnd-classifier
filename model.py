"""
model.py contains code for building models from torchvision models, and defines
a class that acts as a container.
"""

from collections import OrderedDict
import pickle

import torch
from torch import nn
from torch import optim
from torchvision import models, transforms

import util

# Resources:
#   + https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch

# Possible improvements:
#   + version models (e.g. # of epochs of training)
#   + checkpoint file version

log = util.get_logger()


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

    def __init__(self, hp, labels=None, class_to_idx=None):
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
        if labels:
            self.hyper_params["nfeatures"] = len(labels)
            self.labels = labels
        if class_to_idx:
            self.class_to_idx = class_to_idx
            self.idx_to_class = {}
            for key, value in self.class_to_idx.items():
                self.idx_to_class[value] = key

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
        layer_list = [hp["ninputs"]]
        layer_list.extend(hp["layers"])
        layer_list.append(hp["nfeatures"])
        classifier_layers = "x".join([str(x) for x in layer_list])
        self._meta = {
            "architecture": hp["architecture"],
            "layers": classifier_layers,
            "criterion": hp["criterion"],
            "optimizer": hp["optimizer"],
            "learning_rate": hp["learning_rate"],
        }

    def __repr__(self):
        return str(self._meta)

    def checkpoint(self, optimizer=False):
        """
        checkpoint generates a checkpoint for a model; this checkpoint can be
        used to later restore the model. If optimizer is True, the state of the
        optimizer is stored as well.
        """
        checkpoint = {
            "state": self.network.to("cpu").state_dict(),
            "hp": self.hyper_params,
        }

        if optimizer:
            checkpoint["optim"] = self.optimizer.state_dict()

        if self.labels:
            checkpoint["labels"] = self.labels

        if self.class_to_idx:
            checkpoint["class_to_idx"] = self.class_to_idx

        return checkpoint

    def save(self, path):
        """
        save writes the model to disk such that it can be restored later. This
        is useful for checkpointing after training.
        """

        with open(path, "wb") as out:
            pickle.dump(self.checkpoint(), out)

    @classmethod
    def restore(cls, state_data):
        """
        restore takes a checkpoint and restores it.
        """
        labels = None
        if "labels" in state_data:
            labels = state_data["labels"]

        class_to_idx = None
        if "class_to_idx" in state_data:
            class_to_idx = state_data["class_to_idx"]

        model = Model(state_data["hp"], labels, class_to_idx)
        model.network.load_state_dict(state_data["state"])

        if optim in state_data:
            model.optimizer.load_state_dict(state_data["optim"])
        return model

    @classmethod
    def load(cls, path):
        """
        load restores a checkpointed Model from disk.
        """

        with open(path, "rb") as input_file:
            state_data = pickle.load(input_file)
        return Model.restore(state_data)

    def train(self):
        """"
        train places the model in training mode.
        """
        self.network.train()
        log.info("network is in training mode")

    def eval(self):
        """
        eval places the model in evaluation mode.
        """
        self.network.eval()
        log.info("network is in evaluation mode")

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

    def recognize(self, image, topk=5):
        """
        recognise takes a PIL image of a flower and returns the topk predictions
        for what the network thinks that flower is.
        """

        self.eval()

        # Reuse the same image processing pipeline we used in the datasets.
        input_image = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )(image)
        # The unsqueeze makes this a 4D tensor, which acts as a list of of 3D tensors.
        input_image.unsqueeze_(0)

        # The last layer is a log softmax, so we'll want to apply an exponential
        # to undo it and get the actual probability. After forwarding it, we get
        # the topk probabilities.
        # pylint: disable=E1101
        outputs, indices = torch.exp(self.forward(input_image)).topk(topk)
        # pylint: enable=E1101

        # The outputs and indicies are in a 2D tensor, but the first dimension
        # is of length 1 and isn't necessary. E.g. the answer we get back is
        # [[p0, p1, p2, ..., pk]] and squeeze removes that outer set of braces.
        outputs.squeeze_(0)
        indices.squeeze_(0)

        # If we have labels and a mapping from class to index, we can do cool
        # things.
        if self.labels and self.class_to_idx:
            predictions = OrderedDict()

            for i in enumerate(outputs):
                klass = self.idx_to_class[indices[i].item()]
                name = self.labels[klass]
                predictions[name] = outputs[i].item() * 100
            return predictions
        return outputs, indices


def load(path):
    """
    Given a path to a checkpoint, load will attempt to restore a model.
    """
    return Model.load(path)
