"""
model.py contains code for building models from torchvision models, and defines
a class that acts as a container.
"""

from collections import OrderedDict
import pickle

from PIL import Image
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

DEFAULT_HYPER_PARAMETERS = {
    "architecture": "vgg16_bn",
    "criterion": "NLLLoss",  # Note that the classifier uses LogSoftmax for its final layer.
    "dropout": 0.5,
    "layers": [4096, 4096],
    "learning_rate": 0.001,
    "optimizer": "Adam",
}


class Model:
    """A Model is a container for a neural network."""

    def __init__(self, hp, class_to_idx):
        """
        Initialise a new model. The hyperparameters should be
        a dictionary with the following keys:
            "architecture": a string containing the name of a torchvision model
            "criterion": a string containing the criterion function name
            "dropout": a float containing the dropout rate, expressed between 0 and 1
            "layers": a list of integers, containing the sizes of the hidden layers
            "learning_rate": a float containing the learning rate for the optimizer
            "optimizer": a string containing the optimizer function

        Note that the final hidden layer will have a LogSoftmax put on it, which may
        affect the choice of criterion.
        """

        self.hyper_params = hp
        self.class_to_idx = class_to_idx
        self.idx_to_class = {}
        for key, value in self.class_to_idx.items():
            self.idx_to_class[value] = key

        params = self._setup_network(
            hp["architecture"], hp["layers"], len(class_to_idx), hp["dropout"]
        )
        self.criterion = getattr(nn, hp["criterion"])()
        self.optimizer = getattr(optim, hp["optimizer"])(params, lr=hp["learning_rate"])

    def _freeze_model(self):
        for param in self.network.parameters():
            if param.requires_grad:
                param.requires_grad = False

    def _setup_network(self, arch, layers, noutputs, dropout):
        """
        Given an architecture, layer description, and dropout rate, set up a
        pretrained model and return the parameters to pass to the optimiser.
        """
        if arch == "vgg":
            arch = "vgg16_bn"
        elif arch == "densenet":
            arch = "densenet121"
        elif arch == "resnet":
            arch = "resnet152"

        self.network = getattr(models, arch)(pretrained=True)
        params = None  # optimizer params
        if arch.startswith("vgg"):
            self._freeze_model()
            self.network.classifier = _build_classifier(
                layers, self.network.classifier[0].in_features, noutputs, dropout
            )
            params = self.network.classifier.parameters()
        elif arch.startswith("densenet"):
            self._freeze_model()
            self.network.classifier = _build_classifier(
                layers, self.network.classifier.in_features, noutputs, dropout
            )
            params = self.network.classifier.parameters()
        elif arch.startswith("inception"):
            self._freeze_model()
            self.network.fc = _build_classifier(
                layers, self.network.fc.in_features, noutputs, dropout
            )
            params = self.network.fc.parameters()
        elif arch == "alexnet":
            self._freeze_model()
            self.network.classifier = _build_classifier(
                layers, self.network.classifier[1].in_features, noutputs, dropout
            )
            params = self.network.classifier.parameters()
        elif arch.startswith("resnet"):
            self._freeze_model()
            self.network.fc = _build_classifier(
                layers, self.network.fc.in_features, noutputs, dropout
            )
            params = self.network.fc.parameters()
        else:
            raise ValueError("Unsupported architecture " + arch)

        return params

    def checkpoint(self, optimizer=False):
        """
        checkpoint generates a checkpoint for a model; this checkpoint can be
        used to later restore the model. If optimizer is True, the state of the
        optimizer is stored as well.
        """
        checkpoint = {
            "state": self.network.to("cpu").state_dict(),
            "hp": self.hyper_params,
            "class_to_idx": self.class_to_idx,
        }

        if optimizer:
            checkpoint["optim"] = self.optimizer.state_dict()

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

        class_to_idx = state_data["class_to_idx"]
        model = Model(state_data["hp"], class_to_idx)
        model.network.load_state_dict(state_data["state"])

        if "optim" in state_data:
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

    def to(self, device):
        """
        move the model to the selected device.
        """
        self.network.to(device)

    def gpu(self):
        """
        convenience function to move the model to the GPU.
        """
        return self.to("cuda")

    def cpu(self):
        """
        convenience function to move the model to the CPU.
        """

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

    def recognize(self, image, topk=5, labels=None):
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

        predictions = OrderedDict()
        for i in range(len(outputs)):
            klass = self.idx_to_class[indices[i].item()]
            if labels:
                predictions[labels[klass]] = outputs[i].item() * 100
            else:
                predictions[klass] = outputs[i].item() * 100
        return predictions


def _build_classifier(layers, ninputs, nfeatures, dropout):
    classifier = []

    layer0 = [nn.Linear(ninputs, layers[0]), nn.ReLU(), nn.Dropout(p=dropout)]
    classifier.extend(layer0)

    for i in range(1, len(layers)):
        layer = [nn.Linear(layers[i - 1], layers[i], bias=True), nn.ReLU()]
        classifier.extend(layer)

    classifier.extend(
        [nn.Linear(layers[-1], nfeatures, bias=True), nn.LogSoftmax(dim=1)]
    )
    return nn.Sequential(*classifier)


def load(path):
    """
    Given a path to a checkpoint, load will attempt to restore a model.
    """
    return Model.load(path)


def recognize(model, image, topk=5, labels=None):
    """
    recognize is a shortcut for model.recognize(image, topk, labels).
    """
    return model.recognize(image, topk, labels)


def recognize_path(model, image_path, topk=5, labels=None):
    """
    recognize_path loads the image stored in image_path and runs the model's recognize
    method on it.
    """

    image = Image.open(image_path)
    return model.recognize(image, topk, labels)
