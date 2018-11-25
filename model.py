import pickle
import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, models, transforms

def _build_classifier(hp):
    classifier = []
    layers = hp['layers']


    layer0 = [nn.Linear(hp['ninputs'], layers[0]),
              nn.ReLU(),
              nn.Dropout(p=hp['dropout'])]
    classifier.extend(layer0)
    
    for i in range(1, len(layers)):
        layer = [nn.Linear(layers[i-1], layers[i]),
                 nn.ReLU(),
                 nn.Dropout(hp['dropout'])]
        classifier.extend(layer)

    classifier.append(nn.Linear(layers[-1], hp['nfeatures']))
    return nn.Sequential(*classifier)

DEFAULT_HYPER_PARAMETERS = {
    'architecture': 'vgg11_bn',
    'criterion': 'CrossEntropyLoss',
    'dropout': 0.5,
    'epochs': 3,
    'layers': [4096, 2048],
    'learning_rate': 0.01,
    'nfeatures': 102,
    'ninputs': 25088,
    'optimizer': 'Adam',
}

class Model():
    """A Model is a container for a neural network."""

    def __init__(self, hp):
        self.hyper_params = hp

        self.network = getattr(models, hp['architecture'])(pretrained=True)
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.classifier = _build_classifier(hp)
        self.criterion = getattr(nn, hp['criterion'])()
        self.optimizer = getattr(optim, hp['optimizer'])(self.network.classifier.parameters(), lr=hp['learning_rate'])
    
    def save(self, path):
        with open(path, 'wb') as out:
            out.write(pickle.dumps(self))

    @classmethod
    def load(self, path):
        with open(path, 'rb') as input:
            model = pickle.loads(input.read())
        return model