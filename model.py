import pickle
import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, models, transforms

# Resources:
#   + https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch

def _build_classifier(hp):
    classifier = []
    layers = hp['layers']


    layer0 = [nn.Linear(hp['ninputs'], layers[0]),
              nn.ReLU(),
              nn.Dropout(p=hp['dropout'])]
    classifier.extend(layer0)
    
    for i in range(1, len(layers)):
        layer = [nn.Linear(layers[i-1], layers[i], bias=True),
                 nn.ReLU()]
        classifier.extend(layer)

    classifier.extend([nn.Linear(layers[-1], hp['nfeatures'], bias=True), 
                       nn.LogSoftmax(dim=1)])
    return nn.Sequential(*classifier)

DEFAULT_HYPER_PARAMETERS = {
    'architecture': 'vgg16_bn',
    'criterion': 'NLLLoss', # Note that the classifier uses LogSoftmax for its final layer.
    'dropout': 0.5,
    'epochs': 6,
    'layers': [4096, 4096],
    'learning_rate': 0.001,
    'nfeatures': 102,
    'ninputs': 32,
    'optimizer': 'Adam',
}

MODEL_NINPUTS = {
    'alexnet': 9216,
    'densenet121': 1024,
    'squeezenet': 106496,
    'vgg11_bn': 25088,
    'vgg13_bn': 25088,
    'vgg16_bn': 25088,
    'vgg19_bn': 25088,
    'vgg11': 25088,
    'vgg13': 25088,
    'vgg16': 25088,
    'vgg19': 25088,
}

CLASSIFIER_MODELS = ["densenet121", 'vgg11_bn', 'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
]

class Model():
    """A Model is a container for a neural network."""

    def __init__(self, hp):
        self.hyper_params = hp

        self.network = getattr(models, hp['architecture'])(pretrained=True)
        for param in self.network.parameters():
            if param.requires_grad:
                param.requires_grad = False

        if hp['architecture'] in MODEL_NINPUTS:
            hp['ninputs'] = MODEL_NINPUTS[hp['architecture']]

        if hp['architecture'] in CLASSIFIER_MODELS:
            self.network.classifier = _build_classifier(hp)    
            self.optimizer = getattr(optim, hp['optimizer'])(self.network.classifier.parameters(), lr=hp['learning_rate'])
        else:
            self.network.fc = _build_classifier(hp)
            self.optimizer = getattr(optim, hp['optimizer'])(self.network.fc.parameters(), lr=hp['learning_rate'])
           
        self.criterion = getattr(nn, hp['criterion'])()
        
    
    def save(self, path):
        with open(path, 'wb') as out:
            out.write(pickle.dumps(self))

    @classmethod
    def load(self, path):
        with open(path, 'rb') as input:
            model = pickle.loads(input.read())
        return model