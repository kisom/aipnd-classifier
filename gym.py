"""
gym.py defines a training arena, called a gym, for training models.
"""
import datetime
import logging
import numpy as np
import torch


class Gym:
    """
    A Gym represents a training and evaluation arena for a model. It
    is essential a container for a model and dataset.
    """

    def __init__(self, model, dataset, print_every=None):
        """
        A gym is initialised with a model and dataset. If print_every isn't set,
        it's set to the dataset's batchsize.
        """

        if not print_every:
            print_every = dataset.batchsize

        if torch.cuda.is_available():
            logging.info("GPU available")
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = model
        self.dataset = dataset
        self.print_every = print_every

    def train(self):
        """
        train runs the model through backprop for a number of epochs. The number
        of epochs is controlled with the model's hyperparameters. After each epoch,
        a validation run is done.
        """

        logging.info("starting deep learning via {}".format(self.device))

        # Only train the classifier parameters, feature parameters are frozen.
        epochs = self.model.hyper_params["epochs"]
        print_every = self.print_every
        steps = 0
        training_started = datetime.datetime.now()

        self.model.network.to(self.device)
        last_accuracy = 0

        for epoch in range(epochs):
            self.model.train()

            logging.info("starting epoch: {}".format(epoch))
            running_loss = 0
            epoch_started = datetime.datetime.now()
            tdelta = datetime.datetime.now()

            for (inputs, labels) in iter(self.dataset.training):
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                _, loss = self.model.backprop(inputs, labels)
                running_loss += loss

                if steps % print_every == 0:
                    logging.info(
                        "{} {:6}|".format(datetime.datetime.now() - tdelta, steps),
                        "epoch: {}/{}... ".format(epoch + 1, epochs),
                        "loss: {:.4f}".format(running_loss / print_every),
                    )

                    tdelta = datetime.datetime.now()
                    running_loss = 0

            accuracy = self._check_accuracy(self.dataset.validation, "validation")
            if accuracy < last_accuracy:
                logging.warning("accuracy has decreased")
            elif np.isclose(accuracy, last_accuracy, rtol=0.001):
                logging.warning("WARNING: accuracy has not increased")
            last_accuracy = accuracy
            logging.info(
                "epoch completed in: {}".format(datetime.datetime.now() - epoch_started)
            )
            print("-" * 72)

        logging.info(
            "training completed in {}".format(
                datetime.datetime.now() - training_started
            )
        )

    def _check_accuracy(self, dataset, datalabel):
        """
        _check_accuracy puts the network in evaluation mode and checks
        its performance on a given dataset. The datalabel is useful
        for identifying whether this is an evaluation or validation check.
        """

        logging.info("running {} evaluation".format(datalabel))
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataset:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.network.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(
            "{} accuracy over {} test images: {:0.4}% ({}/{})".format(
                datalabel, total, (100 * correct / total), correct, total
            )
        )

        return correct / total

    def evaluate(self):
        """
        evaluate checks the accuracy of the model over the evaluation dataset.
        """
        return self._check_accuracy(self.dataset.testing, "testing")
