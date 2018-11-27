"""
gym.py defines a training arena, called a gym, for training models.
"""
import datetime

import numpy as np
import torch

import util

log = util.get_logger()


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
            log.info("GPU available")
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = model
        self.dataset = dataset
        self.print_every = print_every

    def train(self, max_stalls=3):
        """
        train runs the model through backprop for a number of epochs. The number
        of epochs is controlled with the model's hyperparameters. After each epoch,
        a validation run is done. max_stalls controls how many epochs training can
        be stalled (no improvement in the accuracy) before the training is cutoff.
        """

        log.info("starting deep learning via {}".format(self.device))

        # Only train the classifier parameters, feature parameters are frozen.
        epochs = self.model.hyper_params["epochs"]
        print_every = self.print_every
        steps = 0
        training_started = datetime.datetime.now()

        self.model.network.to(self.device)
        last_accuracy = 0
        best_accuracy = 0
        stalls = 0

        for epoch in range(epochs):
            self.model.train()

            log.info("starting epoch: {}".format(epoch))
            running_loss = 0
            epoch_started = datetime.datetime.now()
            tdelta = datetime.datetime.now()

            for (inputs, labels) in iter(self.dataset.training):
                steps += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                _, loss = self.model.backprop(inputs, labels)
                running_loss += loss

                if steps % print_every == 0:
                    log.info(
                        "{} {:6}|epoch: {}/{}; loss: {:.4f}".format(
                            datetime.datetime.now() - tdelta,
                            steps,
                            epoch + 1,
                            epochs,
                            running_loss / print_every,
                        )
                    )

                    tdelta = datetime.datetime.now()
                    running_loss = 0

            accuracy = self._check_accuracy(self.dataset.validation, "validation")
            if accuracy < last_accuracy:
                log.warn("accuracy has decreased")
                stalls += 1
            elif np.isclose(accuracy, last_accuracy, rtol=0.001):
                log.warn("accuracy has not increased")
                stalls += 1
            elif best_accuracy < accuracy:
                stalls = 0
                best_accuracy = accuracy
            last_accuracy = accuracy
            log.info(
                "epoch completed in: {}".format(datetime.datetime.now() - epoch_started)
            )
            print("-" * 72)

            if stalls >= max_stalls >= 0:
                log.error("training has stalled, stopping")
                break
        log.info(
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

        log.info("running {} evaluation".format(datalabel))
        self.model.eval()
        self.model.network.to(self.device)
        correct = 0
        total = 0

        started = datetime.datetime.now()
        with torch.no_grad():
            for data in dataset:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.network.forward(inputs)

                # max is dynamically generated, so pylint thinks it's not there.
                # pylint: disable=E1101
                _, predicted = torch.max(outputs.data, 1)
                # pylint: enable=E1101
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        log.info(
            "{}: {} accuracy over {} test images: {:0.4}% ({}/{})".format(
                datetime.datetime.now() - started,
                datalabel,
                total,
                (100 * correct / total),
                correct,
                total,
            )
        )

        return correct / total

    def evaluate(self):
        """
        evaluate checks the accuracy of the model over the evaluation dataset.
        """
        return self._check_accuracy(self.dataset.testing, "testing")
