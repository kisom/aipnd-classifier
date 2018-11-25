import datetime
import torch

class Gym():

    def __init__(self, model, dataset, print_every):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.model = model
        self.dataset = dataset
        self.print_every = print_every

    def train(self):
        print("starting deep learning")

        # Only train the classifier parameters, feature parameters are frozen.
        epochs = self.model.hyper_params["epochs"]

        print_every = self.print_every
        steps = 0

        training_started = datetime.datetime.now()
    
        self.model.network.to(self.device) # I consider the time it takes to move a model to the GPU
                        # as part of the time to train.

        for e in range(epochs):
            print("starting epoch:", e)
            running_loss = 0
            epoch_started = datetime.datetime.now()
            tdelta = datetime.datetime.now()
            for (inputs, labels) in iter(self.dataset.training):
                self.model.network.train()
                steps += 1

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.model.optimizer.zero_grad()

                # Forward and backward passes
                outputs = self.model.forward(inputs)
                loss = self.model.criterion(outputs, labels)
                loss.backward()
                self.model.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    print(
                        "{} {:6}|".format(datetime.datetime.now() - tdelta, steps),
                        "epoch: {}/{}... ".format(e + 1, epochs),
                        "loss: {:.4f}".format(running_loss / print_every),
                    )

                    tdelta = datetime.datetime.now()
                    running_loss = 0

            self.check_accuracy(self.dataset.validation, 'validation')
            print("epoch completed in: {}".format(datetime.datetime.now() - epoch_started))
            print("-" * 72)

        print("training completed in {}".format(datetime.datetime.now() - training_started))

    def check_accuracy(self, dataset, datalabel):
        self.model.network.eval()
        max_fn = torch.max
        # if torch.cuda.is_available():
        #     max_fn = torch.cuda.max
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataset:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model.network(inputs)
                _, predicted = max_fn(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            "{} accuracy over {} test images: {:0.4}% ({}/{})".format(
                datalabel, total, (100 * correct / total), correct, total,
            )
        )

        return correct / total
