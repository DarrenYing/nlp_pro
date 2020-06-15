import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from TrainTest.Model_controller import Controller

class Trainer(object):

    def __init__(self, model, train_loader, valid_loader, epochs=2, optimizer=None, criterion=None,
                 print_every=100, save_every=10, prefix=None):

        super(Trainer, self).__init__()

        # 参数类型检查
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be torch.nn.Module, got {type(model)}.")
        if not isinstance(train_loader, DataLoader):
            raise TypeError(f"The type of train_data must be torch.utils.data.DataLoader, got {type(train_loader)}.")
        if not isinstance(valid_loader, DataLoader):
            raise TypeError(f"The type of train_data must be torch.utils.data.DataLoader, got {type(valid_loader)}.")


        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        if optimizer == None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.optimizer = optimizer
        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion =criterion
        self.print_every = print_every
        self.save_every = save_every
        if prefix == None:
            prefix = 'checkpoints/models/'
            # self.save_path = ''
        self.prefix = prefix

        self.controller = Controller()


    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        counter = 0

        # train
        self.model.train()

        for epoch in range(self.epochs):

            # batch loop
            for inputs, labels in self.train_loader:
                counter += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero accumulated gradients
                self.model.zero_grad()

                # get the output from the model
                output = self.model(inputs.long())

                # calculate the loss and perform backprop
                loss = self.criterion(output.squeeze(), labels.float())
                loss.backward()
                self.optimizer.step()

                # loss stats
                if counter % self.print_every == 0:
                    # Get validation loss
                    val_losses = []
                    self.model.eval()
                    correct = 0
                    for inputs, labels in self.valid_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        output = self.model(inputs.long())
                        val_loss = self.criterion(output.squeeze(), labels.float())
                        val_losses.append(val_loss.item())

                        pre = torch.round(output.squeeze())
                        labels = labels.to(torch.float)
                        correct += (pre == labels).sum().item()

                    self.model.train()
                    acc = correct / len(self.valid_loader.dataset)
                    print('Accuracy: {}'.format(acc))
                    print("Epoch: {}/{}...".format(epoch + 1, self.epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

            if (epoch+1) % self.save_every == 0:
                save_path = '{}{}_{}.pkl'.format(self.prefix, self.model.netname, epoch+1)
                self.controller.store_param(self.model, save_path)






