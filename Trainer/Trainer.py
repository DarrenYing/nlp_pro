import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, model, train_loader, valid_loader, epochs=2, optimizer=None, criterion=None,
                 print_every=100, update_every=1, save_path=None, train_on_gpu=False):

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
        self.update_every = update_every
        if save_path == None:
            pass
            # self.save_path = ''
        self.save_path = save_path
        self.train_on_gpu = train_on_gpu


    def train(self):
        if self.train_on_gpu:
            self.model.cuda()

        counter = 0

        # train
        self.model.train()

        for epoch in range(self.epochs):

            # batch loop
            for inputs, labels in self.train_loader:
                counter += 1

                if (self.train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

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
                    for inputs, labels in self.valid_loader:

                        if (self.train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = self.model(inputs.long())
                        val_loss = self.criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    self.model.train()
                    print("Epoch: {}/{}...".format(epoch + 1, self.epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))



