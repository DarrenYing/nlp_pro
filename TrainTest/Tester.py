import torch
from TrainTest import Controller

class Tester(object):

    def __init__(self, model, model_path, test_loader):

        super(Tester, self).__init__()
        self.controller = Controller()
        self.model = self.controller.load_param(model, model_path)
        self.test_loader = test_loader


    def predict(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        pre_res = []
        with torch.no_grad():  # when in test stage, no grad
            correct = 0
            for (inputs, labels) in self.test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.to(torch.int64)
                output = self.model(inputs)
                pre = torch.round(output.squeeze())
                correct += (pre == labels).sum().item()
                pre_res.append(pre)
            acc = correct / len(self.test_loader.dataset)
            print('Accuracy: {}'.format(acc))
        return pre_res, acc


