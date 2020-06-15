import torch
from TrainTest import Controller
import csv
import numpy as np

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
            for inputs in self.test_loader:
                inputs = inputs[0]
                inputs = inputs.to(device)
                output = self.model(inputs.long())
                pre = torch.round(output.squeeze())
                pre = pre.to(torch.int)
                pre_res.extend(pre.numpy().tolist())
        return pre_res

    def predict_valid(self, valid_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        pre_res = []
        with torch.no_grad():  # when in test stage, no grad
            correct = 0
            for (inputs, labels) in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.to(torch.int64)
                output = self.model(inputs)
                pre = torch.round(output.squeeze())
                pre = pre.to(torch.int)
                # labels = labels.to(torch.float)
                correct += (pre == labels).sum().item()
                pre_res.extend(pre.numpy().tolist())
            acc = correct / len(self.test_loader.dataset)
            print('Accuracy: {}'.format(acc))
        return pre_res, acc

    def save_to_csv(self, result, header, start_id=25000):
        path = "data/predict_result.csv"
        csvfile = open(path, "w", newline="")
        writer = csv.writer(csvfile)
        # 写入header
        writer.writerow(header)
        # 写入预测结果
        ids = list(range(start_id, start_id+len(result)))
        zipped_res = zip(ids, result)
        writer.writerows(zipped_res)
        csvfile.close()







