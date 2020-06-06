import torch
from torch.utils.data import TensorDataset, DataLoader


class DataSplitter(object):

    def __init__(self, features, labels, batch_size=50, split_frac=0.8, test_split=False, test_frac=0.5):
        '''
        :param features: 处理好的特征
        :param labels: 标签
        :param split_frac: 训练集比例
        :param test_split: 是否需要划分测试集
        :param test_frac: 测试集和验证集占比，0.5表示test:val=1:1
        '''
        super(DataSplitter, self).__init__()
        self.features = features
        self.labels = labels
        self.split_frac = split_frac
        self.test_split = test_split
        self.test_frac = test_frac
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None

        self.get_dataloader(batch_size)

    def split_dataset(self):
        split_idx = int(len(self.features) * self.split_frac)
        train_x, remaining_x = self.features[:split_idx], self.features[split_idx:]
        train_y, remaining_y = self.labels[:split_idx], self.labels[split_idx:]

        test_x = []
        test_y = []
        if self.test_split:
            test_idx = int(len(remaining_x) * self.test_frac)
            val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
            val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
        else:
            val_x = remaining_x
            val_y = remaining_y

        ## print out the shapes of your resultant feature data
        print("\t\t\tFeature Shapes:")
        print("Train set: \t\t{}".format(train_x.shape),
              "\nValidation set: \t{}".format(val_x.shape)),
        if self.test_split:
              print("Test set: \t\t{}".format(test_x.shape))

        return train_x, train_y, test_x, test_y, val_x, val_y

    def get_dataloader(self, batch_size):
        train_x, train_y, test_x, test_y, val_x, val_y = \
            self.split_dataset()
        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        if self.test_split:
            test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
            self.test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

        # shuffling and batching data
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
