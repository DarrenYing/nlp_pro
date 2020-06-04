'''
仿照jupyter notebook的文本分类模型
简略框架
'''

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from Models import SentimentCNN, TextRNN, AttentionTextRNN
from Trainer.Trainer import Trainer
from Dataset import Embedding_GoogleNews, PreprocessTools, CSVLoader


def split_dataset(features, labels, split_frac):
    split_idx = int(len(features) * split_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = labels[:split_idx], labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    ## print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    return train_x, train_y, test_x, test_y, val_x, val_y

def get_dataloader(features, labels, split_frac, batch_size):
    train_x, train_y, test_x, test_y, val_x, val_y = \
        split_dataset(features, labels, split_frac=split_frac)
    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # shuffling and batching data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader

# 训练


def printten(data):
    for i in range(10):
        print(data[i])

def testsqueeze():
    a = torch.randn(50,1,200,300)
    conv = nn.Conv2d(1, 100, (3,300), padding=(1,0))
    b = conv(a)
    print(a.shape)
    print(b.shape)


if __name__ == '__main__':
    ## Read csv data
    path = "data/train.csv"
    csv_loader = CSVLoader(path)
    sentences, labels, headers = csv_loader.sentences, csv_loader.labels, csv_loader.headers

    ## Preprocess input
    ingore_cols = ['ID']
    input_cols = ['txt']

    # PreprocessTools
    pretool = PreprocessTools()

    texts, all_words = pretool.preprocess_input(sentences, input_cols)

    ## Removing outliers
    texts, labels = pretool.remove_outliers(texts, labels)

    ## Use pretrained embedding layer
    myEmbed = Embedding_GoogleNews()

    tokenized_texts = pretool.tokenize_all_texts(myEmbed.embed_lookup, texts)
    # print(tokenized_texts[0])

    ## Padding sequences
    seq_length = 200
    features = pretool.pad_features(tokenized_texts, seq_length)

    ## Split Train, Test, Validation Data
    ## Get Dataloaders
    split_frac = 0.8
    batch_size = 50
    train_loader, valid_loader, test_loader = \
        get_dataloader(features, labels, split_frac, batch_size)


    ## First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()

    if (train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    ## Models
    output_size = 1  # binary class (1 or 0)
    num_filters = 100
    kernel_sizes = [3, 4, 5]

    # net = SentimentCNN(myEmbed, output_size, num_filters, kernel_sizes)
    # net = TextRNN(myEmbed, hidden_size=100, output_dim=output_size)
    net = AttentionTextRNN(myEmbed, hidden_size=100, output_dim=output_size)

    print(net)

    ## Start Train
    # loss and optimization functions
    lr = 0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epochs = 2
    print_every = 100

    trainer = Trainer(net, train_loader, valid_loader, epochs=epochs,
                      optimizer=optimizer, criterion=criterion, print_every=print_every)

    trainer.train()

