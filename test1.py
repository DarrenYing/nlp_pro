'''
仿照jupyter notebook的文本分类模型
简略框架
'''

import numpy as np

import torch
import torch.nn as nn

from Models import SentimentCNN, TextRNN, AttentionTextRNN
from Trainer import Trainer, DataSplitter
from Dataset import Embedding_GoogleNews, Config, CSVLoader, Utils


def printten(data):
    for i in range(10):
        print(data[i])


if __name__ == '__main__':
    opt = Config()
    ## Read csv data
    csv_train = CSVLoader(opt.train_path)
    csv_train.set_target_cols('label')
    csv_train.set_input_cols('txt')
    csv_train.data_init()
    sentences, labels = csv_train.sentences, csv_train.labels

    csv_test = CSVLoader(opt.test_path)
    csv_test.set_input_cols('txt')
    csv_test.data_init()
    test_sentences = csv_test.sentences
    ## Preprocess input
    # ingore_cols = ['ID']
    # input_cols = ['txt']

    myEmbed = Embedding_GoogleNews()
    seq_length = opt.seq_length

    # Preprocess
    utils = Utils()
    features, labels = utils.preprocess(
        sentences=sentences,
        labels=labels,
        istest=False,
        input_cols=csv_train.input_cols,
        embed=myEmbed,
        seq_len=seq_length
    )

    test_features, _ = utils.preprocess(
        sentences=test_sentences,
        labels=[],
        istest=True,
        input_cols=csv_test.input_cols,
        embed=myEmbed,
        seq_len=seq_length
    )


    ## Split Train, Test, Validation Data
    ## Get Dataloaders
    data_splitter = DataSplitter(features, labels, split_frac=opt.split_frac,
                                 batch_size=opt.batch_size)
    train_loader, valid_loader = data_splitter.train_loader, data_splitter.valid_loader

    ## First checking if GPU is available
    train_on_gpu = utils.check_gpu()

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
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    trainer = Trainer(net, train_loader, valid_loader, epochs=opt.epochs,
                      optimizer=optimizer, criterion=criterion, print_every=opt.print_every)
    trainer.train()

