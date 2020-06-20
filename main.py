'''
仿照jupyter notebook的文本分类模型
简略框架
'''

import numpy as np
import sys, getopt

import torch
import torch.nn as nn

from Models import *
from TrainTest import Trainer, DataSplitter, Tester
from Dataset import configs, CSVLoader, Utils, Embedding_GoogleNews, Embed_Loader
from Dataset.config import loss_func_dict
import fire

def printten(data):
    for i in range(10):
        print(data[i])

def main(**kwargs):
    ## Read csv data

    configs._parse(kwargs)

    csv_train = CSVLoader(configs.train_path)
    csv_train.set_target_cols(configs.label_col)
    csv_train.set_input_cols(configs.input_col)
    csv_train.data_init()
    sentences, labels = csv_train.sentences, csv_train.labels

    csv_test = CSVLoader(configs.test_path)
    csv_test.set_input_cols('txt')
    csv_test.data_init()
    test_sentences = csv_test.sentences

    myEmbed = Embedding_GoogleNews()
    # opt.embed_path = 'word2vec_model/glove_vec/glove.6B.300d.txt'
    # myEmbed = Embed_Loader(embed_path=configs.embed_path)
    seq_length = configs.seq_length

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
    data_splitter = DataSplitter(features, labels, split_frac=configs.split_frac,
                                 batch_size=configs.batch_size)

    test_loader = data_splitter.get_onlytest(test_features, configs.batch_size)

    ## First checking if GPU is available

    ## Models
    output_size = 1  # binary class (1 or 0)
    num_filters = 100
    kernel_sizes = [3, 4, 5]

    if configs.model_type.lower() == "sc":
        net = SentimentCNN(myEmbed, output_size, num_filters, kernel_sizes,dropout=configs.dropout)
    elif configs.model_type.lower() == "tr":
        net = TextRNN(myEmbed, hidden_size=100, num_layers=1, output_dim=output_size,dropout=configs.dropout)
    elif configs.model_type.lower() == "atr":
        net = AttentionTextRNN(myEmbed, hidden_size=100, output_dim=output_size,dropout=configs.dropout)
    elif configs.model_type.lower() == "blstma":
        net = BiLSTM_atte(myEmbed, hidden_size=200, num_layers=1, output_dim=output_size,dropout=configs.dropout)
    else:
        raise Exception("Wrong model arg,Plz check")

    print(net)

    ## Start Train
    # loss and optimization functions
    if not loss_func_dict[configs.loss_func.lower()]:
        raise Exception("Wrong Loss func arg! Plz check again")
    else:
        criterion = loss_func_dict[configs.loss_func]()

    if configs.optimizer.lower()=="sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=configs.lr)
    elif configs.optimizer.lower() == "mot":
        optimizer = torch.optim.SGD(net.parameters(), lr=configs.lr, momentum=0.8, nesterov=True)
    elif configs.optimizer.lower() == "rms":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=configs.lr, alpha=0.9)
    elif configs.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=configs.lr, betas=(0.9, 0.99))
    elif configs.optimizer.lower() == "adg":
        optimizer = torch.optim.Adagrad(net.parameters(), lr=configs.lr)
    else:
        raise Exception("Wrong optimizer arg! Plz check again")

    configs.save_path = "./checkpoints/models/"

    trainer = Trainer(net, data_splitter, epochs=configs.epochs,
                      optimizer=optimizer, criterion=criterion, print_every=configs.print_every,
                      prefix=configs.save_path)
    if configs.mode == "t" or configs.mode == "a":
        trainer.train()
    if configs.mode == "p" or configs.mode == "a":
        tester = Tester(net.eval(), configs.load_path, test_loader)
        pred_result = tester.predict()
        print(pred_result)
    if configs.mode != "p" and configs.mode != "t" and configs.mode != "a":
        raise Exception("Wrong mode arg,Plz check")




if __name__ == '__main__':
    fire.Fire(main)
    # main(sys.argv[1:])
