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


def printten(data):
    for i in range(10):
        print(data[i])


def helplist():
    print("""default and reference:
    -h : help_list
    -m : model_name = {}
    -o : model_prefix_dir = {}
    --tr_dpath : train_data_path = {}
    --te_dpath : test_data_path = {}
    --epoch : epochs = {}
    --embedpath : embed_path = {}
    --lr : learn_rate = {}
    --ldpath : load_path = {}
    
    --mode :
    t : train_only
    p : predict_only
    a : train and predict
            
    """.format(configs.model_type, configs.model_prefix, configs.train_path, configs.test_path, configs.epochs,
               configs.embed_path, configs.lr,configs.load_path)

          )


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "-h-m:-o:", ["epoch=", "mode=", "--embedpath=", "--batchsize=", "--lr=","--ldpath="])
    except getopt.GetoptError:
        helplist()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            helplist()
            sys.exit(0)
        if opt == "--epoch":
            configs.epochs = int(arg)
        if opt == "--embedpath":
            configs.embed_path = arg
        if opt == "--batchsize":
            configs.batch_size = int(arg)
        if opt == "-m":
            configs.model_type = arg
        if opt == "--lr":
            configs.lr = float(arg)
        if opt == "--mode":
            configs.mode = arg
        if opt == "--ldpath":
            configs.load_path = arg
    print(opts, args)
    exit(0)

    ## Read csv data
    csv_train = CSVLoader(configs.train_path)
    csv_train.set_target_cols('label')
    csv_train.set_input_cols('txt')
    csv_train.data_init()
    sentences, labels = csv_train.sentences, csv_train.labels

    csv_test = CSVLoader(configs.test_path)
    csv_test.set_input_cols('txt')
    csv_test.data_init()
    test_sentences = csv_test.sentences

    # myEmbed = Embedding_GoogleNews()
    # opt.embed_path = 'word2vec_model/glove_vec/glove.6B.300d.txt'
    myEmbed = Embed_Loader(embed_path=configs.embed_path)
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
    train_loader, valid_loader = data_splitter.train_loader, data_splitter.valid_loader
    test_loader = data_splitter.get_onlytest(test_features, configs.batch_size)

    ## First checking if GPU is available

    ## Models
    output_size = 1  # binary class (1 or 0)
    num_filters = 100
    kernel_sizes = [3, 4, 5]

    if configs.model_type.lower() == "sc":
        net = SentimentCNN(myEmbed, output_size, num_filters, kernel_sizes)
    elif configs.model_type.lower() == "tr":
        net = TextRNN(myEmbed, hidden_size=100, num_layers=2, output_dim=output_size)
    elif configs.model_type.lower() == "atr":
        net = AttentionTextRNN(myEmbed, hidden_size=100, output_dim=output_size)
    elif configs.model_type.lower() == "blstma":
        net = BiLSTM_atte(myEmbed, hidden_size=100, output_dim=output_size)
    else:
        raise Exception("Wrong model arg,Plz check")

    print(net)

    ## Start Train
    # loss and optimization functions
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=configs.lr)

    configs.save_path = "./checkpoints/models/"

    trainer = Trainer(net, train_loader, valid_loader, epochs=configs.epochs,
                      optimizer=optimizer, criterion=criterion, print_every=configs.print_every,
                      prefix=configs.save_path)
    if configs.mode == "t" or configs.mode == "a":
        trainer.train()
    if configs.mode == "p" or configs.mode == "a":
        tester = Tester(net.eval(), configs.load_path, test_loader)
        pred_result = tester.predict()
        header = ['ID', 'Label']
        tester.save_to_csv(pred_result, header)
    else:
        raise Exception("Wrong mode arg,Plz check")


if __name__ == '__main__':
    main(sys.argv[1:])
