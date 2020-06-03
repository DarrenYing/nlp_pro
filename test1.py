'''
仿照jupyter notebook的文本分类模型
简略框架
'''

import numpy as np
import csv
from string import punctuation
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

from gensim.models import KeyedVectors

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from Models import SentimentCNN, TextRNN, AttentionTextRNN
from Trainer.Trainer import Trainer
from Dataset import Embedding_GoogleNews

def read_csv(path, encoding='utf-8-sig', headers=None, sep=',', dropna=True):
    with open(path, 'r', encoding=encoding) as csv_file:
        f = csv.reader(csv_file, delimiter=sep)
        start_idx = 0

        if headers is None:
            headers = next(f)
            # print(headers)
            start_idx += 1

        # ID,txt,Label
        sentences = []
        labels = []
        for line_idx, line in enumerate(f, start_idx):
            contents = line

            _dict = {}
            for header, content in zip(headers, contents):
                if str(header).lower() == "label":
                    labels.append(content)
                else:
                    _dict[header] = str(content).lower()    #小写
            sentences.append(_dict)

    return sentences, labels, headers

def preprocess_input(data, input_cols):
    texts = []
    all_words  = []
    stop_words = get_stopwords()
    for line in data:
        #每行是一个字典
        for key in line:
            if key in input_cols:
                new_line = deletebr(str(line[key]))
                words = ''.join([c for c in new_line if c not in punctuation])
                word_tokens = word_tokenize(words)
                filtered_words = [w for w in word_tokens if w not in stop_words]
                texts.append(filtered_words)
                all_words.extend(filtered_words)
    return texts, all_words

def preprocess_labels(data):
    pass

# 转为在embed_lookup中的idx序列
def tokenize_all_texts(embed_lookup, texts):
    tokenized_texts = []
    for line in texts:
        ints = []
        for word in line:
            try:
                idx = embed_lookup.vocab[word].index
            except:
                idx = 0
            ints.append(idx)
        tokenized_texts.append(ints)

    return tokenized_texts


def pad_features(tokenized_texts, seq_length):
    '''
    长度不足的补0，多出的截断
    '''
    features = np.zeros((len(tokenized_texts), seq_length), dtype=int)

    for i, row in enumerate(tokenized_texts):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

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


def get_stopwords():
    stop_words = set(stopwords.words('english'))
    return stop_words

def deletebr(line):
    new_line = re.sub(r'<br\s*.?>', r'', line)
    return new_line

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
    sentences, labels, headers = read_csv(path)
    labels = np.array([1 if label == '1' else 0 for label in labels])

    ## Preprocess input
    ingore_cols = ['ID']
    input_cols = ['txt']

    texts, all_words = preprocess_input(sentences, input_cols)
    # printten(texts)
    # printten(all_words)
    # print(len(labels), type(labels))

    ## Removing outliers
    sentence_lens = Counter([len(x) for x in texts])
    # print(sentence_lens)
    # print("Minimum review length: {}".format(min(sentence_lens)))
    # print("Maximum review length: {}".format(max(sentence_lens)))
    # 去除空字符串
    non_zero_idx = [ii for ii, sent in enumerate(texts) if len(sent) != 0]
    texts = [texts[ii] for ii in non_zero_idx]
    labels = np.array([labels[ii] for ii in non_zero_idx])

    # print(len(texts), len(labels))

    ## Use pretrained embedding layer
    myEmbed = Embedding_GoogleNews()

    tokenized_texts = tokenize_all_texts(myEmbed.embed_lookup, texts)
    # print(tokenized_texts[0])


    ## Padding sequences
    seq_length = 200
    features = pad_features(tokenized_texts, seq_length=seq_length) #左侧填充0

    # print(len(features))
    # printten(features)

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

