import torch.nn as nn
import torch.nn.functional as F
import torch
from gensim.models import KeyedVectors
from gensim.downloader import api

class Embedding_GoogleNews(nn.Module):

    ###parameters：
    ###dropout dropout层的概率
    ###download_model_name 下载模型的名称   example:glove-twitter-25
    ###model_path  已经下载好的语料模型的路径
    def __init__(self, dropout=0,download_model_name=None,model_path = 'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin'):

        super(Embedding_GoogleNews, self).__init__()
        #如果需要下载模型
        if(download_model_name != None):
            self.embed_lookup = api.load(download_model_name)
        else:
            if model_path.endswith("bin"):
                self.embed_lookup = KeyedVectors.load_word2vec_format(model_path, binary=True)
            else:
                self.embed_lookup = KeyedVectors.load_word2vec_format(model_path, binary=False)
        pretrained_words = []
        for word in self.embed_lookup.vocab:
            pretrained_words.append(word)
        self.vocab_size = len(pretrained_words)
        self.embedding_dim = len(self.embed_lookup[pretrained_words[0]])

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(self.embed_lookup.vectors))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed =  self.embedding(x)
        return self.dropout(embed)

