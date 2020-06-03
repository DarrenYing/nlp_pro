import torch.nn as nn
import torch.nn.functional as F
import torch
from gensim.models import KeyedVectors


class Embedding_GoogleNews(nn.Module):

    def __init__(self, dropout=0):

        super(Embedding_GoogleNews, self).__init__()
        self.embed_lookup = KeyedVectors.load_word2vec_format(
            'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
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

