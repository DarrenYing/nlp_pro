import torch.nn as nn
import torch.nn.functional as F
import torch
from gensim.models import KeyedVectors


class Embed_Loader(nn.Module):

    def __init__(self, embed_path, dropout=0):

        super(Embed_Loader, self).__init__()
        self.binary = False
        if embed_path.endswith('bin'):
            self.binary = True
        self.embed_lookup = KeyedVectors.load_word2vec_format(
            embed_path, binary=self.binary)
        pretrained_words = []
        for word in self.embed_lookup.vocab:
            pretrained_words.append(word)
        self.vocab_size = len(pretrained_words)
        self.embedding_dim = len(self.embed_lookup[pretrained_words[0]])

        print(self.vocab_size, self.embedding_dim)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(self.embed_lookup.vectors))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed =  self.embedding(x)
        return self.dropout(embed)

