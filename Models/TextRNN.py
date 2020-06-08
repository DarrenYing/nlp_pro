import torch.nn as nn
import torch.nn.functional as F
import torch

from Modules import encoder


class TextRNN(nn.Module):

    def __init__(self, embed, hidden_size, num_layers=1,
                 bidirectional=True, dropout=0.5, output_dim=1):
        super(TextRNN, self).__init__()

        self.netname = "TextRNN"

        # 1. embedding layer
        self.embed = embed

        # 2. lstm
        self.lstm = encoder.LSTM(
            input_size = self.embed.embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            dropout = dropout
        )

        # 3. fc and dropout
        self.num_direction = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size*self.num_direction, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(output_dim)

    def forward(self, x):

        embeds = self.embed(x) # (batch, seq_len, embedding_dim)
        output, hx = self.lstm(embeds)  #output [batch, seq_len, hidden_size*num_direction]
                                        #hx     [num_layers*num_direction, batch, hidden_size]
        hx = self.dropout(
            torch.cat((hx[-2, :, :], hx[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        logit = self.fc(hx)

        return self.softmax(logit)

