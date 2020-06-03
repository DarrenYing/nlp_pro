import torch.nn as nn
import torch.nn.functional as F
import torch

from Modules import encoder


class AttentionTextRNN(nn.Module):

    def __init__(self, embed, hidden_size, num_layers=1,
                 bidirectional=True, dropout=0.5, output_dim=1):
        super(AttentionTextRNN, self).__init__()

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

        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(output_dim)

        self.W_w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        embeds = self.embed(x) # (batch, seq_len, embedding_dim)

        output, hidden = self.lstm(embeds)
        # output    [batch, seq_len, hidden_size*num_direction]
        # hidden    [num_layers*num_direction, batch, hidden_size]

        """ tanh attention 的实现 """
        score = torch.tanh(torch.matmul(output, self.W_w))
        # score: [batch_size, seq_len, hidden_size * num_direction]

        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        # attention_weights: [batch_size, seq_len, 1]

        scored_x = output * attention_weights
        # scored_x : [batch_size, seq_len, hidden_size * num_direction]

        feat = torch.sum(scored_x, dim=1)
        # feat : [batch_size, hidden_size * num_direction]
        logit = self.fc(feat)

        return self.softmax(logit)
