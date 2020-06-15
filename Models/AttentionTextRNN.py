import torch.nn as nn
import torch.nn.functional as F
import torch

from Modules import encoder


class AttentionTextRNN(nn.Module):

    def __init__(self, embed, hidden_size, num_layers=1,
                 bidirectional=True, dropout=0.5, output_dim=1):
        super(AttentionTextRNN, self).__init__()

        self.netname = "AttentionTextRNN"

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

        num_direction = 2 if bidirectional else 1
        self.model_size = hidden_size*num_direction
        self.fc = nn.Linear(self.model_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(output_dim)

        n_head = 8
        d_k = d_v = int(self.model_size/n_head)
        self.atte = encoder.MultiHeadAttention(
            input_size=self.model_size,
            key_size=d_k,
            value_size=d_v,
            num_head=n_head,
            dropout=dropout
        )

        # self.W_w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        # self.u_w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        #
        # nn.init.uniform_(self.W_w, -0.1, 0.1)
        # nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        embeds = self.embed(x) # (batch, seq_len, embedding_dim)

        output, hidden = self.lstm(embeds) #batch_first
        # output    [batch, seq_len, hidden_size*num_direction]
        # hidden    [batch, num_layers*num_direction, hidden_size]

        output = self.atte(output, output, output)
        # out   [batch, seq_len_q, model_size] model_size=hidden_size*num_direction

        output = output + self.dropout(output)
        feat = torch.sum(output, dim=1)
        # feat  [batch_size, model_size]

        # """ tanh attention 的实现 """
        # score = torch.tanh(torch.matmul(output, self.W_w))
        # # score: [batch_size, seq_len, hidden_size * num_direction]
        #
        # attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        # # attention_weights: [batch_size, seq_len, 1]
        #
        # scored_x = output * attention_weights
        # # scored_x : [batch_size, seq_len, hidden_size * num_direction]

        # feat = torch.sum(scored_x, dim=1)
        # feat : [batch_size, hidden_size * num_direction]

        logit = self.fc(feat)

        return self.softmax(logit)
