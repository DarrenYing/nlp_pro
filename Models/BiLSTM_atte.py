import torch.nn as nn
import torch.nn.functional as F
import torch

from Modules import encoder

class BiLSTM_atte(nn.Module):

    def __init__(self, embed, hidden_size, num_layers=1,
                dropout=0.5, output_dim=1):
        super(BiLSTM_atte, self).__init__()
        self.netname = "BiLSTM_atte"

        self.embed = embed

        self.output_dim = output_dim

        self.lstm = encoder.LSTM(
            input_size=self.embed.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout
        )

        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )

        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_dim),
            nn.Sigmoid()
        )


    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]

        lstm_hidden = torch.sum(lstm_hidden, dim=1) # [batch_size, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1) # [batch_size, 1, n_hidden]

        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):

        sen_input = self.embed(x)
        # sen_input : [batch_size, len_seq, embedding_dim]

        output, hidden_state = self.lstm(sen_input)
        # output : [batch_size, len_seq, hidden_size * 2]
        # hidden_state : [batch_size, num_layers*2, hidden_size]
        hidden_state = hidden_state.permute(1, 0, 2)

        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        atten_out = self.attention_net_with_w(output, hidden_state)
        return self.fc_out(atten_out)

