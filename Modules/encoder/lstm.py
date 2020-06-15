import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class LSTM(nn.Module):
    r"""
    LSTM 模块, 轻量封装的Pytorch LSTM. 在提供seq_len的情况下，将自动使用pack_padded_sequence; 同时默认将forget gate的bias初始化
    为1; 且可以应对DataParallel中LSTM的使用问题。

    """

    def __init__(self, input_size, hidden_size=100, num_layers=1, dropout=0.0, batch_first=True,
                 bidirectional=False, bias=True):
        r"""

        :param input_size:  输入 `x` 的特征维度
        :param hidden_size: 隐状态 `h` 的特征维度. 如果bidirectional为True，则输出的维度会是hidden_size*2
        :param num_layers: rnn的层数. Default: 1
        :param dropout: 层间dropout概率. Default: 0
        :param bidirectional: 若为 ``True``, 使用双向的RNN. Default: ``False``
        :param batch_first: 若为 ``True``, 输入和输出 ``Tensor`` 形状为
            :(batch, seq, feature). Default: ``False``
        :param bias: 如果为 ``False``, 模型将不会使用bias. Default: ``True``
        """
        super(LSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.init_param()

    ## 初始化bias
    def init_param(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                # based on https://github.com/pytorch/pytorch/issues/750#issuecomment-280671871
                param.data.fill_(0)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1)
            else:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        output, (hidden, _) = self.lstm(x)
        return output, hidden