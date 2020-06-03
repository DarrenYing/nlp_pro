__all__ = [
    "ConvMaxpool"
]

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMaxpool(nn.Module):
    """
    集合了Convolution和Max-Pooling于一体的层。给定一个batch_size x max_len x input_size的输入，
    返回batch_size x, sum(output_channels) 大小的matrix。
    在内部，是先使用CNN给输入做卷积，然后经过activation激活层，
    再通过在长度(max_len)这一维进行max_pooling。最后得到每个sample的一个向量表示。

    """

    def __init__(self, in_channels, out_channels, kernel_sizes, activation="relu", embedding_dim=None):
        super(ConvMaxpool, self).__init__()

        if isinstance(kernel_sizes, (list, tuple, int)):
            if isinstance(kernel_sizes, int) and isinstance(out_channels, int):
                #都是整数，扩展为列表
                out_channels = [out_channels]
                kernel_sizes = [kernel_sizes]
            elif isinstance(kernel_sizes, (tuple, list)) and isinstance(out_channels, (tuple, list)):
                #都是元组或列表，长度必须相等
                assert len(out_channels) == len(
                    kernel_sizes), "The number of out_channels should be equal to the number" \
                                   " of kernel_sizes."
            else:
                raise ValueError("The type of out_channels and kernel_sizes should be the same.")

        self.convs = None
        if embedding_dim is not None:
            self.convs = nn.ModuleList([nn.Conv2d(
                in_channels=in_channels,
                out_channels=oc,
                kernel_size=(ks, embedding_dim),     # (ksize, embedding_size)
                stride=1,
                padding=(ks-2, 0),  # embedding的维度不pad
                dilation=1,
                groups=1,
                bias=None)
                for oc, ks in zip(out_channels, kernel_sizes)])
        else:
            self.convs = nn.ModuleList([nn.Conv2d(
                in_channels=in_channels,
                out_channels=oc,
                kernel_size=ks,  # (ks, ks)
                stride=1,
                padding=(ks - 2, 0),  # embedding的维度不pad
                dilation=1,
                groups=1,
                bias=None)
                for oc, ks in zip(out_channels, kernel_sizes)])

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh
        else:
            raise Exception(
                "Undefined activation function: choose from: relu, tanh, sigmoid")

    def forward(self, x):
        '''
        :param x: embedding后的输入，(batch_size, seq_length, embedding_dim)
        :return:
        '''
        # [50, 1, 200, 300] -> [50, 100, 200, 1]
        # [batch_size, in_channel, H=seq_length, W=embedding_dim]
        # [batch_size, out_channel, H_out, W_out]

        # convolution
        #(batch_size, out_channels=num_filters, seq_length)
        xs = [self.activation(conv(x)).squeeze(3) for conv in self.convs]
        # max_pool
        #(batch_size, out_channels=num_filters)
        xs_max = [F.max_pool1d(input=i, kernel_size=i.size(2)).squeeze(2)
              for i in xs]

        return torch.cat(xs_max, dim=-1)






