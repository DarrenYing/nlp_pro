
import math
import torch
import torch.nn.functional as F
from torch import nn


class DotAttention(nn.Module):
    """
    DotAttention，作为MultiHeadAttention的组成部分
    """

    def __init__(self, key_size, value_size, dropout=0.0):
        '''
        :param key_size:   键的维度
        :param value_size: 一般和key_size相等
        :param dropout:
        '''
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask_out=None):
        """
        :param Q: [..., seq_len_q, key_size]
        :param K: [..., seq_len_k, key_size]
        :param V: [..., seq_len_k, value_size]
        :param mask_out: [..., 1, seq_len] or [..., seq_len_q, seq_len_k]
        """
        output = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -1e9)
        output = self.softmax(output)
        output = self.drop(output)
        return torch.matmul(output, V)


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention，即多个DotAttention堆叠
    """

    def __init__(self, input_size, key_size, value_size, num_head, dropout=0.1):
        """

        :param input_size: int, 输入维度的大小。同时也是输出维度的大小。
        :param key_size: int, 每个head的维度大小。
        :param value_size: int，每个head中value的维度。
        :param num_head: int，head的数量。
        :param dropout: float。
        """
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_head = num_head

        in_size = key_size * num_head
        self.q_in = nn.Linear(input_size, in_size)
        self.k_in = nn.Linear(input_size, in_size)
        self.v_in = nn.Linear(input_size, in_size)
        self.attention = DotAttention(key_size=key_size, value_size=value_size, dropout=dropout)
        self.out = nn.Linear(value_size * num_head, input_size)

        self.reset_parameters() #初始化参数

    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(1.0 / self.input_size))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(1.0 / self.input_size))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(1.0 / self.input_size))
        nn.init.normal_(self.out.weight, mean=0, std=sqrt(1.0 / self.input_size))

    def forward(self, Q, K, V, atte_mask_out=None):
        """

        :param Q: [batch, seq_len_q, model_size]
        :param K: [batch, seq_len_k, model_size]
        :param V: [batch, seq_len_k, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, sq, _ = Q.size()
        sk = K.size(1)
        d_k, d_v, n_head = self.key_size, self.value_size, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, sq, n_head, d_k).transpose(1, 2)
        k = self.k_in(K).view(batch, sk, n_head, d_k).transpose(1, 2)
        v = self.v_in(V).view(batch, sk, n_head, d_v).transpose(1, 2)

        if atte_mask_out is not None:
            atte_mask_out = atte_mask_out[:, None, :, :]  # [bsz,1,1,len]
        atte = self.attention(q, k, v, atte_mask_out).view(batch, n_head, sq, d_v)

        # concat all heads, do output linear
        atte = atte.transpose(1, 2).contiguous().view(batch, sq, -1)
        output = self.out(atte) # [batch, seq_len_q, num_head*key_size]
        return output           # [batch, seq_len_q, model_size]






