
import torch
import torch.nn as nn

class MLP(nn.Module):
    '''
    多层感知器
    就是多个全连接层
    '''

    def __init__(self, size_layer, activation='relu', output_activation=None, initial_method=None, dropout=0.0):
        '''
        :param List[int] size_layer: 一个int的列表，用来定义MLP的层数，列表中的数字为每一层是hidden数目。MLP的层数为 len(size_layer) - 1
        :param Union[str,func,List[str]] activation: 一个字符串或者函数的列表，用来定义每一个隐层的激活函数，字符串包括relu，tanh和
            sigmoid，默认值为relu
        :param Union[str,func] output_activation:  字符串或者函数，用来定义输出层的激活函数，默认值为None，表示输出层没有激活函数
        :param str initial_method: 参数初始化方式
        :param float dropout: dropout概率，默认值为0
        '''
        super(MLP, self).__init__()
        self.hiddens = nn.ModuleList()
        self.output = None

        # 隐藏层
        for i in range(1, len(size_layer)):
            if i + 1 == len(size_layer):
                self.output = nn.Linear(size_layer[i - 1], size_layer[i]) #最后一层即输出层
            else:
                self.hiddens.append(nn.Linear(size_layer[i - 1], size_layer[i]))

        self.dropout = nn.Dropout(p=dropout)

        self.hidden_active = [] #隐藏层的激活函数
        actives = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }

        if not isinstance(activation, list): #每个隐藏层配一个激活函数，输入输出层不算在内
            activation = [activation] * (len(size_layer) - 2)

        for func in activation:
            if func.lower() in actives:
                self.hidden_active.append(actives[func])
            elif callable(activation): #可调用的
                self.hidden_active.append(activation)
            else:
                raise ValueError("{} 激活函数不正确".format(activation))

        # 输出层激活函数
        self.output_activation = output_activation # 可以是None
        if self.output_activation is not None:
            if output_activation.lower() in actives:
                self.output_activation = actives[output_activation.lower()]

        #参数初始化
        pass


    def forward(self, x):
        for layer, active_func in zip(self.hiddens, self.hidden_active):
            x = self.dropout(active_func(layer(x)))
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        x = self.dropout(x)
        return x