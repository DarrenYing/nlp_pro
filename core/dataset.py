
__all__ = [
    "DataSet",
]

from collections import Counter

import numpy as np
from ..preprocess.labels import preprocess_labels
from ..preprocess.preprocess import preprocess_input
from ..preprocess.preprocess import tokenize_all_texts
from gensim.models import KeyedVectors

class ApplyResultException(Exception):
    def __init__(self, msg, index=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了


class DataSet(object):
    r"""
    fastNLP的数据容器，详细的使用方法见文档  :mod:`fastNLP.core.dataset`
    """

    def __init__(self, data=None):
        r"""
        sentencs 为 句子文本
        labels 为 句子的标签
        headers 为 属性的头
        all_words 为 所有训练集的词典
        """
        self.sentences = []
        self.labels = []
        self.headers = []
        self.all_words = []

    def preprocess_labels(self):
        # preprocess_labels(self.labels)
        if self.labels[0] == "positive" or self.labels[0] == "negtive":
            # labels转成数字类型
            labels = np.array([1 if label == 'positive' else 0 for label in self.labels])
        elif self.labels[0] == "1" or self.labels[0] == "0":
            labels = np.array([1 if label == '1' else 0 for label in self.labels])
        return labels


    #把sentences转成words
    def preprocess_input(self):
        input_cols = ['txt']
        self.sentences, self.all_words = preprocess_input(self.sentences,input_cols)


    #remove_zero 是否去除空字符串
    #print_sentence_len 是否要打印查看每个句子的长度
    def remove_null_string(self,remove_zero = True,print_sentence_len = False):
        if remove_zero == True:
            sentence_lens = Counter([len(x) for x in self.sentences])
            if print_sentence_len == True:
                print(sentence_lens)
                print("Minimum review length: {}".format(min(sentence_lens)))
                print("Maximum review length: {}".format(max(sentence_lens)))
            if min(sentence_lens) == 0:
                # non_zero_idx = [ii for ii, sent in enumerate(self.sentences) if len(sent) != 0]
                # self.sentences = [self.sentences[ii] for ii in non_zero_idx]
                # self.labels = np.array([self.sentences[ii] for ii in non_zero_idx])
                zero_idx = [ii for ii, sent in enumerate(self.sentences) if len(sent) == 0]
                for i in zero_idx:
                    self.sentences.pop(i)
                    self.labels.pop(i)

            # 去除空字符串

    def embedding(self,methods = "static"):
        if methods == "static":
            embed_lookup = KeyedVectors.load_word2vec_format(
                'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
            # store pretrained vocab
            pretrained_words = []
            for word in embed_lookup.vocab:
                pretrained_words.append(word)

        self.sentences = tokenize_all_texts(embed_lookup, self.sentences)



    def append(self, sentences, labels, headers):
        r"""
        将一个instance对象append到DataSet后面。

        :param ~fastNLP.Instance instance: 若DataSet不为空，则instance应该拥有和DataSet完全一样的field。

        """
        self.sentences = sentences
        self.labels = labels
        self.headers = headers

