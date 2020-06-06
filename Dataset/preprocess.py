import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from collections import Counter

'''
数据预处理操作集合
'''

class PreprocessTools():

    def __init__(self):
        pass


    def preprocess_labels(self, labels):
        if labels[0] == "positive" or labels[0] == "negtive":
            #labels转成数字类型
            labels = np.array([1 if label == 'positive' else 0 for label in labels])
        elif labels[0] == "1" or labels[0] == "0":
            labels = np.array([1 if label == '1' else 0 for label in labels])
        return labels



    def preprocess_input(self, data, input_cols):
        texts = []
        all_words  = []
        stop_words = self.get_stopwords()
        for line in data:
            #每行是一个字典
            for key in line:
                #如果是key是'txt'
                if str(key).lower() in input_cols:
                    #txt进行处理，把句子转化成单词的list
                    new_line = self.deletebr(str(line[key]))
                    words = ''.join([c for c in new_line if c not in punctuation])
                    word_tokens = word_tokenize(words)
                    filtered_words = [w for w in word_tokens if w not in stop_words]
                    texts.append(filtered_words)
                    all_words.extend(filtered_words)
        return texts, all_words


    # 转为在embed_lookup中的idx序列
    def tokenize_all_texts(self, embed_lookup, texts):
        tokenized_texts = []
        for line in texts:
            ints = []
            for word in line:
                try:
                    idx = embed_lookup.vocab[word].index
                except:
                    idx = 0
                ints.append(idx)
            tokenized_texts.append(ints)

        return tokenized_texts

    # 去除outliers
    def remove_outliers(self, sentences, labels, minlen=0, istest=False):
        sentence_lens = Counter([len(x) for x in sentences])
        # if print_sentence_len == True:
        #     print(sentence_lens)
        #     print("Minimum review length: {}".format(min(sentence_lens)))
        #     print("Maximum review length: {}".format(max(sentence_lens)))
        if min(sentence_lens) <= minlen:
            # non_zero_idx = [ii for ii, sent in enumerate(self.sentences) if len(sent) != 0]
            # self.sentences = [self.sentences[ii] for ii in non_zero_idx]
            # self.labels = np.array([self.sentences[ii] for ii in non_zero_idx])
            zero_idx = [ii for ii, sent in enumerate(sentences) if len(sent) == 0]
            if istest:
                for i in zero_idx:
                    sentences.pop(i)
            else:
                for i in zero_idx:
                    sentences.pop(i)
                    labels.pop(i)

        return sentences, labels

    def pad_features(self, tokenized_texts, seq_length):
        '''
        长度不足的补0，多出的截断
        '''
        features = np.zeros((len(tokenized_texts), seq_length), dtype=int)

        for i, row in enumerate(tokenized_texts):
            features[i, -len(row):] = np.array(row)[:seq_length]

        return features


    def get_stopwords(self):
        stop_words = set(stopwords.words('english'))
        return stop_words

    def deletebr(self, line):
        new_line = re.sub(r'<br\s*.?>', r'', line)
        return new_line

    def printten(self, data):
        for i in range(10):
            print(data[i])