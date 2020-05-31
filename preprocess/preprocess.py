import numpy as np
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from string import punctuation


def get_stopwords():
    stop_words = set(stopwords.words('english'))
    return stop_words

def deletebr(line):
    new_line = re.sub(r'<br\s*.?>', r'', line)
    return new_line

def printten(data):
    for i in range(10):
        print(data[i])


def preprocess_labels(labels):
    if labels[0] == "positive" or labels[0] == "negtive":
        #labels转成数字类型
        labels = np.array([1 if label == 'positive' else 0 for label in labels])
    elif labels[0] == "1" or labels[0] == "0":
        labels = np.array([1 if label == '1' else 0 for label in labels])
    return labels



def preprocess_input(data, input_cols):
    texts = []
    all_words  = []
    stop_words = get_stopwords()
    for line in data:
        #每行是一个字典
        for key in line:
            #如果是key是'txt'
            if key in input_cols:
                #txt进行处理，把句子转化成单词的list
                new_line = deletebr(str(line[key]))
                words = ''.join([c for c in new_line if c not in punctuation])
                word_tokens = word_tokenize(words)
                filtered_words = [w for w in word_tokens if w not in stop_words]
                texts.append(filtered_words)
                all_words.extend(filtered_words)
    return texts, all_words


# 转为在embed_lookup中的idx序列
def tokenize_all_texts(embed_lookup, texts):
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