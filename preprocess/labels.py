import numpy as np

def preprocess_labels(labels):
    if labels[0] == "positive" or labels[0] == "negtive":
        #labels转成数字类型
        labels = np.array([1 if label == 'positive' else 0 for label in labels])
    elif labels[0] == "1" or labels[0] == "0":
        labels = np.array([1 if label == '1' else 0 for label in labels])
    return labels