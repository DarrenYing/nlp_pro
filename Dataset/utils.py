import torch
from Dataset import PreprocessTools


class Utils(object):

    def __init__(self):
        self.tool = PreprocessTools()


    def preprocess(self, sentences, labels, istest, input_cols, embed, seq_len):
        texts, all_words = self.tool.preprocess_input(sentences, input_cols)
        texts, labels_after = self.tool.remove_outliers(texts, labels, istest=istest)
        tokenized_texts = self.tool.tokenize_all_texts(embed.embed_lookup, texts)
        features = self.tool.pad_features(tokenized_texts, seq_len)

        return features, labels_after

    def check_gpu(self):
        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')
        return train_on_gpu