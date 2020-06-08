from gensim.models import word2vec
import logging

class Embed_Base(object):

    def __init__(self, embedding_dim=300, islog=False):
        self.embedding_dim = embedding_dim
        self.islog = islog

    # word2vec Text8 的训练
    def train_save_model(self, save_path):
        if self.islog:
            logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)',level=logging.INFO)
        # 加载预料
        data_path = r'E:/Study/Codings/python_work/nlp_pro1/word2vec_model/text8'
        sentences = word2vec.Text8Corpus(data_path)
        model = word2vec.Word2Vec(sentences, size=self.embedding_dim)
        model.save(save_path)

    # 加载模型
    def load_model(self, path):
        model = word2vec.Word2Vec.load(path)
        # simi = model.similar_by_vector('women', 'men')
        # print(simi)
        # print(model.most_similar('man'))
        # print(model['red'])
        return model

