from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 输入文件
glove_file = r'E:/Study/Codings/python_work/nlp_pro1/word2vec_model/glove.6B/glove.6B.300d.txt'
# 输出文件
tmp_file = r'E:/Study/Codings/python_work/nlp_pro1/word2vec_model/glove_vec/glove.6B.300d.txt'

# 命令行调用
# python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>

# 开始转换
glove2word2vec(glove_file, tmp_file)

# 加载转化后的文件
model = KeyedVectors.load_word2vec_format(tmp_file)