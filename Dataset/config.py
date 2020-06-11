
class Config(object):
    train_path = "data/train.csv"
    test_path = "data/test_noLabel.csv"
    seq_length = 200
    split_frac = 0.8
    batch_size = 50
    lr = 0.001
    epochs = 2
    print_every = 100
    model_path = None   # 预训练模型路径
    save_path = ''
    model_prefix = 'checkpoints/'  # 模型保存路径
    embed_path = 'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin'

opt = Config()
