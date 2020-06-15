class Config(object):
    mode = "a"
    model_type = "BiLSTM_atte"
    train_path = "data/train.csv"
    test_path = "data/test_noLabel.csv"
    load_path = "checkpoints/models/AttentionTextRNN_200.pkl"
    seq_length = 200
    split_frac = 0.9
    batch_size = 50
    lr = 0.001
    epochs = 200
    print_every = 100
    model_path = None  # 预训练模型路径
    save_path = ''
    model_prefix = 'checkpoints/'  # 模型保存路径
    embed_path = 'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin'


configs = Config()
