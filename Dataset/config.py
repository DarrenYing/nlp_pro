import warnings
from torch import t, nn, optim

loss_func_dict = {"bce": nn.BCELoss, "l1": nn.L1Loss, "mse": nn.MSELoss, "cel": nn.CrossEntropyLoss,
                  "bce_l": nn.BCEWithLogitsLoss, "s_l1": nn.SmoothL1Loss}


class Config(object):
    mode = "a"
    model_type = "blstma"
    train_path = "data/train.csv"
    test_path = "data/test_noLabel.csv"
    label_col = "label"
    input_col = "txt"

    seq_length = 200
    split_frac = 0.9
    batch_size = 50
    lr = 0.001
    epochs = 50
    print_every = 100
    dropout = 0.5

    load_path = "./checkpoints/models/BiLSTM_atte" + "_%d" % epochs + ".pkl"

    embed_path = 'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin'
    loss_func = "bce"
    optimizer = "adam"

    def _parse(self, kwargs):
        if "h" in kwargs.keys():
            self._helplist()
            exit(0)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
                exit(-1)
            setattr(self, k, v)

        print('[*]user config:')
        print("=====================================")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, ":",getattr(self, k))
        print("=====================================")

    def _helplist(self):
        print("""give --x to set attribute x
    optional params and their default:
    [*]
        --mode = "a"
        --model_type = "blstma"
        --train_path = "data/train.csv"
        --test_path = "data/test_noLabel.csv"
        --seq_length = 200
        --split_frac = 0.9
        --batch_size = 50
        --lr = 0.001
        --epochs = 50
        --print_every = 100
        --dropout = 0.5
        --load_path = "./checkpoints/models/" + model_type + "_%d" % epochs + ".pkl"
    
        --embed_path = 'word2vec_model/GoogleNews-vectors-negative300-SLIM.bin'
        --loss_func = "bce"
        --optimizer = "adam"
    
    some of them special mentioned:
    --mode : 
             t---train_only
             p---predict_only
             a---all of above
    --model_type:
             sc---SentimentCNN
             tr---TextRNN
             atr---AttentionTextRNN
             blstma---BiLSTM_atte
    --loss_func:
             l1---L1Loss
             bce---BCELoss
             mse---MSELoss
             cel---CrossEntropyLoss
             s_l1---SmoothL1Loss
             bce_l---BCEWithLogitsLoss
    --optimizer:
             sgd---SGD
             mot---SGD with momentnum
             rms---RMSprop
             adg---Adagrad
             adam---ADAM
         """
)


configs = Config()
