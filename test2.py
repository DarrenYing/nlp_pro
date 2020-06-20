
import matplotlib.pyplot as plt
import numpy as np

def embed_acc():
    x = [1,2,3,4,5]
    y_sc = [0.8852, 0.8696,	0.8796,	0.8808, 0.891]
    y_ba = [0.874, 0.8868, 0.8896, 0.8808, 0.8752]
    y_tr = [0.8744,	0.8812,	0.8828,	0.874, 0.8968]

    plt.plot(x, y_sc, label='SentimentCNN')
    plt.plot(x,y_ba, label='Bilstm_Attention')
    plt.plot(x,y_tr, label='TextRNN')

    plt.xlabel('Word Embedding')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def hidden_size_acc():
    x = [100, 200, 300]
    y_sc = [0.8696,	0.882, 0.8796]
    y_ba = [0.878, 0.8788, 0.8768]
    y_tr = [0.8828,	0.8904,	0.8904]

    plt.plot(x, y_sc, label='SentimentCNN')
    plt.plot(x, y_ba, label='Bilstm_Attention')
    plt.plot(x, y_tr, label='TextRNN')

    plt.xlabel('hidden size')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("hidden_size_acc.png")
    plt.show()

def dropout_acc():
    x = [0.3, 0.4, 0.5, 0.6]
    y_sc = [0.8796,	0.8856,	0.882, 0.8876]
    y_ba = [0.8764,	0.8732,	0.8768,	0.88]
    y_tr = [0.8848,	0.884, 0.8904, 0.8864]

    plt.plot(x, y_sc, label='SentimentCNN')
    plt.plot(x, y_ba, label='Bilstm_Attention')
    plt.plot(x, y_tr, label='TextRNN')

    plt.xlabel('dropout')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("dropout_acc.png")
    plt.show()


hidden_size_acc()
# dropout_acc()