# -*- coding:utf-8 -*-
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_curve, auc
import json
from gen_how_scores import how_scores, gen_scores


EMBEDDING_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/w2v_model.bin'
TRAIN_DATA_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/dataset.csv'
test_file = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/sim_qs_cut.json'

MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15
act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set
STAMP = './model/w2v_sgd_b16'
save = True
load_tokenizer = False
save_path = "./model"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "./model/embedding_matrix.npy"


def data_prepare(datafile):
    """从文件中读取数据"""
    data = pd.read_csv(datafile)
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['similarity'].values
    return sentences1, sentences2, labels


def tokenize(sentences=None):
    """获取所有文本中的词语"""
    if load_tokenizer:
        print('load tokenizer')
        tokenizer = pickle.load(open(os.path.join(save_path, tokenizer_name), 'rb'))
    else:
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(sentences)
        if save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pickle.dump(tokenizer, open(os.path.join(save_path, tokenizer_name), "wb"))
    return tokenizer


def sent2seq(tokenizer, sentences):
    """把句子转换成序列，如‘如何 来 防治 水稻 稻瘟病’----->[6, 383, 2, 1, 12]"""
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # 维度统一为MAX_SEQUENCE_LENGTH，不足的补0
    return sequences


def w2v(tokenizer, nb_words):
    """prepare embeddings"""
    print('Preparing embedding matrix')
    word2vec = Word2Vec.load(EMBEDDING_FILE)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if word in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv.word_vec(word)
        else:
            print(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    np.save(embedding_matrix_path, embedding_matrix)
    return embedding_matrix


def get_model(nb_words, embedding_matrix):
    """定义模型结构"""
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['acc'])
    model.summary()
    return model


def train_model(model, seq1, seq2, labels):
    """训练模型"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

    hist = model.fit([seq1, seq2], labels,
                     # validation_data=([test_seq1[:-100], test_seq2[:-100]], test_labels[:-100]),
                     validation_split=0.2,
                     epochs=100, batch_size=16, shuffle=True, callbacks=[model_checkpoint])

    model.load_weights(bst_model_path)
    bst_score = min(hist.history['loss'])
    bst_acc = max(hist.history['acc'])
    print(bst_acc, bst_score)
    print("Test score", min(hist.history["val_loss"]))
    print("Test acc", max(hist.history["val_acc"]))
    print(hist.history['val_loss'])
    print(hist.history['val_acc'])

    fig = plt.figure()
    x = range(1, len(hist.history['val_loss'])+1)
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(x, hist.history['val_loss'], 'r-', label='损失曲线 Loss curve')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x, hist.history['val_acc'], '--', label='精度曲线 Accuracy curve')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.set_ylabel('损失函数值 Loss function value')
    ax2.set_ylabel('精度 Accuracy')
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax1.legend(lns, labs, loc=7)
    ax1.set_xlabel("训练轮数 Epochs")
    plt.savefig('w2v_sgd_-1_b16.png')
    plt.show()


def test(model, test_sentences1, test_sentences2, test_seq1, test_seq2, test_labels):
    predicts = model.predict([test_seq1, test_seq2], batch_size=16, verbose=1)
    print(predicts)
    pres = np.array([1 if p[0]>0.5 else 0 for p in predicts])
    for i in range(len(test_labels)):
        print("t1: {}, t2: {}, score: {}, real_sim: {}".
              format(test_sentences1[i], test_sentences2[i], predicts[i], test_labels[i])
              )
    print(sum(pres==test_labels))

    # 画roc曲线
    fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(test_labels, predicts)
    fpr_gen, tpr_gen, thresholds_gen = roc_curve(test_labels, gen_scores)
    fpr_how, tpr_how, thresholds_how = roc_curve(test_labels, how_scores)

    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
    roc_auc_gen = auc(fpr_gen, tpr_gen)
    roc_auc_how = auc(fpr_how, tpr_how)

    plt.plot(fpr_lstm, tpr_lstm, '-', label='本文模型 ROC曲线 ROC of this model')
    plt.plot(fpr_gen, tpr_gen, '-.', label='w2v_cosine ROC曲线 ROC of w2v_cosine')
    plt.plot(fpr_how, tpr_how, ':', label='HowNet ROC曲线 ROC of HowNet')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('真正率 True Positive Rate')
    plt.xlabel('假正率 False Positive Rate')
    plt.legend()
    plt.savefig('roc_auc.png')
    plt.show()

    # 画各阈值下的准确率曲线
    thresholds = [i / 100 for i in range(10, 90)]
    p = []
    for thres in thresholds:
        preds = [1 if p[0] > thres else 0 for p in predicts]
        preds = np.array(preds)
        print(thres, sum(preds == test_labels))
        p.append(sum(preds == test_labels))
    print(p)

    plt.plot(thresholds, p)
    plt.show()


def evaluate(model, test_seq1, test_seq2, test_labels):
    score = model.evaluate([test_seq1, test_seq2], test_labels, batch_size=10)
    print(score)


def sentence_most_similarity():
    # 计算一个问题与其它问题之间的相似度，找出与其最相似的问题
    data_dict = json.load(open(test_file, encoding='utf-8'))
    num_success = 0

    model = load_model('./model/w2v_sgd_b16.h5')

    global load_tokenizer
    load_tokenizer = True
    tokenizer = tokenize()

    for q in list(data_dict.keys()):
        qs = list(data_dict.keys())
        qs.remove(q)
        tests1 = np.array([q] * len(qs))
        tests2 = np.array(qs)
        test_seq1 = sent2seq(tokenizer, tests1)
        test_seq2 = sent2seq(tokenizer, tests2)

        predicts = model.predict([test_seq1, test_seq2], batch_size=16, verbose=1)
        max_inx = np.argmax(predicts, axis=0)[0]
        sort_inx = np.argsort(predicts, axis=0)[::-1]
        print(sort_inx)
        print(max_inx)
        # for i in range(len(tests1)):
        for i in sort_inx[:10]:
            print("t1: {}, t2: {}, score: {}".
                  format(tests1[i], tests2[i], predicts[i][0]))

        print('\n' + tests1[max_inx] + ';' + tests2[max_inx] + ';', predicts[max_inx][0])

        most_similarity_sentence = tests2[max_inx]
        print('相似问题集：\n', data_dict[q])
        print('预测结果:\n', most_similarity_sentence)
        if most_similarity_sentence in data_dict[q]:
            print('预测正确')
            num_success += 1
        print('\n')
    print(num_success)


def main():
    print('\n从文件中读取数据..............................')
    sentences1, sentences2, labels = data_prepare(TRAIN_DATA_FILE)
    train_sentences1, train_sentences2, train_labels = sentences1[:-3207], sentences2[:-3207], labels[:-3207]
    test_sentences1, test_sentences2, test_labels = sentences1[-3207:], sentences2[-3207:], labels[-3207:]
    print('Found %s texts in train.csv' % len(sentences1))

    sentence_all = np.concatenate((sentences1, sentences2), axis=0)

    print('\n获取所有文本中的词语..........................')
    tokenizer = tokenize(sentence_all)
    # print(tokenizer.word_index)
    # print([tokenizer.word_index[word] for word in ['如何', '来', '防治', '水稻', '稻瘟病']])
    nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index)) + 1

    print('\n把句子转换成序列， 并进行长度补全...............')
    train_seq1 = sent2seq(tokenizer, train_sentences1)
    train_seq2 = sent2seq(tokenizer, train_sentences2)
    test_seq1 = sent2seq(tokenizer, test_sentences1)
    test_seq2 = sent2seq(tokenizer, test_sentences2)

    # print('\n计算每个词语的向量............................')
    # embedding_matrix = w2v(tokenizer, nb_words)
    # # embedding_matrix = np.ones((nb_words, EMBEDDING_DIM)) # bow模型
    #
    # print('\n设计模型结构..................................')
    # model = get_model(nb_words, embedding_matrix)
    #
    # print('\n训练模型.....................................')
    # train_model(model, train_seq1, train_seq2, train_labels)

    print('\n测试模型.....................................')
    model = load_model('./model/w2v_sgd_b16.h5')
    test(model, test_sentences1, test_sentences2, test_seq1, test_seq2, test_labels)
    # evaluate(model, test_seq1, test_seq2, test_labels)


if __name__ == '__main__':
    # main()
    sentence_most_similarity()
