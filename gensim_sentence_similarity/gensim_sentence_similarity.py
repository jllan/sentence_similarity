import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import time
import matplotlib.pyplot as plt
import json

TEST_DATA_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/dataset.csv'
EMBEDDING_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/w2v_model.bin'
test_file = '/media/jlan/E/Projects/nlp/crop_qa/data4/part/sim_qs_cut.json'

model = Word2Vec.load(EMBEDDING_FILE)

def data_prepare(datafile):
    """从文件中读取数据"""
    data = pd.read_csv(datafile)
    data = data.iloc[-3207:]
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['similarity'].values
    return sentences1, sentences2, labels


def sentence_most_similarity():
    data_dict = json.load(open(test_file, encoding='utf-8'))
    num_success = 0

    print(len(list(data_dict.keys())))

    for q in list(data_dict.keys()):
        predicts = []
        qs = list(data_dict.keys())
        qs.remove(q)

        for sent in qs:
            try:
                score = model.n_similarity(q.split(), sent.split())
            except Exception:
                score = 0
            # print('{}, {}, {}'.format(q, sent, score))
            predicts.append(score)

        print(predicts)

        max_inx = np.argmax(predicts, axis=0)
        sort_inx = np.argsort(predicts, axis=0)[::-1]
        print(sort_inx)
        print(max_inx)
        # for i in sort_inx[:10]:
        #     print("t1: {}, t2: {}, score: {}".
        #           format(q, qs[i], predicts[i]))


        print('\n' + q + ';' + qs[max_inx] + ';', predicts[max_inx])

        most_similarity_sentence = qs[max_inx]
        print('相似问题集：\n', data_dict[q])
        print('预测结果:\n', most_similarity_sentence)
        if most_similarity_sentence in data_dict[q]:
            print('预测正确')
            num_success += 1
        print('\n')
    print(num_success)


def main():
    predicts = []
    sentences1, sentences2, labels = data_prepare(TEST_DATA_FILE)
    for sent1, sent2 in zip(sentences1, sentences2):
        try:
            score = model.n_similarity(sent1.split(), sent2.split())
        except Exception:
            score = 0
        print('{}, {}, {}'.format(sent1, sent2, score))
        predicts.append(score)

    print(labels)
    print(predicts)
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(labels, predicts)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    start = time.time()
    # main()
    sentence_most_similarity()
    print('run time： ', time.time() - start)