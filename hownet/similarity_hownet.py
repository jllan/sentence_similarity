import numpy as np
import pandas as pd
from hownet import similar_sentence
import time
import matplotlib.pyplot as plt

TRAIN_DATA_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/dataset.csv'


def data_prepare(datafile):
    data = pd.read_csv(datafile)
    data = data.iloc[-3207:]
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['similarity'].values
    return sentences1, sentences2, labels


def main():
    predicts = []
    sentences1, sentences2, labels = data_prepare(TRAIN_DATA_FILE)
    for sent1, sent2 in zip(sentences1, sentences2):
        score = similar_sentence(sent1.split(), sent2.split())
        print('{}, {}, {}'.format(sent1, sent2, score))
        predicts.append(score)

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
    print(predicts)

    # thresholds = [i / 100 for i in range(10, 90)]
    # # thresholds = [0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
    # p = []
    # for thres in thresholds:
    #     preds = [1 if p > thres else 0 for p in predicts]
    #     preds = np.array(preds)
    #     print(thres, sum(preds == labels))
    #     p.append(sum(preds == labels))
    #
    # plt.plot(thresholds, p)
    # plt.title('各阈值下的准确率')
    # plt.ylabel('准确率 Accuracy')
    # plt.xlabel('阈值 Thresholds')
    # plt.show()
    # print(p)


if __name__ == '__main__':
    start = time.time()
    for _ in range(1):
        main()
    print('run time： ', time.time()-start)