import os
import json
import math
import pandas as pd
import numpy as np

DATA_DIR = '/media/jlan/E/Projects/nlp/crop_qa/data4'
TRAIN_DATA_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/dataset.csv'
TEST_FILE = os.path.join(DATA_DIR, '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/test.txt') # 测试文件


def data_prepare(datafile):
    with open(datafile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d
                                                          / self.avgdl)))
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        while True:
            most_sim_index = scores.index(max(scores))
            most_sim = self.docs[most_sim_index]
            if most_sim == doc:
                scores.remove(scores[most_sim_index])
            else:
                return most_sim


    def main(self):
        std_sim_dict = json.load(open(os.path.join(DATA_DIR, 'part/sim_qs_cut.json'), encoding='utf-8'))
        num_std_q = len(std_sim_dict) # 标准问题数量
        num_correct = 0  # 预测正确的数量

        for sentence in self.docs:
            print(sentence)
            most_sim = self.simall(sentence)
            print(most_sim)
            sims = std_sim_dict.get(sentence)
            print(sims)
            if sims:  # 有的标准问题没有与之对应的相似问题
                if most_sim in sims: # 从相似问题预测出每一个标准问题中反向查找相似问题，如果找到该相似问题，则预测成功
                    num_correct += 1
        print('num_correct:', num_correct)
        print('num_std_q:', num_std_q)
        print('accuracy: ', num_correct/num_std_q)


if __name__ == '__main__':
    sentences = data_prepare(TEST_FILE)
    bm = BM25(sentences)
    bm.main()