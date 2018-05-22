# 首先使用word2vec对句子中的词语向量化，然后加和求平均作为句子向量，最后计算句子相似度

import os
import jieba
import json
import numpy as np
from functools import reduce
from gensim.models import Word2Vec


DATA_DIR = os.path.join('/media/jlan/E/Projects/nlp/crop_qa/data4')
# MODEL_DIR = os.path.join('/home/jlan/Projects/agri/crop_qa_web/app/sent2vec/model')
GEN_MODEL_FILE = os.path.join(DATA_DIR, 'gen_thu_rice_model.bin')    # gensim生成的模型
AGRI_WORDS = os.path.join(DATA_DIR, 'agri_words.txt')  # 农业领域词典

TRAIN_FILE = os.path.join(DATA_DIR, 'train.txt')
# TRAIN_FILE = '/home/jlan/Projects/nlp/数据集/thu_train_file.txt'
CORPORA = os.path.join(DATA_DIR, 'part/test.txt')
CORPORA_DICT = os.path.join(DATA_DIR, 'qs.json')
TEST_FILE = os.path.join(DATA_DIR, 'part/test.txt') # 测试文件

# jieba.load_userdict(AGRI_WORDS)

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, encoding='utf8', errors='ignore'):
            yield line.split()


class GenS2V:
    def __init__(self):
        # self.model = Word2Vec.load(GEN_MODEL_FILE)
        pass

    def train(self):
        sentences = MySentences('/media/jlan/E/Projects/nlp/数据集/thu_rice.txt')  # a memory-friendly iterator
        model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
        model.save(GEN_MODEL_FILE)
        return model

    def cut_words(self, sentence):
        with open(os.path.join(DATA_DIR, 'stop_words2.txt'), 'r') as f:
            stop_words = [word.strip() for word in f.readlines() if word.strip()]
        word_list = jieba.cut(sentence)
        word_list = [word.strip() for word in word_list if word.strip() and word.strip() not in stop_words]
        # word_list = [word.strip() for word in word_list if word.strip()] # 不去停用词
        return word_list

    def s2v(self, word_list):
        """
        :param sentence:  输入一个句子
        :return:  该句子的句向量
        算法重点创新的地方
        """
        # sent_vec = reduce(lambda x,y: np.add(x,y), [self.model.wv[word] for word in word_list])/len(word_list)
        result_vec = np.zeros(100)
        sent_len = len(word_list)
        for word in word_list:
            try:
                word_vec = self.model.wv[word]
            except Exception:
                sent_len -= 1
            else:
                result_vec += word_vec/np.linalg.norm(word_vec)
        sent_vec = result_vec/sent_len
        return sent_vec

    def compute_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦值
        :param vec1:
        :param vec2:
        :return:
        """
        num = np.dot(vec1,vec2)  # 若为行向量则 A * B.T
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        cos = num / denom  # 余弦值
        # sim = 0.5 + 0.5 * cos  # 归一化
        # print(cos)
        # print(sim)
        return cos

    def std_qs_vec(self, crop):
        with open(crop, 'r') as f:
            lines = f.readlines()
        words_vec_dict = {}
        for line in lines:
            line = line.strip()
            vec = self.s2v(line.split())
            words_vec_dict[line] = vec
        # with open(os.path.join(DATA_DIR, 'part/gen_words_vec_dict.json'), 'w', encoding='utf8') as json_file:
        #     json_file.write(json.dumps(words_vec_dict, ensure_ascii=False, indent=2))
        return words_vec_dict

    def get_nn_sent(self, words_vec, words_vec_dict, k):
        """
        从words_vec_dict中找与words_vec最相似的前k个
        :param words_vec: 要找与其相似的句子的向量
        :param words_vec_dict: 存储了{句子: 句向量}
        :param k:  前k个最相似
        :return:  前k个最相似的句子和相似值
        """
        similarity_sents = {}
        for q, v in words_vec_dict.items():
            cos = self.compute_similarity(words_vec, v)
            similarity_sents[q] = cos
        sorted_results = sorted(similarity_sents.items(), key=lambda item: item[1], reverse=True) # 按cos值对字典排序
        # print(sorted_results[:k])
        return sorted_results[1:k+1]

    def compute_precision(self, k=1):
        nn_sents_result = json.load(open(os.path.join(DATA_DIR, 'part/gen_nn_sents.json'), encoding='utf-8'))
        std_sim_dict = json.load(open(os.path.join(DATA_DIR, 'part/sim_qs_cut.json'), encoding='utf-8'))
        num_std_q = len(nn_sents_result) # 相似问题数量，一个标准问题可能对应多个相似问题
        num_correct = 0  # 预测正确的数量

        for std_q, sim_qs in nn_sents_result.items():
            find = False
            print(std_q)
            print(sim_qs)
            # print('std_qs', std_qs)
            sims = std_sim_dict.get(std_q)
            for sim_q in sim_qs[:k]:
                if sim_q[0].strip() in sims: # 从相似问题预测出每一个标准问题中反向查找相似问题，如果找到该相似问题，则预测成功
                    num_correct += 1
                    find = True
                    # print('success predict std: ', std_qs)
                    break
            if not find:
                print('std: ', std_q)
                print('predict sims: ', sim_qs)
                print('correct sims: ', sims)
        print('num_correct:', num_correct)
        print('num_sim_q:', num_std_q)
        print('accuracy: ', num_correct/num_std_q)
        return num_correct/num_std_q

    def main(self):
        sents_vec_dict = self.std_qs_vec(CORPORA)
        # print(sents_vec_dict)
        # f = open(CORPORA_DICT, encoding='utf-8')
        # words_vec_dict = json.load(f)
        with open(TEST_FILE, 'r') as f1:
            lines = f1.readlines()
            nn_sents_result = {}
            for line in lines:
                line = line.strip()
                sent_vec = self.s2v(line.split())
                nn_sents = self.get_nn_sent(sent_vec, sents_vec_dict, 9)
                nn_sents_result[line] = nn_sents
        with open(os.path.join(DATA_DIR, 'part/gen_nn_sents.json'), 'w', encoding='utf8') as json_file:
            json_file.write(json.dumps(nn_sents_result, ensure_ascii=False, indent=2))
        self.compute_precision()



if __name__ == '__main__':
    gs2v = GenS2V()
    gs2v.train()
    # sent1 = '编写史记的人受到了什么处罚'
    # sent2 = '司马迁收到了什么刑罚'
    # words1 = gs2v.cut_words(sent1)
    # words2 = gs2v.cut_words(sent2)
    # vec1 = gs2v.s2v(words1)
    # vec2 = gs2v.s2v(words2)
    # gs2v.compute_similarity(vec1, vec2)
    # sents_vec_dict = gs2v.std_qs_vec(CORPORA)
    # gs2v.get_nn_sent(words1, sents_vec_dict, 3)

    # gs2v.main()
    # gs2v.compute_precision(9)

# sim:  稻鸭共作 鸭子 的 作用 是
# failure predict std:  [('水稻纹枯病 的 菌核 是 怎么 形成 的', 0.86038722664756473)]
# sim:  水稻 已经 发生 稻曲病 用 什么 药物 防治 最佳
# failure predict std:  [('水稻 稻曲病 在 水稻 什么 时期 防治 最好', 0.86975828631826579)]

