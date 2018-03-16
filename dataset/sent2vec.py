import os
import subprocess
import jieba
import json
from subprocess import call


DATA_DIR = '/media/jlan/E/Projects/nlp/crop_qa/data4'
# MODEL_DIR = os.path.join('/home/jlan/Projects/agri/crop_qa_web/app/sent2vec/model')
GEN_MODEL_FILE = os.path.join(DATA_DIR, 'gen_rice_model.bin')    # gensim生成的模型

TRAIN_FILE = os.path.join(DATA_DIR, 'thu_train_file.txt')
# TRAIN_FILE = '/home/jlan/Projects/nlp/数据集/thu_train_file.txt'
# CORPORA = os.path.join(DATA_DIR, 'part/std_q_partial_cut.txt')
CORPORA = os.path.join(DATA_DIR, 'part/sim_qs_cut.txt')
CORPORA_DICT = os.path.join(DATA_DIR, 'qs.json')
# TEST_FILE = os.path.join(DATA_DIR, 'part/sim_q_partial_cut.txt') # 测试文件
TEST_FILE = os.path.join(DATA_DIR, 'part/test.txt') # 测试文件



class Sent2Vec:
    def __init__(self):
        self.fasttext = os.path.join(DATA_DIR, 'fasttext')
        self.model = os.path.join(DATA_DIR, 'rice_model.bin')    # sen2vec生成的模型

    def train(self, input_file):
        # train_command = '{} sent2vec -input {} -output {} -minCount 8 -dim 700 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns ' \
        #                 '-neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000'\
        #     .format(self.fasttext, input_file, self.model)

        train_command = '{} sent2vec -input {} -output {}'.format(self.fasttext, input_file, self.model)
        print(train_command)
        call(train_command, shell=True)


    def cut_words(self, sentence):
        with open(os.path.join(DATA_DIR, 'stop_words.txt'), 'r') as f:
            stop_words = [word.strip() for word in f.readlines()if word.strip()]
        word_list = jieba.cut(sentence)
        word_list = [word.strip() for word in word_list if word.strip() and word.strip() not in stop_words]
        # word_list = [word.strip() for word in word_list if word.strip()] # 不去停用词
        return word_list


    def get_nnSent(self, corpora, test_file, k):
        """
        执行"fasttext nnSent model.bin corpora.txt [k]，找到相似句子
        """
        test_command = '{} nnSent {} {} {} {}' \
            .format(self.fasttext, self.model, corpora, test_file, k)
        result = subprocess.check_output(test_command, shell=True)
        result = result.decode('utf8')
        with open(os.path.join(DATA_DIR, 'part/result.txt'), 'w') as f:
            f.write(result)
        # result = result.split('\n')[2:]
        # result = [i.split(',')[0].strip() for i in result if i.strip()]
        return result


    def nn_result_process(self, result):
        """
        通过nnSent初步配对相似问题和标准问题，后续再手动进一步处理
        :param result:
        :return:
        """
        result = result.strip('\nPre-computing sentence vectors... done.')
        result = result.split('\n\n')
        f = open(CORPORA_DICT, encoding='utf-8')
        q_dict = json.load(f)
        stan_sim = {}
        with open(os.path.join(DATA_DIR, 'stan_sim.txt'), 'w') as f:
            for res in result:
                qs = res.split('\n')
                print(qs)
                stan_q = qs[0].strip()
                stan_q_o = q_dict.get(stan_q, '')
                stan_sim[stan_q_o] = []
                f.write(stan_q_o+'\n')
                sim_qs = qs[2:] # qs[1]=qs[0]
                for sq in sim_qs:
                    sqq, sqc = sq.split(',')
                    sqq_o = q_dict.get(sqq.strip())
                    if not sqq_o:
                        print('sq:', sq)
                    stan_sim[stan_q_o].append([sqq_o, sqc])
                    f.write(sqq_o + '.....' + sqc + '\n')
                f.write('\n')
        with open(os.path.join(DATA_DIR, 'stan_sim.json'), 'w', encoding='utf8') as json_file:
            json_file.write(json.dumps(stan_sim, ensure_ascii=False, indent=2))


    def sim_qs_distinct(self):
        f = open(os.path.join(DATA_DIR, 'stan_sim.json'), encoding='utf-8')
        sim_qs_dict = json.load(f)
        sim_qs_dict2 = sim_qs_dict.copy()
        for q, simqs in sim_qs_dict.items():
            for sq in simqs[:5]:
                # del sim_qs_dict2[sq[0]]
                sim_qs_dict2.pop(sq[0], None)
        print(len(sim_qs_dict))
        print(len(sim_qs_dict2))
        with open(os.path.join(DATA_DIR, 'stan_sim_distinct.json'), 'w', encoding='utf8') as json_file:
            json_file.write(json.dumps(sim_qs_dict2, ensure_ascii=False, indent=2))
        with open(os.path.join(DATA_DIR, 'stan_sim_distinct.txt'), 'w') as f:
            for k,vs in sim_qs_dict2.items():
                f.write(k+'\n')
                for v in vs:
                    f.write(v[0]+'.......'+v[1]+'\n')
                f.write('\n\n')


    def search(self, text):
        """
        :param text: 从django后端传过来的
        :return: 与text最相似的问题
        """
        word_list = self.cut_words(text)
        with open(TEST_FILE, 'w') as f:
            f.write(' '.join(word_list))
        result = self.get_nnSent(CORPORA, TEST_FILE, k=3)
        f = open(CORPORA_DICT, encoding='utf-8')
        q_dict = json.load(f)
        result_id = [q_dict.get(i) for i in result]
        return result, result_id


    def compute_precision(self, result, k):
        std_sim_dict = json.load(open(os.path.join(DATA_DIR, 'part/std_sim_partial_dict_cut.json'), encoding='utf-8'))
        num_std_q = len(std_sim_dict) # 标准问题数量
        num_sim_q = sum([len(i) for i in std_sim_dict.values()]) # 相似问题数量，一个标准问题可能对应多个相似问题
        num_correct = 0  # 预测正确的数量

        # 从get_nnSent函数生成的结果中读取，每k+1行为一个单元，第一行是相似问题，后k行是找出的最接近的k个标准问题
        with open(result, 'r') as f:
            lines = [line.strip() for line in f.readlines()[1:] if line.strip()]
        results_each = [lines[i:i+k+1] for i in range(0, len(lines), k+1)]
        num_sim_q2 = len(results_each)
        print('results_each', results_each)

        for result_each in results_each:
            sim_q = result_each[0]  # 相似问题
            std_qs = [v.split(',')[0].strip() for v in result_each[1:]] # 预测出的标准问题
            # print('sim: ', sim_q)
            # print(std_qs)
            find = False
            # print('std_qs', std_qs)
            for j in range(k):
                sims = std_sim_dict.get(std_qs[j])
                if sims: # 有的标准问题没有与之对应的相似问题
                    if sim_q in sims: # 从相似问题预测出每一个标准问题中反向查找相似问题，如果找到该相似问题，则预测成功
                        num_correct += 1
                        find = True
                        # print('success predict std: ', std_qs)
                        break
            if not find:
                print('sim: ', sim_q)
                print('failure predict std: ', std_qs)
        print('num_correct:', num_correct)
        print('num_std_q:', num_std_q)
        print('num_sim_q:', num_sim_q)
        print('num_sim_q2:', num_sim_q2)
        print('accuracy: ', num_correct/num_sim_q)


    def compute_precision2(self, result, k):
        std_sim_dict = json.load(open(os.path.join(DATA_DIR, 'part/sim_qs_cut.json'), encoding='utf-8'))
        num_std_q = len(std_sim_dict) # 标准问题数量
        num_sim_q = sum([len(i) for i in std_sim_dict.values()]) # 相似问题数量，一个标准问题可能对应多个相似问题
        num_correct = 0  # 预测正确的数量

        # 从get_nnSent函数生成的结果中读取，每k+1行为一个单元，第一行是相似问题，后k行是找出的最接近的k个标准问题
        with open(result, 'r') as f:
            lines = [line.strip() for line in f.readlines()[1:] if line.strip()]
        results_each = [lines[i:i+k+1] for i in range(0, len(lines), k+1)]
        num_std_q2 = len(results_each)

        for result_each in results_each:
            std_q = result_each[0]  # 问题
            sim_qs = [v.split(',')[0].strip() for v in result_each[2:]] # 计算出的与之相似的问题
            find = False
            for j in range(k-1):
                sims = std_sim_dict.get(std_q)
                if sims: # 有的标准问题没有与之对应的相似问题
                    if sim_qs[j] in sims: # 从相似问题预测出每一个标准问题中反向查找相似问题，如果找到该相似问题，则预测成功
                        num_correct += 1
                        find = True
                        break
            if not find:
                print('std: ', std_q)
                print('predict sims: ', sim_qs)
                print('correct sims: ', sims)
                print('\n')
        print('num_correct:', num_correct)
        print('num_std_q:', num_std_q)
        print('num_sim_q:', num_sim_q)
        print('num_std_q2:', num_std_q2)
        print('accuracy: ', num_correct/num_std_q2)


if __name__ == '__main__':
    s2v = Sent2Vec()
    # s2v.train(TRAIN_FILE)
    k = 2

    result = s2v.get_nnSent(CORPORA, TEST_FILE, k)
    # s2v.nn_result_process(result)

    # sim_qs_distinct()

    s2v.compute_precision2(os.path.join(DATA_DIR, 'part/result.txt'), k)