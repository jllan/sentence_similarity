import re
import os
import csv
import jieba
import time
import json
from functools import partial
from pymongo import MongoClient
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import random
from functools import reduce
from itertools import combinations


DATA_DIR = os.path.join('/media/jlan/E/Projects/nlp/crop_qa/data4')
ANS_CUT = os.path.join(DATA_DIR, 'corp_cut.txt')  # 处理答案后生成的文件
QS_ORIGIN = os.path.join(DATA_DIR, 'qs_origin.txt')  # 原始问题
QS_CUT = os.path.join(DATA_DIR, 'qs_cut.txt')  # 处理后的问题
QA_DICT = os.path.join(DATA_DIR, 'qa.json')    # json格式保存未经处理的问答对,格式如下
"""
{'q1':  # 问题
    {
        'id': '', # 问题id
        'content': '',  # 问题答案
        'sqs': [            # 相似问题集
            {
                'sq': '',   # 相似问题
                'sid': '',  # 相似问题id
                'scon': ''  # 相似问题答案
            },
        ]
    },
}
"""
QS_DICT = os.path.join(DATA_DIR, 'qs.json')  # {分词后问题： 原始问题}
AGRI_WORDS = os.path.join(DATA_DIR, 'agri_words.txt')  # 农业领域词典

with open(os.path.join(DATA_DIR, 'stop_words.txt'), 'r') as f:
    stop_words = [word.strip() for word in f.readlines() if word.strip()]

# jieba.load_userdict(AGRI_WORDS)

def union_agri_words(dir_name='/home/jlan/Projects/nlp/数据集/农业词汇', target_file='/home/jlan/Projects/nlp/数据集/农业词汇/agri_words_gather.txt'):
    """把dir_name目录下所有的农业领域词汇合并到一起"""
    files = os.listdir(dir_name)
    words = set()
    for f in files:
        f = os.path.join(dir_name, f)
        print(f)
        with open(f, 'r') as sf:
            lines = [line.split()[-1].strip()+'\n' for line in sf.readlines()]
            words |= set(lines)
    print(len(words))
    with open(target_file, 'a+') as tf:
        tf.writelines(words)


def cut_sentences(text):
    """分句"""
    sentence_list = re.split('[。！!？?\n]', text)
    sentence_list = [i.strip() for i in sentence_list if i.strip()]
    return sentence_list


def cut_words(sentence, stop_words):
    """分词"""
    print(sentence)
    sentence = re.sub('[\W+\s+\d+a-zA-Z]', '', sentence)
    word_list = jieba.cut(sentence, HMM=False)
    # word_list = [word.strip() for word in word_list if word.strip() and word.strip() not in stop_words]
    word_list = [word.strip() for word in word_list if word.strip()] # 不去停用词
    print(word_list)
    return word_list


def dump_text_to_disk(words, target_file):
    """把预处理后的文本写入文件，每次一句"""
    with open(target_file, 'a+') as f:
        f.write(words + ' \n')


def process_ans(text, stop_words, ans_file):
    # 对每一行进行处理，一行可能有多句
    print(text)
    text = re.sub('<.*?>', '', text).strip()
    if text:
        sentences = cut_sentences(text)
        for sentence in sentences:
            words = cut_words(sentence, stop_words)
            if words:
                dump_text_to_disk(' '.join(words), ans_file)


def process_qs(text, stop_words, qs_origin, qs_cut):
    # 对每一行进行处理，一行可能有多句
    text = re.sub('<.*?>', '', text).strip()
    if text.strip():
        dump_text_to_disk(text.strip(), qs_origin)
        words = cut_words(text, stop_words)
        dump_text_to_disk(words, qs_cut)


def process_qs_id(qs, stop_words, qs_file): # 以{分词后问问题： 问题id}存储
    # 对每一行进行处理，一行可能有多句
    qs_dict = {}
    for k,v in qs.items():
        q = re.sub('<.*?>', '', k).strip()
        if q:
            words = cut_words(q, stop_words)
            qs_dict[' '.join(words)] = v
            # dump_text_to_disk(' '.join(words), qs_file)
    print(qs_dict)
    with open(os.path.join(DATA_DIR, 'aj_rice_qs_dict.json'), 'w', encoding='utf8') as json_file:
        json_file.write(json.dumps(qs_dict, ensure_ascii=False, indent=2))
    for q in qs_dict.keys():
        dump_text_to_disk(q, qs_file)


def process_qs2(qs, stop_words, qs_cut_file, qs_origin_file, qs_dict_file): # 以{分词后问问题： 原始问题}存储
    # 对每一行进行处理，一行可能有多句
    qs_dict = {}
    for q in qs:
        qq = re.sub('<.*?>', '', q).strip()
        if qq:
            words = cut_words(qq, stop_words)
            if len(words)>2:
                dump_text_to_disk(qq, qs_origin_file) # 原始问题
                qs_dict[' '.join(words)] = q          # {q_cut: 原始问题}
    for q in qs_dict.keys():
        dump_text_to_disk(q, qs_cut_file)
    with open(qs_dict_file, 'w', encoding='utf8') as json_file:
        json_file.write(json.dumps(qs_dict, ensure_ascii=False, indent=2))


def get_data():
    """从mongodb获取数据"""
    client = MongoClient('localhost', 27017)
    db = client['agri_resource']
    collection = db['agri_corp']
    data = collection.find()
    # res = {d['title']: d['content'] for d in data}
    # res = {}
    res = []
    for d in data:
        res.append((d['title'], ''.join(d['content']), str(d['_id'])))
        # res.get(d['title'], []).append(re.sub('<.*?>', '', d['content']).strip())
    return res


def stan_sim_process(keyword):
    """
    从stan_sim_distinct.txt中取出具有相同关键字的问题
    :param keyword:
    :return:
    """
    with open(os.path.join(DATA_DIR, keyword+'.txt'), 'a+') as f1:
        with open(os.path.join(DATA_DIR, 'stan_sim_distinct.txt'), 'r') as f2:
            lines = f2.readlines()
            sim_lines = set()
            fangzhi = set()
            yufang = set()
            zhengzhuang = set()
            yuanyin = set()
            yao = set()
            shibie = set()
            guilv = set()
            other = set()
            for line in lines:
                line = line.split('.......')[0]
                if keyword in line:
                    # sim_lines.add(line+'\n')
                    if '识别' in line:
                        fangzhi.add(line+'\n')
                    if '规律' in line:
                        guilv.add(line + '\n')
                    elif '预防' in line:
                        yufang.add(line+'\n')
                    elif '症状' in line:
                        zhengzhuang.add(line+'\n')
                    elif '原因' in line:
                        yuanyin.add(line+'\n')
                    elif '药' in line:
                        yao.add(line+'\n')
                    elif '防治' in line:
                        shibie.add(line+'\n')
                    else:
                        other.add(line+'\n')
            f1.writelines(fangzhi)
            f1.write('\n')
            f1.writelines(yufang)
            f1.write('\n')
            f1.writelines(zhengzhuang)
            f1.write('\n')
            f1.writelines(yuanyin)
            f1.write('\n')
            f1.writelines(yao)
            f1.write('\n')
            f1.writelines(shibie)
            f1.write('\n')
            f1.writelines(other)


def from_stan_sim_distinct_partial_to_dict():
    """
    把stan_sim_distinct_partial.txt中的问题变成{std_q: simqs}的形式
    :return:
    """
    std_sim_dict = {}
    with open(os.path.join(DATA_DIR, 'part/stan_sim_distinct_partial.txt'), 'r') as f1:
        data = f1.read()
    qss = data.split('\n\n') # 把相似问题分开
    for qs in qss:
        qs = qs.split('\n')  # 对于每一堆相似问题,分割成一个个问题
        std_q = qs[0].split('....')[0]  # 第一个问题作为标准问题
        std_q = ' '.join(cut_words(std_q, stop_words))
        if len(qs) > 1:  # 如果相似问题集的数量>1
            sim_qs = [q.split('....')[0] for q in qs[1:]]
            sim_qs = [' '.join(cut_words(q, stop_words)) for q in sim_qs]
            std_sim_dict[std_q] = sim_qs
        # else:
        #     std_sim_dict[std_q] = []
    print(len(std_sim_dict))
    for q in std_sim_dict.keys():
        dump_text_to_disk(q, target_file=os.path.join(DATA_DIR, 'part/std_q_partial_cut.txt'))
    for qs in std_sim_dict.values():
        for q in qs:
            dump_text_to_disk(q, target_file=os.path.join(DATA_DIR, 'part/sim_q_partial_cut.txt'))
    with open(os.path.join(DATA_DIR, 'part/std_sim_partial_dict_cut.json'), 'w', encoding='utf8') as json_file:
        json_file.write(json.dumps(std_sim_dict, ensure_ascii=False, indent=2))


def from_stan_sim_distinct_partial_to_list():
    """
    把stan_sim_distinct_partial.txt中的问题变成[simq1,simq2...]的形式
    :return:
    """
    results = {}
    with open(os.path.join(DATA_DIR, 'part/stan_sim_distinct_partial.txt'), 'r') as f1:
        data = f1.read()
    qss = data.split('\n\n') # 把相似问题分开
    for qs in qss:
        qs = qs.split('\n')  # 对于每一堆相似问题,分割成一个个问题
        if len(qs)>1:
            qs = [' '.join(cut_words(q.split('...')[0], stop_words)) for q in qs]
            for inx, q in enumerate(qs):
                results[q] = qs[0:inx]+qs[inx+1:]
                dump_text_to_disk(q, target_file=os.path.join(DATA_DIR, 'part/sim_qs_cut.txt'))
            # dump_text_to_disk('\n', target_file=os.path.join(DATA_DIR, 'part/sim_qs.txt'))
    with open(os.path.join(DATA_DIR, 'part/sim_qs_cut.json'), 'w', encoding='utf8') as json_file:
        json_file.write(json.dumps(results, ensure_ascii=False, indent=2))


def data_structure():
    """
    从{std_q: [sim_q, sim_q2...]}结构的问题字典中构造训练数据，
    q:sim_q; # 正向训练样本，x为一条问题，y为与之相似的一个问题
    q:not_sim_q # 负向训练样本，x为一条问题，y为与之不相似的一个问题
    """
    data_dict = json.load(open(os.path.join(DATA_DIR, 'part/sim_qs_cut.json'), encoding='utf-8'))
    data_dict_copy = data_dict.copy()
    num_std_q = len(data_dict)  # 标准问题数量
    print(num_std_q)
    dataset_file = os.path.join(DATA_DIR, 'part/dataset/dataset.txt')
    dataset_file = os.path.join(DATA_DIR, 'part/dataset/dataset.csv')
    qs_positive_file = os.path.join(DATA_DIR,'part/dataset/qs_positive.txt') # 正向训练样本
    qs_negative_n_file = os.path.join(DATA_DIR,'part/dataset/qs_negative_n.txt') # 最近邻负向训练样本
    qs_negative_ran_file = os.path.join(DATA_DIR,'part/dataset/qs_negative_ran.txt') # 随机负向训练样本
    qs_origin = set()

    """把data_dict中的字典合并起来"""
    for k, v in data_dict.items():
        q = [i.strip() for i in v]
        q.append(k.strip())
        q.sort()
        qs_origin.add(tuple(q))
    print(len(qs_origin))

    negative_rans = []

    # with open(qs_positive_file, 'a+') as f1, open(qs_negative_n_file, 'a+') as f2, open(qs_negative_ran_file, 'a+') as f3:
    # with open(dataset_file, 'a+') as f:
    with open(dataset_file, "a+") as csvfile:
        writer = csv.writer(csvfile)
        for q in qs_origin:
            print(q)

            # 从一堆共有n个相似问题中取出两条作为x和y，共有n*(n-1)/2种选法
            for i in list(combinations(q, 2)):
                # f.write(i[0]+',' + i[1] + ','+ '1' + '\n')
                writer.writerow([i[0], i[1], 1])

            """把除了q项的剩余项合并"""
            qs_c = qs_origin.copy()
            qs_c.remove(q)
            qs_c = list(reduce(lambda x, y: x + y, qs_c))
            for i in q:
                print(i)
                js = random.sample(qs_c, 2) # 然后从中随机选择n=2个作为随机负样本
                print(js)
                for j in js:
                    # f.write(i + ',' + j + ','+ '0' + '\n')
                    writer.writerow([i, j, 0])
                    qs_c.remove(j)

                # 从除了q项和上一步随机取出的一条后的剩余项中取出一条最接近的，相同词越多可视为越接近
                i_ns = sorted(qs_c, key=lambda x: len(set(x)&set(i)), reverse=True)[:4]
                print(i_ns)
                for i_n in i_ns:
                    if sorted([i, i_n]) in negative_rans:
                        continue
                    else:
                        negative_rans.append(sorted([i, i_n]))
                        # f.write(i + ',' + i_n + ','+ '0' + '\n')
                        writer.writerow([i, i_n, 0])
                print('\n')



def main():
    start = time.time()

    with open(os.path.join(DATA_DIR, 'stop_words.txt'), 'r') as f:
        stop_words = [word.strip() for word in f.readlines() if word.strip()]

    data = get_data()
    qs, ans, qa = set(), set(), {}
    for d in data:
        qs.add(d[0])
        ans.add(d[1])
    print(len(qs))
    print(len(ans))
    # with open(QA_DICT, 'w', encoding='utf8') as json_file:
    #     json_file.write(json.dumps(qa, ensure_ascii=False, indent=2))
    # process_qs2(qs, stop_words, qs_cut_file=QS_CUT, qs_origin_file=QS_ORIGIN, qs_dict_file=QS_DICT)



    # with Pool() as pool:
    #     pool.map(partial(process_qs, stop_words=stop_words, qs_origin=QS_ORIGIN), qs)

    with ProcessPoolExecutor() as executor:
        executor.map(partial(process_ans, stop_words=stop_words, ans_file=ANS_CUT), ans)

    print('run time: ', time.time() - start)


if __name__ == '__main__':
    # main()
    # union_agri_words()
    # stan_sim_process('颖枯')
    # from_stan_sim_distinct_partial_to_dict()
    # from_stan_sim_distinct_partial_to_list()
    data_structure()


