
import nltk
import re
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

dataset = pd.read_csv("../train.csv",encoding = 'utf8')



maxlen = 30
dim = 100
random_state =100

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    return text

dataset['question1_n'] = dataset.question1.apply(lambda x :text_to_wordlist(x))
dataset['question2_n'] = dataset.question2.apply(lambda x :text_to_wordlist(x))

tokenizer = Tokenizer()


tokenizer.fit_on_texts(dataset.question1_n.tolist() + dataset.question2_n.tolist())

dataset['question1_seq']= tokenizer.texts_to_sequences(dataset.question1_n)
dataset['question2_seq']= tokenizer.texts_to_sequences(dataset.question2_n)



from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state= random_state)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state= random_state)

from gensim.models import KeyedVectors

num_word = len(tokenizer.word_index)
print(num_word)



from gensim.models import Word2Vec
EMBEDDING_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/w2v_model.bin'
def w2v(tokenizer, nb_words):
    """
    prepare embeddings
    :param tokenizer:
    :param nb_words:
    :return:
    """
    print('Preparing embedding matrix')
    word2vec = Word2Vec.load(EMBEDDING_FILE)
    embedding_matrix = np.zeros((nb_words, dim))
    embedding_matrix = embedding_matrix.astype(np.float32)
    print(embedding_matrix.dtype)
    for word, i in tokenizer.word_index.items():
        if word in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv.word_vec(word)
        else:
            print(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix

# glove_dict = loadGloveModel("../../glove.6B/glove.6B.300d.txt")


embedding_matrix = w2v(tokenizer, num_word+1)


import tensorflow as tf
import time
import tensorflow.contrib.layers as layers

min_temp_result = None

def create_filter_blockA_weight(n_grams, w_dim, num_kernel):
    weight = tf.Variable(tf.random_normal((n_grams, w_dim, num_kernel),stddev=0.1),name='blockA_W')
    # weight = tf.Variable(tf.ones((n_grams, w_dim, num_kernel)))
    bias = tf.Variable(tf.zeros(num_kernel),name='blockA_b')
    return weight , bias

def create_filter_blockB_weight(n_grams, w_dim, num_kernel):
    weight = tf.Variable(tf.random_normal((n_grams, 1, 1, num_kernel), stddev=0.1),name='blockB_W')
    # weight = tf.Variable(tf.ones((n_grams, 1, 1, num_kernel)))
    bias = tf.Variable(tf.zeros(num_kernel),name='blockB_b')
    return weight,bias


def create_horizontal_conv(input,sequence_length, kernel_weight, type, min_mask):
    '''
    :param input:  N x L x w_dim
    :param kernel_weight: [ window_size, 1, 1, number_kernel]
    :param type:
    :return: N x w_dim x num_kernel
    '''
    with tf.name_scope("horizontal_conv"):
        i0 = tf.constant(0)
        num_kernel = int(kernel_weight[0].get_shape()[-1])
        # result = tf.zeros([0,w_dim, num_kernel])

        input = tf.expand_dims(input, 3)
        output = tf.nn.conv2d(input, kernel_weight[0], [1, 1, 1, 1], 'SAME') + kernel_weight[1]  # N, height, width, out_kernel
        output = tf.nn.relu(output)
        if type == 'min':
            output = min_pool_operation2d(output,min_mask )
        elif type == 'mean':
            output = mean_pool_operation2d(output,sequence_length)
        elif type == 'max':
            output = tf.reduce_max(output, axis=1)  # N, out_kernel
        else:
            raise Exception("no such type")

        return output

        # result = tf.while_loop(_cond, _run_conv, loop_vars=[i0, result],shape_invariants=[i0.get_shape(), tf.TensorShape([None, w_dim, num_kernel])])
        # return result[1]



def create_vertical_conv(input, sequence_length,  kernel_weight, type, min_mask):
    '''
    :param input:
    :param sequence_length:
    :param kernel_weight:
    :param type:
    :return:  N, num_kernel
    '''
    # num_kernel = 3
    # filter = tf.Variable(tf.ones((2, 4, num_kernel)))
    with tf.name_scope("vertical_conv"):
        num_kernel = int(kernel_weight[0].get_shape()[2])
        # print(input.dtype, kernel_weight[0].dtype)
        output = tf.nn.conv1d(input, kernel_weight[0], 1,  'SAME') + kernel_weight[1]  # None, out_width  , out_kernel
        output = tf.nn.relu(output)

        if type == 'min':
            output = min_pool_operation(output,min_mask )
        elif type == 'mean':
            output = mean_pool_operation(output,sequence_length)
        elif type == 'max':
            output = tf.reduce_max(output, axis=1)  # N, out_kernel
        else:
            raise Exception("no such type")
        return output



def create_direct_pool(input,num_kernel, type):
    '''
    :param input:
    :param num_kernel:
    :param type:
    :return: N, num_kernel
    '''
    if type == 'min':
        output = tf.tile(tf.expand_dims(tf.reduce_min(tf.reduce_min(input, axis=1), axis=1),axis=1),[1,num_kernel])
    elif type == 'mean':
        output = tf.tile(tf.expand_dims(tf.reduce_mean(tf.reduce_mean(input, axis=1), axis=1),axis=1),[1,num_kernel])
    elif type == 'max':
        output = tf.tile(tf.expand_dims(tf.reduce_max(tf.reduce_max(input, axis=1), axis=1),axis=1),[1,num_kernel])
    else:
        raise Exception("no such type")
    return output


def l2distance(input1,input2):
    l2diff =tf.reduce_sum(tf.square(tf.subtract(input1, input2)),
                                   axis=1)
    l2diff = tf.clip_by_value(l2diff,0.1,1e7)
    l2diff = tf.sqrt(l2diff)
    return l2diff


def l1distance(input1,input2):
    l1diff = tf.square(tf.subtract(input1, input2))
    l1diff = tf.sqrt(tf.clip_by_value(l1diff,0.1,1e7))
    l1diff = tf.reduce_sum(l1diff, axis=1)
    return l1diff

def cosine_similarity(input1,input2):
    n_input1 = tf.nn.l2_normalize(input1, dim=1,epsilon=1e-7)
    n_input2 = tf.nn.l2_normalize(input2, dim=1,epsilon=1e-7)
    cosine_sim = tf.reduce_sum(tf.multiply(n_input1, n_input2), axis=1)
    return cosine_sim


def pairwise_distance1(input1, input2):
    '''
    :param input1: N, num_kernel, 1
    :param input2: N, num_kernel, 1
    :return:
    '''
    with tf.name_scope("pairwise_distance1"):
        # return tf.stack([cosine_similarity(input1,input2),l2distance(input1,input2)],axis=1)
        return tf.concat([cosine_similarity(input1,input2),l2distance(input1,input2)],axis=1)


def pairwise_distance2(input1, input2):
    '''
    :param input1: N, num_kernel, 1
    :param input2: N, num_kernel, 1
    :return:
    '''
    with tf.name_scope("pairwise_distance2"):
        # return tf.stack([cosine_similarity(input1,input2),l2distance(input1,input2)],axis=1)
        return tf.concat([cosine_similarity(input1,input2),l2distance(input1,input2)],axis=1)


def get_init_min_mask_value(input_sequence):
    value = np.zeros(shape=(input_sequence.shape[0],maxlen))
    for i, l in enumerate(input_sequence):
        value[i, l:] = 1e7
    return value

def min_pool_operation(tf_var, min_mask):
    '''
    :param tf_var:
    :param min_mark:
    :return:
    '''
    global min_temp_result
    min_mask = tf.expand_dims(min_mask,axis=2)
    # min_mask = tf.reshape(min_mask, (None,min_mask_mask.shape[0], tf_var.shape[2]))
    temp = tf.add(tf_var, tf.cast(min_mask,tf.float32))
    min_temp_result = temp
    return tf.reduce_min(temp, axis=1)


def mean_pool_operation(tf_var, input_sequence):
    '''
    :param tf_var:
    :param min_mark:
    :return:
    '''
    input_sequence = tf.reshape(input_sequence,[-1,1])
    temp = tf.divide(tf.reduce_sum(tf_var,axis=1), tf.add(tf.cast(input_sequence,tf.float32),1e-7))
    return temp


def min_pool_operation2d(tf_var, min_mask):
    '''
    :param tf_var:
    :param min_mark:
    :return:
    '''
    min_mask = tf.expand_dims(tf.expand_dims(min_mask,2),3)
    min_mask = tf.tile(min_mask,[1,1,int(tf_var.shape[2]),int(tf_var.shape[3])])
    # min_mask = tf.reshape(min_mask, (None,min_mask_mask.shape[0], tf_var.shape[2]))
    temp = tf.add(tf_var, tf.cast(min_mask,tf.float32))
    return tf.reduce_min(temp, axis=1)


def mean_pool_operation2d(tf_var, input_sequence):
    '''
    :param tf_var:
    :param min_mark:
    :return:
    '''
    input_sequence = tf.reshape(input_sequence,[-1,1,1])
    input_sequence = tf.tile(input_sequence,[1,int(tf_var.shape[2]),int(tf_var.shape[3])])
    temp = tf.divide(tf.reduce_sum(tf_var,axis=1), tf.add(tf.cast(input_sequence,tf.float32), 1e-7))
    return temp


class MPCNN:

    def __init__(self, maxlen, dim, embedding_weight):
        self.input = tf.placeholder(tf.int32,(None,maxlen),name='input1')
        self.input2 = tf.placeholder(tf.int32,(None,maxlen),name='input2')
#         self.input = tf.placeholder(tf.float32,(None,maxlen, dim),name='input1')
#         self.input2 = tf.placeholder(tf.float32,(None,maxlen, dim),name='input2')
        self.seq_length1 = tf.placeholder(tf.int32,(None),name='seq_len_1')
        self.seq_length2 = tf.placeholder(tf.int32,(None),name='seq_len_2')
        self.min_mask1 = tf.placeholder(tf.int32, (None, maxlen),name='min_mask1')
        self.min_mask2 = tf.placeholder(tf.int32, (None, maxlen),name='min_mask2')
        self.num_kernel_a = 32
        self.num_kernel_b = 32
        self.embedding_weight = tf.Variable(embedding_weight, name="E_W")
        self.y = tf.placeholder(tf.int32, shape=(None,2),name='ans')

        input = tf.nn.embedding_lookup(self.embedding_weight, self.input)
        print('hhh', input.dtype)
        input2 = tf.nn.embedding_lookup(self.embedding_weight, self.input2)
#         input = self.input
#         input2 = self.input2
        
        num_kernel_a = self.num_kernel_a
        num_kernel_b = self.num_kernel_b
        seq_length1 = self.seq_length1
        seq_length2 = self.seq_length2
        min_mask1 = self.min_mask1
        min_mask2 = self.min_mask2
        y = self.y
        
        w_dim = dim
        
        n_grams_types = list(range(1,4)) + [-1]
        blockA_type= ['max','mean']
        blockA_weights = {}
        self.blockA_weights = blockA_weights
        regularizers = []
        for n_g in n_grams_types:
            for type in blockA_type:
                if n_g  == - 1:
                    continue
                t_w = create_filter_blockA_weight(n_g,w_dim, num_kernel_a)
                regularizers.append(tf.nn.l2_loss(t_w[0]))
                blockA_weights[(n_g,type)] = t_w

        
        blockA_convs = [{},{}]
        self.blockA_convs = blockA_convs
        for n_g in n_grams_types :
            for type in blockA_type:
                if n_g == -1 :
                    blockA_convs[0][(n_g,type)] = create_direct_pool(input,num_kernel_a,type)
                    blockA_convs[1][(n_g,type)] = create_direct_pool(input2,num_kernel_a,type)
                else:
                    t_w = blockA_weights[(n_g,type)]
                    print(input.dtype)
                    blockA_convs[0][(n_g,type)] = create_vertical_conv(input,seq_length1,t_w,type, min_mask1)
                    blockA_convs[1][(n_g,type)] = create_vertical_conv(input2,seq_length2,t_w,type, min_mask2)


        #---------- block B ------------------
        blockB_type= ['max','mean']
        blockB_weights = {}
        self.blockA_weights = blockA_weights
        for n_g in n_grams_types:
            for type in blockB_type:
                if n_g  == - 1:
                    continue
                t_w = create_filter_blockB_weight(n_g,w_dim, num_kernel_b)
                regularizers.append(tf.nn.l2_loss(t_w[0]))
                blockB_weights[(n_g,type)] = t_w


        blockB_convs = [{},{}]
        self.blockB_convs = blockB_convs
        for n_g in n_grams_types :
            for type in blockB_type:
                if n_g == -1 :
                    continue
                else:
                    t_w = blockB_weights[(n_g,type)]
                    blockB_convs[0][(n_g,type)] = create_horizontal_conv(input,seq_length1,t_w,type,min_mask1)
                    blockB_convs[1][(n_g,type)] = create_horizontal_conv(input2,seq_length2,t_w,type,min_mask2)


                    
        outputs = []
        #------------vertical-----comparison -------------

        with tf.name_scope("vertical_comparison"):
            vertical_gp1 = []
            vertical_gp2 = []
            for type in blockA_type:
                for n_g1 in n_grams_types:
                    o1 = blockA_convs[0][(n_g1, type)]
                    for n_g2 in n_grams_types:
                        o2 = blockA_convs[1][(n_g2, type)]
                        print(n_g1,n_g2, type)
                        vertical_gp1.append(o1)
                        vertical_gp2.append(o2)

            vertical_gp1 = tf.stack(vertical_gp1,axis=2)
            vertical_gp2 = tf.stack(vertical_gp2,axis=2)
            self.temp_gp1 = vertical_gp1
            self.temp_gp2 = vertical_gp2
            o = pairwise_distance1(vertical_gp1, vertical_gp2)
            outputs.append(o)


            vertical_gp1 = []
            vertical_gp2 = []
            for n_g in n_grams_types:
                if n_g == -1:
                    continue
                for type in blockB_type:
                    vertical_gp1.append(blockB_convs[0][(n_g, type)])
                    vertical_gp2.append(blockB_convs[1][(n_g, type)])
          
            vertical_gp1 = tf.concat(vertical_gp1,axis=2)
            vertical_gp2 = tf.concat(vertical_gp2,axis=2)
            self.temp_gp1 = vertical_gp1
            self.temp_gp2 = vertical_gp2
            o = pairwise_distance1(vertical_gp1, vertical_gp2)
            outputs.append(o)
 

        #-----------horizontal----comparison -------------------
        with tf.name_scope("horizontal_comparison"):
            gp1 =[]
            gp2 =[]
            for type in blockA_type:
                # r1 = []
                # r2 = []
                for n_g1 in n_grams_types:
                    gp1.append(blockA_convs[0][(n_g1, type)]) # N, num_kernel
                    gp2.append(blockA_convs[1][(n_g1, type)]) # N, num_kernel
         
            gp1 = tf.reshape(tf.concat(gp1,axis=1),(-1,len(n_grams_types),num_kernel_a * len(blockA_type)))
            gp2 = tf.reshape(tf.concat(gp2,axis=1),(-1, len(n_grams_types), num_kernel_a  * len(blockA_type)))
            o = pairwise_distance2(gp1, gp2)
            outputs.append(o) 

        self.outputs = outputs
        concat_output = tf.concat(outputs,axis=1)
        self.concat_output = concat_output

#         fc_ol = layers.fully_connected(concat_output, 64)
        
       
        
        def create_fc_layer(num_node, prev_input):
            weight = tf.Variable(tf.truncated_normal([int(prev_input.shape[1]), num_node],stddev=0.1),name='fc_W')
            regularizers.append(tf.nn.l2_loss(weight))
            fc_biases_1 = tf.Variable(tf.zeros([num_node]),name='fc_b')
            output = tf.nn.elu(tf.matmul(prev_input,weight) + fc_biases_1)
            return output
       
        prob = tf.placeholder_with_default(1.0, shape=())
        self.prob = prob
        
        concat_output = tf.nn.dropout(concat_output, prob)
        fc_output = create_fc_layer(64, concat_output)
        concat_output = tf.nn.dropout(fc_output, prob)
        output = create_fc_layer(2, fc_output)
        
        
        self.output = output

        self.pred = tf.nn.softmax(output,dim=1)

        self.total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
        self.loss = tf.reduce_mean(self.total_loss)
        total_l2_loss = tf.zeros(1)
        for r in regularizers:
            total_l2_loss += r
        self.loss += 1e-7 * total_l2_loss

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.output, 1)), tf.float32))
        self.acc = acc
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        



model = MPCNN(maxlen,dim,embedding_matrix)


def get_feature_X(df, maxlen):
    x1 = []
    x2 = []
    l1 = []
    l2 = []
    for q1, q2 in zip(df.question1_seq.values, df.question2_seq.values):
        
        x1.append(q1)
        x2.append(q2)
        
#         x1.append([embedding_matrix[t] for t in q1])
#         x2.append([embedding_matrix[t] for t in q2])
        l1.append(len(q1))
        l2.append(len(q2))
    
    return pad_sequences(x1,maxlen,padding='post'), pad_sequences(x2,maxlen,padding='post'), np.array(l1),np.array(l2)



import sys
import os
from ipywidgets import FloatProgress
import time
from IPython.display import display
def evaluate (self, df, is_training, batch_size, sess, dropout_prob = 0.2):
    X = get_feature_X(df,maxlen)
    Y = pd.get_dummies(df.is_duplicate)
    sess = self.sess
    start_index = 0
    final_loss = 0
    final_acc = 0
    current_total_trained =0  
    p_bar = FloatProgress()
    display(p_bar)
    start_time = time.time()
    while start_index < X[0].shape[0]:
        temp_x1 = X[0][start_index:start_index+batch_size]
        temp_x2 = X[1][start_index:start_index+batch_size]
        temp_seq_len1 = X[2][start_index:start_index+batch_size]
        temp_seq_len2 = X[3][start_index:start_index+batch_size]
        test_y = Y[start_index:start_index+batch_size]

        feed_dict = {
            self.min_mask1: get_init_min_mask_value(temp_seq_len1),
            self.min_mask2: get_init_min_mask_value(temp_seq_len2),
            self.seq_length1: temp_seq_len1,
            self.seq_length2: temp_seq_len2,
            self.input: temp_x1,
            self.input2: temp_x2,
            self.y: test_y
        }
        
        if is_training:
            feed_dict[self.prob] = 1 - dropout_prob
        
        current_total_trained += temp_x1.shape[0]

        if is_training:
            # the exact output you're looking for:
            _, c, ac =  sess.run([self.optimizer, self.loss, self.acc], feed_dict=feed_dict)
            final_loss += c * temp_x1.shape[0]
            final_acc += ac * temp_x1.shape[0]
            #print("%s/%s training loss %s"  % (start_index, X[0].shape[0], final_loss/current_total_trained))
#             sys.stdout.write("\r%s/%s training loss %s"  % (start_index, X[0].shape[0], c))
#             sys.stdout.flush()
            duration = time.time() - start_time
            speed = duration/current_total_trained
            eta = (X[0].shape[0]-current_total_trained)*speed
            p_bar.value = current_total_trained/X[0].shape[0]
            p_bar.description = "%s/%s, eta %s sec"%(current_total_trained, X[0].shape[0], eta)
        else:
            c, ac, pred, real =  sess.run([self.loss, self.acc, self.output, self.y], feed_dict=feed_dict)
            final_loss += c * temp_x1.shape[0]
            final_acc += ac * temp_x1.shape[0]
            # print('real:', real)
            # print('pred:', pred)
            print(sum(np.argmax(real, axis=1)==np.argmax(pred, axis=1)))
        start_index += batch_size
        
    final_loss = final_loss/X[0].shape[0]
    final_acc = final_acc/X[0].shape[0]
    return final_loss, final_acc

def gradients(self, df , batch_size, sess):
    X = get_feature_X(df,maxlen)
    Y = pd.get_dummies(df.is_duplicate)
    sess = self.sess
    start_index = 0
    final_loss = 0
    current_total_trained =0  
    p_bar = FloatProgress()
    display(p_bar)
    start_time = time.time()
    while start_index < X[0].shape[0]:
        temp_x1 = X[0][start_index:start_index+batch_size]
        temp_x2 = X[1][start_index:start_index+batch_size]
        temp_seq_len1 = X[2][start_index:start_index+batch_size]
        temp_seq_len2 = X[3][start_index:start_index+batch_size]
        test_y = Y[start_index:start_index+batch_size]

        feed_dict = {
            self.min_mask1: get_init_min_mask_value(temp_seq_len1),
            self.min_mask2: get_init_min_mask_value(temp_seq_len2),
            self.seq_length1: temp_seq_len1,
            self.seq_length2: temp_seq_len2,
            self.input: temp_x1,
            self.input2: temp_x2,
            self.y: test_y
        }
        
      
        current_total_trained += temp_x1.shape[0]
        
        var_grad = tf.gradients(self.loss, [self.output])[0]
 
        # the exact output you're looking for:
        g =  sess.run([var_grad, self.concat_output], feed_dict=feed_dict)
        print("gradient %s"  % (g))
#             sys.stdout.write("\r%s/%s training loss %s"  % (start_index, X[0].shape[0], c))
#             sys.stdout.flush()
        duration = time.time() - start_time
        speed = duration/current_total_trained
        eta = (X[0].shape[0]-current_total_trained)*speed
        p_bar.value = current_total_trained/X[0].shape[0]
        p_bar.description = "%s/%s, eta %s sec"%(current_total_trained, X[0].shape[0], eta)

        start_index += batch_size
        break
        
    final_loss = final_loss/X[0].shape[0]
    return final_loss


def fit(self, train_df, val_df, epochs, dropout_prob=0.2, batch_size=64, check_point_name="./default_cnn_model"):

    sess = self.sess
    
    saver = tf.train.Saver(tf.global_variables ())
    best_epoch = 0
    best_loss = 1e9
    os.mkdir(check_point_name)
#     saver.save(self.sess, check_point_name+'/model', global_step=0)
    for i in range(epochs):
        print("training epoch ",i)
        train_loss, train_acc = evaluate(self,train_df,True,batch_size, sess, dropout_prob=dropout_prob)
        print("train loss:{}, train acc:{}".format(train_loss, train_acc))
        loss, acc = evaluate(self, val_df, False, 64, sess)
        print("val loss:{}, val acc:{}".format(loss, acc))
        if loss < best_loss:
            best_epoch = i
            best_loss = loss
            print("save best_epoch %s to %s"%(best_epoch,check_point_name))
            saver.save(self.sess, check_point_name+'/model', global_step=i)
            
    return best_loss

from keras.models import load_model
def tunning_model(model):
    # dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
    dropouts = [0.2]
    for d in dropouts:
        print("train with dropout %s"%(d))
        model.sess.run(model.init)
        best_loss = fit(model, train_df, val_df, epochs=5, dropout_prob=d, check_point_name="./mpcnn_model_%s"%(d))
        with open('mpcnn_val_result.txt','a') as f:
            f.write(str({'dropout': d, 'score': best_loss})+"\n")
        

# tunning_model(model)
#
#
# import json
# for line in open('./mpcnn_val_result.txt','r'):
#     print(line)
#
#
# # dropout 0.2 is best, as loss function = 0.31134441
#
#
# saver = tf.train.Saver()
#
# saver.restore(model.sess, './mpcnn_model_0.2/model-2')

import time
start = time.time()
for _ in range(10):
    test_loss, test_acc = evaluate(model, test_df, False, 64, model.sess)
    print(test_loss, test_acc)
print('run time: ', time.time()-start)



