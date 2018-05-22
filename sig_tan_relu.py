# !/usr/bin/python3

# coding:utf8
# @Author: Jlan
# @Time: 18-4-8 上午10:35

import matplotlib.pyplot as plt
import numpy as np
import math

def sigmod(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    y = np.tanh(x)
    return y

def relu(x):
    y = x.copy()
    y[y<0]=0
    return y

x = np.arange(-50.0,50.0,0.1)
y_relu = relu(x)
y_sigmod = sigmod(x)
y_tanh = tanh(x)

plt.plot(x,y_relu,c='r',label="Relu",linestyle='--')
plt.plot(x,y_sigmod,c='g',label="Sigmod",linestyle='-.')
plt.plot(x,y_tanh,c='b',label="Tanh")
plt.ylim([-1,4])
plt.xlim([-4,4])
plt.legend(loc=2)
plt.savefig('sig_tan_relu.png')
plt.show()