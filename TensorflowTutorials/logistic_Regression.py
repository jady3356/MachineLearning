#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01)) # 随机初始化w, stddev表示标准差. 输出为(784, 10)的tensor, 所有的值正态分布在-0.02到0.02之间

def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy
                           # X(None, 784) * w(784, 10) = (None, 10)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#print(trX.shape) (55000, 784) 55000张图片

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression
                            # 784个像素，10个特征输出
'''
test = tf.random_normal([20, 10], stddev=0.01)
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    #for x in test.eval():
    a = plt.hist(sess.run(test))
        #print(x)
plt.legend()
plt.show()
'''
py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression
                                # argmax 返回 数组10个数值最大的对应的位置，即输出结果

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)): # (0,127,255,383,...,549999), (128, 256, 384, ..., 550000) -->
                                                                                     # (0, 127), (128, 256),(255, 384), ... 
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]}) # 每次读取128张图片training
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX}))) # 打印准确率
