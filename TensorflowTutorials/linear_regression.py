#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

trX = np.linspace(-1, 1, 101) # 创建均匀分布在-1到1之间的101个离散点，返回一个数组(列表)： [-1, -0.98, -0.96, ... , 0.96, 0.98, 1]
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise, 
                                                   # *trX表示取数组里面的每个元素, *trX.shape=101, randn产生101个0-1的数
trXPlot = np.arange(-1, 1, 0.01) # 画预测直线用，提高精度

a = plt.subplot(1,1,1)
pointShow = a.scatter(trX, trY, c = 'b', marker = 'x') # scatter表示画离散的点

X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")

def model(X, w):
    return tf.multiply(X, w) # lr is just X*w so this model line is pretty simple
    						 # 定义线性回归模型为简单的过原点的X*w的方程

w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

cost = tf.square(Y - y_model) # use square error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip(trX, trY): # 将离散点用zip打包成[(-1, -2), (-0.98, -2*-0.98), (-0.96, 2*-0.96), ... , (0.96, 2*0.96), (0.98, 2*0.98), (1, 2*1)]
            sess.run(train_op, feed_dict={X: x, Y: y})

    #print(sess.run(w))  # It should be something around 2
    lineResultShow = a.plot(trXPlot, trXPlot*sess.run(w) , 'r-') # plot拟合直线  

plt.title("Points VS Prediction Line")
plt.xlabel("trX")
plt.ylabel("trY")
plt.show()