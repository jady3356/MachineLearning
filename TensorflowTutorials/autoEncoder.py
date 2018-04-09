import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#自动编码器
#作用：降维和特征学习
#原理：由编码器和解码器组成，基本思想是用较少神经元的隐藏层来提取输入层的特征，达到特征提取和降维的目的。是一种无监督学习，利用反向传播的方法使得目标值等于输入值。

import matplotlib # to plot images
# Force matplotlib to not use any X-server backend.
matplotlib.use('Agg') # 表示只画保存图片，并不显示出来
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## Visualizing reconstructions
def vis(images, save_name):
    dim = images.shape[0] # 取图像的维度784，shape[0]即为行的维度数784
    n_image_rows = int(np.ceil(np.sqrt(dim))) # 得到图像的宽度28，ceil向上取整
    n_image_cols = int(np.ceil(dim * 1.0/n_image_rows)) # 得到图像的高度28
    gs = gridspec.GridSpec(n_image_rows,n_image_cols,top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
    for g,count in zip(gs,range(int(dim))):
        ax = plt.subplot(g)
        ax.imshow(images[count,:].reshape((28,28)))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_name + '_vis.png')

mnist_width = 28
n_visible = mnist_width * mnist_width # 784
n_hidden = 500
corruption_level = 0.3

# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X') # [None, 784]

# create node for corruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask') # [None, 784]

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden)) # 0.27
W_init = tf.random_uniform(shape=[n_visible, n_hidden], # [784, 500]
                           minval=-W_init_max, # -0.27
                           maxval=W_init_max)  # +0.27

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder, [500,784]
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime') # [784]


def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X  # corrupted X, [None, 784] * [None, 784]

    Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state, [None, 500]
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input, [None, 784]
    return Z

# build model graph
Z = model(X, mask, W, b, W_prime, b_prime) # [None, 784]

# create cost function
cost = tf.reduce_sum(tf.pow(X - Z, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer
predict_op = Z
# load MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            input_ = trX[start:end] # [128,784]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape) # binomial:二项概率分布，n=1,p=0.7的概率, 类似dropout
            sess.run(train_op, feed_dict={X: input_, mask: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
        print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))
    # save the predictions for 100 images
    mask_np = np.random.binomial(1, 1 - corruption_level, teX[:100].shape)
    predicted_imgs = sess.run(predict_op, feed_dict={X: teX[:100], mask: mask_np})
    input_imgs = teX[:100] # 取100张
    # plot the reconstructed images
    vis(predicted_imgs,'pred')
    vis(input_imgs,'in')
    print('Done')