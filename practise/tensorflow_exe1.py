
# coding: utf-8

# In[207]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import mpl_toolkits.mplot3d  

loss_plot = []
weight_plot_x = []
weight_plot_y = []

x_data = np.float32(np.random.rand(2,400))
#print(x_data.shape)# 1x300
#y_data = np.square(x_data - 5)
y_data = np.dot([1, -10], x_data) + 25
#y_data_view = x_data[0] * 1 + (-10)*x_data[1] + 25
#y_data_view = y_data_view.reshape(500,1)
#print(y_data.shape)

#x_data[0,:], x_data[1,:] = np.meshgrid(x_data[0], x_data[1])

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure(figsize=(9,6))
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x_data[0], x_data[1], y_data, rstride=2, cstride=2, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)


#x = np.arange(0,10, 0.25)
#y = np.square(x-5)
#x,y = np.meshgrid(x, y)
#z = 
#plt.figure(1)
#plt.plot(x, y, linewidth=1.0, linestyle="-")
#plt.scatter(x, y, c = 'b',marker = 'x')  

#ax=plt.subplot(111,projection='3d')  
#surf = ax.plot_wireframe(x_data[0], x_data[1], y_data_view ,rstride=1,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8) 
ax = plt.subplot(211, projection='3d') 
ax.scatter(x_data[0], x_data[1],y_data, c = 'b',marker = 'x')  

weight = tf.Variable(tf.random_uniform([1,2], dtype = tf.float32))
#print(weight.shape)
bias = tf.Variable(tf.zeros([1]))
y = tf.matmul(weight, x_data) + bias

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(400): 
    sess.run(train)
    if  step%20 == 0: 
         print(step,sess.run(weight),sess.run(bias), sess.run(loss))
         loss_plot.append(sess.run(loss))
         weight_plot_x.append(sess.run(weight)[0][0])
         weight_plot_y.append(sess.run(weight)[0][1])

#print(loss_plot)
#print(weight_plot_x)
#print(weight_plot_y)

ax = plt.subplot(212, projection='3d') 
ax.scatter(weight_plot_x, weight_plot_y, loss_plot, c = 'b',marker = 'x')  



