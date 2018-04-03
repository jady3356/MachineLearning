#导入依赖库
import numpy as np #这是Python的一种开源的数值计算扩展，非常强大
import tensorflow as tf  #导入tensorflow 

##构造数据##
x_data=np.random.rand(100).astype(np.float32) #随机生成100个类型为float32的值
y_data=x_data*0.1+0.3  #定义方程式y=x_data*A+B

x_data = np.random.rand(2,100)
y_data = np.dot([1,-10], x_data) + 25
##-------##

##建立TensorFlow神经计算结构##
weight=tf.Variable(tf.random_uniform([1,2],-1.0,20)) 
biases=tf.Variable(tf.zeros([1]))     
y=tf.matmul(weight, x_data)+biases
##-------##


loss=tf.reduce_mean(tf.square(y-y_data))  #判断与正确值的差距
optimizer=tf.train.GradientDescentOptimizer(0.5) #根据差距进行反向传播修正参数
train=optimizer.minimize(loss) #建立训练器

init=tf.global_variables_initializer() #初始化TensorFlow训练结构
sess=tf.Session()  #建立TensorFlow训练会话
sess.run(init)     #将训练结构装载到会话中

for  step in range(400): #循环训练400次
     sess.run(train)  #使用训练器根据训练结构进行训练
     if  step%20==0:  #每20次打印一次训练结果
        print(step,sess.run(weight),sess.run(biases)) #训练次数，A值，B值