#!/usr/bin/python3
#-*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import csv
from pandas import Series,DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import collections
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

'''
##### 天池大赛，新人离线赛
##### 1. 数据处理
##### 2. 定义TF模型
##### 3. 启动训练
##### author：Pei Taiyi
##### Time: 2018-04-19
'''
#用csv读取数据
user_behavior_file = 'fresh_comp_offline\\tianchi_fresh_comp_train_user_test.csv'
user_behavior_set = []

with open(user_behavior_file,"r", encoding='UTF-8') as f:
    reader = csv.reader(f)
    # 读取一行，下面的reader中已经没有该行了
    head_row = next(reader)
    for row in reader:
        # 行号从2开始
        user_behavior_set.append(row)
        #print(reader.line_num, row)

arr_data = np.array(user_behavior_set)
#print(arr_data)

# 利用pandas读取并处理csv
data = pd.read_csv(r'fresh_comp_offline\\tianchi_fresh_comp_train_user.csv')
# 去重 ？？？

print("Raw data info:\n", data.info())
#data['user_geohash'].convert_objects(convert_numeric=True)
#data['user_geohash'] = pd.to_numeric(data['user_geohash'], errors="ignore" )
#print(dict(data['user_geohash']))

'''
##############################
#### 数据清洗和特征处理#########
#### 1. 数据向量化： str -> float/int
#### 2. 特征扩充
#### 3. 归一化
#############################
'''
# 1. 把用户地理位置列转为数值
geohash = data['user_geohash'] # 把用户位置列取出来，这里并没有用values得到列表
count = collections.Counter(geohash).most_common() # 按照每个地址的出现的次数来排序，出现越多排序越靠前，其中也包含了NaN。返回元组列表：[(nan, 2717), ('96p6u4v', 60), ('96p6tuo', 58),...
# Build dictionaries
rdic = [i[0] for i in count] #取每个元组的第一个元素，即所有的地理位置hash值
dic = {w: i for i, w in enumerate(rdic)} #enumerate给每个元素添加index，得到字典：{nan: 0, '96p6u4v': 1, '96p6tuo': 2, '96p6tvd': 3, '96p6tuk': 4,....
#print(dic[np.nan],dic['96p6u4v']) 有点意思，这里一定要用np.nan来索引，nan则不行

# 把每个地理位置映射成index
geohash = [dic[weizhi] for weizhi in geohash] # 遍历每个地址位置key，然后在dic中找到对应的键值value(即数字)
data['user_geohash'] = geohash
#print(data.info())

# 2. 把时间列转数值
datatime = data['time'].str.replace('-','') # 去掉-
data['time'] = datatime.str.replace(' ','').astype(np.int64) # 去掉空格并转为int64
#print(data['time'])
data = pd.DataFrame(data, dtype='float')

# 3. 特征分析和扩充


# 4. 归一化
data = (data - data.mean()) / (data.std())
print("Clean data info:\n",data.info())

# user ID set, 总的用户数20000
user_set = data['user_id'].drop_duplicates().values # drop_duplicates去重，values转列表
print("user ID set:", user_set.shape)

#初始化
batch_size = 1024

# 定义train的输入trX, train的label trY。 验证集teX和teY
X_data_set =  data.loc[0:9999, ['user_id', 'item_id', 'user_geohash', 'item_category', 'time']]
Y_data_set =  data.loc[0:9999, ['behavior_type']]
#teX =  data.loc[20000:31999, ['user_id', 'item_id', 'user_geohash', 'item_category', 'time']]
#teY =  data.loc[20000:31999, ['behavior_type']]

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y_data_set)
# convert integers to dummy variables (one hot encoding)
dummy_Y = np_utils.to_categorical(encoded_Y)

#用sklearn的train_test_split来随机划分训练和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_data_set, dummy_Y, test_size=0.2, random_state=0)

print("X_train:", X_train.shape) 
print("X_train[0:50]:\n")
print(X_train[0:50])
print("Y_train:", Y_train.shape) 
#print("Y_train[0:50]:", Y_train[0:40])
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
#print("Y_test[0:50]:", Y_test[0:40])

#定义模型
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float64)) # 随机初始化w, stddev表示标准差. 输出为(5, 4)的tensor, 所有的值正态分布在-0.02到0.02之间

def model(X, w):
    return tf.matmul(X, w)

# Input data
X = tf.placeholder(tf.float64, shape=[None, 5]) # 5个特征：user_id, item_id, user_geohash, item_category, time
# need to shape [batch_size, 1] for nn.nce_loss
Y = tf.placeholder(tf.float64, shape=[None, 4]) # 4个输出：1,2,3,4. 

w = init_weights([5, 4]) # 5个特征，4个输出

py_x = model(X, w) # [None,5] * [5, 4] = [None, 4]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y)) 
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) 
predict_op = tf.argmax(py_x, 1) # 1表示行内取最大，返回其index:0,1,2,3. shape=(None,)

#训练和预测

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(X_train), 512), range(512, len(X_train)+1, 512)): # (0,127,255,383,...,549999), (128, 256, 384, ..., 550000) -->
                                                                                     # (0, 127), (128, 256),(255, 384), ... 
            sess.run(train_op, feed_dict={X: X_train[start:end], Y: Y_train[start:end]}) # 每次读取128条信息
        #print(i, np.argmax(Y_test, axis=1)[0:10], sess.run(predict_op, feed_dict={X: X_test})[0:10])
        print(i, np.mean(np.argmax(Y_test, axis=1) ==
                        sess.run(predict_op, feed_dict={X: X_test}))) # 打印准确率
        #result = sess.run(predict_op, feed_dict={X: X_test})
        #print(i, result.shape)
