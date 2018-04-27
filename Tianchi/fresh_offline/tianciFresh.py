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
'''
# 利用pandas读取并处理csv
data = pd.read_csv(r'fresh_comp_offline\\data_clean_feature_expended_10_features.csv')
# 去重 
data = data.drop_duplicates() # 时间精度是1小时，用户有可能是一小时内点击多次，所以还是去掉一个小时内同一个商品的多次点击

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


# 1. 把用户地理位置列转为数值
geohash = data['user_geohash'] # 把用户位置列取出来，这里并没有用values得到列表
count = collections.Counter(geohash).most_common() # 按照每个地址的出现的次数来排序，出现越多排序越靠前，其中也包含了NaN。返回元组列表：[(nan, 2717), ('96p6u4v', 60), ('96p6tuo', 58),...
# Build dictionaries
rdic = [i[0] for i in count] #取每个元组的第一个元素，即所有的地理位置hash值
dic = {w: i for i, w in enumerate(rdic)} #enumerate给每个元素添加index，得到字典：{nan: 0, '96p6u4v': 1, '96p6tuo': 2, '96p6tvd': 3, '96p6tuk': 4,....
#print(dic[np.nan],dic['96p6u4v']) 这里一定要用np.nan来索引，nan则不行

# 把每个地理位置映射成index
geohash = [dic[weizhi] for weizhi in geohash] # 遍历每个地址位置key，然后在dic中找到对应的键值value(即数字)
data['user_geohash'] = geohash
#print(data.info())

# 2. 把时间列转数值
datatime = data['time'].str.replace('-','') # 去掉-
datatime = datatime.str.replace('2014','') # 去掉2014
data['time'] = datatime.str.replace(' ','').astype(np.int64) # 去掉空格并转为int64
#print(data['time'])
data = pd.DataFrame(data, dtype='float')

# 3. 特征工程：分析和扩充
# 商品属性： 被点击收藏，加购物车，购买的总数
# 用户属性： 总点击，收藏数，加购物车，购买数

# user ID set, 总的用户数20000
#user_set = data['user_id'].drop_duplicates() # drop_duplicates去重，values转列表
#item_set = data['item_id'].drop_duplicates()
item_cat_set = data['item_category'].drop_duplicates()
#user_set.to_csv('user_id_set.csv',index=False)
#item_set.to_csv('item_id_set.csv',index=False)
#print("user ID set:", user_set.shape)
#print("item ID set:", item_set.shape) # 4758484 个商品。。。 没办法对商品的属性做操作，计算量太大
print("item cat set:", item_cat_set.shape) # 9557 个商品分类

#data['User_Buy_Ratio'] = data['behavior_type'] # 先为data增加User_Buy_Ratio，值都临时等于购买behavior
#for user_index in user_set: # _为每个用户的行为数据，这里没有用
#    temp = data[data['user_id']==user_index] # temp是一个frame
#    total_behavior_count = temp['behavior_type'].count() # int, 单个user总的点击，购买，收藏，购物车数量
#    buyUser = temp[temp['behavior_type']==4] # butUser也是一个frame，只是过滤的所有的购买行
#    buyUserCount = buyUser['behavior_type'].count() # butUserCount是一个int, 单个user总的购买数
#    data['User_Buy_Ratio'][data['user_id']==user_index] = buyUserCount/total_behavior_count # 总的购买数/总的点击数


data['Item_cat_Hot_Ratio'] = data['User_Buy_Ratio']
for item_cat_index in item_cat_set:
    temp = data[data['item_category']==item_cat_index] # 单个商品的dataframe
    total_behavior_count = temp['behavior_type'].count()
    bought =  temp[temp['behavior_type']==4]
    boughtCount = bought['behavior_type'].count()
    data['Item_cat_Hot_Ratio'][data['item_category']==item_cat_index] = boughtCount/total_behavior_count

#print(data['behavior_type'].groupby(data['user_id']).count()) # 每个user总的收藏，浏览，购买，购物车数量
print(data.head(10))
data.to_csv('data_clean_feature_expended_2.csv', index=False)


# 4. 归一化
#data = (data - data.mean()) / (data.std())
#data['behavior_type'].loc[data['behavior_type'] < 4] = 0
#data['behavior_type'].loc[data['behavior_type'] == 4] = 1

user_set = data['user_id'].drop_duplicates()
data['User_like_Cat_1'] = data['item_category']
data['User_like_Cat_2'] = data['item_category']
data['User_like_Cat_3'] = data['item_category']
for user_index in user_set:
	temp = data[data['user_id']==user_index]['item_category']
	tempCount = collections.Counter(temp).most_common()
	#print(tempCount[0][0], tempCount[1][0],tempCount[2][0])
	data['User_like_Cat_1'][data['user_id']==user_index] = tempCount[0][0]
	if len(tempCount) >= 2:
	    data['User_like_Cat_2'][data['user_id']==user_index] = tempCount[1][0]
	else:
		data['User_like_Cat_2'][data['user_id']==user_index] = 0
	if len(tempCount) >= 3:
	    data['User_like_Cat_3'][data['user_id']==user_index] = tempCount[2][0]
	else:
	    data['User_like_Cat_3'][data['user_id']==user_index] = 0
 
data.to_csv('data_clean_feature_expended_10_features.csv', index=False)
'''
print("Clean data info:\n", data.info())
print(data.head(50))
#初始化
batch_size = 1024

# 定义train的输入trX, train的label trY。 验证集teX和teY
X_data_set =  data.loc[0:9999999, ['user_id', 'item_id', 'user_geohash', 'item_category', 'time', 'User_Buy_Ratio', 'Item_cat_Hot_Ratio', \
                                   'User_like_Cat_1', 'User_like_Cat_2', 'User_like_Cat_3']]
X_data_set = (X_data_set - X_data_set.mean()) / (X_data_set.std()) # 归一化
Y_data_set =  data.loc[0:9999999, ['behavior_type']]

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y_data_set)
# convert integers to dummy variables (one hot encoding)
dummy_Y = np_utils.to_categorical(encoded_Y)
#print("X_data_set[0:50]:", X_data_set[0:50])
#print("dummy_Y[0:50]:", dummy_Y[0:50])
#用sklearn的train_test_split来随机划分训练和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_data_set, dummy_Y, test_size=0.2, random_state=0)

print("X_train:", X_train.shape) 
print("Y_train:", Y_train.shape) 
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
#print("Y_test[0:50]:", Y_test[0:40])

#定义模型
def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float64), name=name) # 随机初始化w, stddev表示标准差. 输出为(7, 4)的tensor, 所有的值正态分布在-0.02到0.02之间

def model(X, w):
    return tf.matmul(X, w)

# Input data
X = tf.placeholder(tf.float64, shape=[None, 10], name='X') # 7个特征：user_id, item_id, user_geohash, item_category, time + 2(User_Buy_Ratio, Item_Hot_Ratio)
# need to shape [batch_size, 1] for nn.nce_loss
Y = tf.placeholder(tf.float64, shape=[None, 4], name='Y') # 4个输出：1,2,3,4. 

w = init_weights([10, 4],'w') # 7个特征，4个输出
#b = tf.Variable(tf.truncated_normal([4], dtype=tf.float64))

py_x = model(X, w) # [None,7] * [7, 4] = [None, 4]

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y)) 
    train_op = tf.train.GradientDescentOptimizer(0.00002).minimize(cost) 
    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"):
    #predict_op = tf.argmax(py_x, 1) # 1表示行内取最大，返回其index:0,1,2,3. shape=(None,)
    correct_pred =  tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))
    tf.summary.scalar("accuracy", acc_op)

#训练和预测

# Launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/taichi", sess.graph) # for 1.0
    merged = tf.summary.merge_all()
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(tf.global_variables()) 

    for i in range(100):
        for start, end in zip(range(0, len(X_train), 512), range(512, len(X_train)+1, 512)): # (0,127,255,383,...,549999), (128, 256, 384, ..., 550000) -->
                                                                                     # (0, 127), (128, 256),(255, 384), ... 
            sess.run(train_op, feed_dict={X: X_train[start:end], Y: Y_train[start:end]}) # 每次读取128条信息
        #print(i, np.argmax(Y_test, axis=1)[0:10], sess.run(predict_op, feed_dict={X: X_test})[0:10])
        summary, acc = sess.run([merged, acc_op], feed_dict={X: X_test, Y: Y_test})
        writer.add_summary(summary, i)  # Write summary
        saver.save(sess, 'model/tianchi.module') 
        print(i, acc)                   # Report the accuracy

        #print(i, np.mean(np.argmax(Y_test, axis=1) ==
                        #sess.run(predict_op, feed_dict={X: X_test}))) # 打印准确率
        #result = sess.run(predict_op, feed_dict={X: X_test})
        #print(i, result.shape)
