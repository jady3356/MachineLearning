#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import collections
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score

##### 天池大赛，新人离线赛
##### 1. 数据处理
##### 2. 定义TF模型
##### 3. 启动训练
##### author：Pei Taiyi
##### Time: 2018-04-26


# 利用pandas读取并处理csv
item_data = pd.read_csv(r'fresh_comp_offline\\tianchi_fresh_comp_train_item.csv')
item_data = item_data.drop_duplicates()
#print(item_data.info())

# 1. 把用户地理位置列转为数值
geohash = item_data['item_geohash'] # 把用户位置列取出来，这里并没有用values得到列表
count = collections.Counter(geohash).most_common() # 按照每个地址的出现的次数来排序，出现越多排序越靠前，其中也包含了NaN。返回元组列表：[(nan, 2717), ('96p6u4v', 60), ('96p6tuo', 58),...
# Build dictionaries
rdic = [i[0] for i in count] #取每个元组的第一个元素，即所有的地理位置hash值
dic = {w: i for i, w in enumerate(rdic)} #enumerate给每个元素添加index，得到字典：{nan: 0, '96p6u4v': 1, '96p6tuo': 2, '96p6tvd': 3, '96p6tuk': 4,....
#print(dic[np.nan],dic['96p6u4v']) 这里一定要用np.nan来索引，nan则不行

# 把每个地理位置映射成index
geohash = [dic[weizhi] for weizhi in geohash] # 遍历每个地址位置key，然后在dic中找到对应的键值value(即数字)
item_data['item_geohash'] = geohash
print(item_data.info()) # item_id, item_geohash, item_category
print(item_data.head(10))



#X = item_data

'''




with tf.Session() as sess:
    tf.global_variables_initializer().run() 
       
    saver = tf.train.Saver(tf.global_variables())  
    saver.restore(sess, 'model/tianchi.module')

    result_data = sess.run([result],feed_dict={X: X_test, Y: Y_test})
'''
