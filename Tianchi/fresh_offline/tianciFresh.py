#!/usr/bin/python3
#-*- coding: UTF-8 -*-
'''
天池大赛，新人离线赛
author：Taiyi
'''
import tensorflow as tf
import numpy as np
import csv
from pandas import Series,DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import collections

#用csv读取数据，并去重
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

#利用pandas读取并处理csv
data = pd.read_csv(r'fresh_comp_offline\\tianchi_fresh_comp_train_user_test.csv')
#data['user_geohash'].convert_objects(convert_numeric=True)
#data['user_geohash'] = pd.to_numeric(data['user_geohash'], errors="ignore" )
#print(dict(data['user_geohash']))
# 1. 把商品地理位置列转为数值
geohash = data['user_geohash'] # 把商品位置列取出来，这里并没有用values得到列表
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
print(data.info())

#初始化
batch_size = 128

#定义模型

#训练和预测

