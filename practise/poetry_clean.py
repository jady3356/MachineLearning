#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import collections  
import numpy as np  

'''
author: log16
Data: 2017/5/4
'''
#-------------------------------数据预处理---------------------------#  
   
poetry_file ='poetry.txt'  
   
# 诗集  
poetrys = []  
with open(poetry_file, "r", encoding='UTF-8') as f:  
    for line in f:  # 迭代读取每一行
        try:  
            #line = line.decode('UTF-8')
            line = line.strip(u'\n') # 剥去换行符 
            title, content = line.strip(u' ').split(u':')  # 剥去头尾空格并分离出题目和内容
            content = content.replace(u' ',u'')  # 替换去除内容中的空格
            if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:  
                continue  # 如果这首（行）包含以上字符则直接跳过
            if len(content) < 5 or len(content) > 79:
                continue  # 如果这首太短或者太长都跳过
            content = u'[' + content + u']'  # 给每首头尾加上括号
            poetrys.append(content)  # 合并所有有效诗构成学习样本
        except Exception as e:   
            pass  
   
# 按诗的字数排序  
poetrys = sorted(poetrys,key=lambda line: len(line)) 
print(len(poetrys[1000]))
print('唐诗总数: ', len(poetrys)) 

# 统计每个字出现次数  
all_words = []  
for poetry in poetrys:  
    all_words += [word for word in poetry]  # 分割成word的数组
counter = collections.Counter(all_words)  

count_pairs = sorted(counter.items(), key=lambda x: -x[1]) # 把每个字按出现的频率排序 
#print(count_pairs)
words, _ = zip(*count_pairs)  # words为出现的所有的字
#print(words)

# 取前多少个常用字  
words = words[:len(words)] + (' ',)  
#print(words)
# 每个字映射为一个数字ID  
word_num_map = dict(zip(words, range(len(words))))  #给每个字按照出现的次数从多到少排序形成字典
#print(word_num_map)
# 把诗转换为向量形式，参考TensorFlow练习1  
to_num = lambda word: word_num_map.get(word, len(words))  
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]  # 把所有诗歌向量话，没首诗的每个字都换成字所对应的字典中的排序数

xdata = np.full((3,3), 1, np.int32) 
print(xdata)
for row in range(3):  
    xdata[row,:2] = row
print(xdata)  
ydata = np.copy(xdata)  
ydata[:,:-1] = xdata[:,1:] 
print(ydata)
print(xdata[:,1:])