import jieba
import pandas as pd
import re

train_data=pd.read_csv(r"D:\小样本迁移\Data/new.csv",encoding='gbk')
print(train_data)
train_text = []
for line in train_data['rateContent']:
    line=str(line)
    t = jieba.lcut(line)
    train_text.append(t)


sentence_length = [len(x) for x in train_text] #train_text是train.csv中每一行分词之后的数据

import matplotlib.pyplot as plt
plt.hist(sentence_length,100,normed=1,cumulative=True)
plt.xlim(0,1000)
plt.show()