import matplotlib.pyplot as plt
import jieba
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

from torch.autograd import Variable

SENTENCE_LIMIT_SIZE=200
EMBEDDING_SIZE=300
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

stopwordFile='.\Data/stopwords.txt'
trainFile='./Data/multi_brand3000.csv'
wordLabelFile = 'wordLabel.txt'
lengthFile = 'length.txt'

def read_stopword(file):
    data = open(file, 'r', encoding='utf-8').read().split('\n')

    return data

def loaddata(trainfile,stopwordfile):
    a=pd.read_csv(trainfile,encoding='gbk')
    stoplist = read_stopword(stopwordfile)
    text=a['rateContent']
    y=a['package']
    x=[]

    for line in text:
        line=str(line)
        title_seg = jieba.cut(line, cut_all=False)
        use_list = []
        for w in title_seg:
            if w in stoplist:
                continue
            else:
                use_list.append(w)
        x.append(use_list)

    return x,y


def dataset(trainfile,stopwordfile):
    word_to_idx = {}
    idx_to_word = {}
    stoplist = read_stopword(stopwordfile)
    a = pd.read_csv(trainfile,encoding='gbk')
    datas=a['rateContent']
    datas = list(filter(None, datas))
    try:
        for line in datas:
            line=str(line)
            title_seg = jieba.cut(line, cut_all=False)
            length = 2
            for w in title_seg:
                if w in stoplist:
                    continue
                if w in word_to_idx:
                    word_to_idx[w] += 1
                    length+=1
                else:
                    word_to_idx[w] = length
    except:
        pass
    word_to_idx['<unk>'] = 0
    word_to_idx['<pad>'] =1
    idx_to_word[0] = '<unk>'
    idx_to_word[1] = '<pad>'
    return word_to_idx

a=dataset(trainFile,stopwordFile)
print(len(a))
b={v: k for k, v in a.items()}
VOCAB_SIZE = 352217
x,y=loaddata(trainFile,stopwordFile)
def convert_text_to_token(sentence, word_to_token_map=a, limit_size=SENTENCE_LIMIT_SIZE):
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]

    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence]

    if len(tokens) < limit_size:                      #补齐
        tokens.extend([0] * (limit_size - len(tokens)))
    else:                                             #截断
        tokens = tokens[:limit_size]

    return tokens

x_data=[convert_text_to_token(sentence) for sentence in x]
x_data=np.array(x_data)

X_train,X_test,y_train,y_test=train_test_split(x_data, y, test_size=0.3)

def get_batch(x,y,batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")


    n_batches = int(x.shape[0] / batch_size)      #统计共几个完整的batch

    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]

        yield x_batch, y_batch

class TextCNN(nn.Module):   #output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, kernel_lst=(3,4,5), dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                             nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                                            nn.ReLU(),
                                            nn.MaxPool2d((SENTENCE_LIMIT_SIZE - kernel + 1, 1)))
                              for kernel in kernel_lst])
        self.fc = nn.Linear(filter_num * len(kernel_lst), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        out = [conv(x) for conv in self.convs]

        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        out = self.dropout(out)
        logit = self.fc(out)
        return logit


def resnet_cifar(net, input_data):
    x = net.embedding(input_data)
    x = x.unsqueeze(1)
    out = [conv(x) for conv in net.convs]
    out = torch.cat(out, dim=1)
    out = out.view(x.size(0), -1)
    out = net.dropout(out)
    return out

class next(nn.Module):
    def __init__(self, output_size, filter_num=100, kernel_lst=(3, 4, 5)):
        super(next, self).__init__()
        self.fc = nn.Linear(filter_num * len(kernel_lst), output_size)

    def forward(self,x):
        logit = self.fc(x)
        return logit

cnn=TextCNN(VOCAB_SIZE, 300, 2)
cnn.load_state_dict(torch.load('./model1.pkl'))
cnnnext=next(2)
optimizer = optim.Adam(cnnnext.parameters(), lr=LEARNING_RATE)
criteon = nn.CrossEntropyLoss()

def train(cnnnext, optimizer, criteon):


    avg_acc = []
    cnnnext.train()        #表示进入训练

    for x_batch, y_batch in get_batch(X_train,y_train):
        try:
            x_batch = torch.LongTensor(x_batch)
            y_batch = torch.LongTensor(y_batch.to_numpy())

            y_batch = y_batch.squeeze()
            x_batch=resnet_cifar(cnn,x_batch)
            pred = cnnnext(x_batch)

            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)

            loss =criteon(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            pass

    avg_acc = np.array(avg_acc).mean()
    return avg_acc,loss

def evaluate(cnnnext, criteon):
    avg_acc = []
    cnnnext.eval()         #表示进入测试模式

    with torch.no_grad():
        for x_batch, y_batch in get_batch(X_test, y_test):
            try:
                x_batch = torch.LongTensor(x_batch)
                y_batch = torch.LongTensor(y_batch.to_numpy())

                y_batch = y_batch.squeeze()       #torch.Size([128])
                x_batch = resnet_cifar(cnn, x_batch)

                pred = cnnnext(x_batch)               #torch.Size([128, 2])

                acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
                avg_acc.append(acc)
            except:
                pass

    avg_acc = np.array(avg_acc).mean()
    return avg_acc

def binary_acc(preds, y):
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

for epoch in range(15):

    train_acc,loss = train(cnnnext, optimizer, criteon)
    print('epoch={},训练准确率={},误判率 ={}'.format(epoch, train_acc,loss))
    test_acc = evaluate(cnnnext, criteon)
    print("epoch={},测试准确率={}".format(epoch, test_acc))
