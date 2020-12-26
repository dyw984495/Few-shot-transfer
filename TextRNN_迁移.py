import jieba
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

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
wvmodel=KeyedVectors.load_word2vec_format('word60.vector')
static_embeddings = np.zeros([VOCAB_SIZE,EMBEDDING_SIZE ])
for word, token in tqdm(a.items()):

        if word in wvmodel.vocab.keys():
            static_embeddings[token, :] = wvmodel[word]
        elif word == '<pad>':
            static_embeddings[token, :] = np.zeros(EMBEDDING_SIZE)
        else:
            static_embeddings[token, :] = 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1

print(static_embeddings.shape)

X_train,X_test,y_train,y_test=train_test_split(x_data, y, test_size=0.3)

def get_batch(x,y,batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")


    n_batches = int(x.shape[0] / batch_size)      #统计共几个完整的batch

    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]

        yield x_batch, y_batch


class TextRNN_Var(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, n_layers):
        super(TextRNN_Var, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.8)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(2*hidden_dim, n_layers)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        logit = self.fc(output[:, -1, :])
        return logit

def resnet_cifar(net, input_data):
    x = net.dropout(net.embedding(input_data))
    out, (final_hidden_state, final_cell_state) = net.rnn(x)
    return out

class next(nn.Module):
    def __init__(self, output_size ):
        super(next, self).__init__()
        self.fc = nn.Linear(128, output_size)

    def forward(self,x):
        logit = self.fc(x[:, -1, :])
        return logit

rnn=TextRNN_Var(vocab_size=VOCAB_SIZE,embedding_dim=EMBEDDING_SIZE,hidden_dim=64,n_layers=2)
rnn.load_state_dict(torch.load('./model-TextRNN1.pkl'))
rnnnext=next(2)
optimizer = optim.Adam(rnnnext.parameters(), lr=LEARNING_RATE)
criteon = nn.CrossEntropyLoss()

def train(rnnnext, optimizer, criteon):

    global loss
    avg_acc = []
    rnnnext.train()        #表示进入训练

    for x_batch, y_batch in get_batch(X_train,y_train):
        try:
            x_batch = torch.LongTensor(x_batch)
            y_batch = torch.LongTensor(y_batch.to_numpy())

            y_batch = y_batch.squeeze()
            x_batch = resnet_cifar(rnn, x_batch)
            pred = rnnnext(x_batch)
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

def evaluate(rnnnext, criteon):
    avg_acc = []
    rnnnext.eval()         #表示进入测试模式

    with torch.no_grad():
        for x_batch, y_batch in get_batch(X_test, y_test):
            try:
                x_batch = torch.LongTensor(x_batch)
                y_batch = torch.LongTensor(y_batch.to_numpy())

                y_batch = y_batch.squeeze()       #torch.Size([128])

                x_batch = resnet_cifar(rnn, x_batch)
                pred = rnnnext(x_batch)               #torch.Size([128, 2])

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

for epoch in range(10):

    train_acc,loss = train(rnnnext, optimizer, criteon)
    print('epoch={},训练准确率={},误判率 ={}'.format(epoch, train_acc,loss))
    test_acc = evaluate(rnnnext, criteon)
    print("epoch={},测试准确率={}".format(epoch, test_acc))
