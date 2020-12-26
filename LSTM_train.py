import jieba
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import LSTM_model
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

SENTENCE_LIMIT_SIZE=300
EMBEDDING_SIZE=300
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SEED = 123
hidden_dim=64
n_layers=2
# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 为CPU设置随机种子
torch.manual_seed(123)


stopwordFile='.\Data/stopwords.txt'
trainFile='.\Data/new.csv'
wordLabelFile = 'wordLabel.txt'
lengthFile = 'length.txt'


def read_stopword(file):
    data = open(file, 'r', encoding='utf-8').read().split('\n')

    return data

def get_batch(x,y,batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0], print("error shape!")


    n_batches = int(x.shape[0] / batch_size)      #统计共几个完整的batch

    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]

        yield x_batch, y_batch

def loaddata(trainfile,stopwordfile):
    a=pd.read_csv(trainfile,encoding='gbk')
    stoplist = read_stopword(stopwordfile)
    text=a['rateContent']
    y=a['other']
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
x_data_1=torch.LongTensor(x_data)


wvmodel=KeyedVectors.load_word2vec_format('word60.vector')
static_embeddings = np.zeros([VOCAB_SIZE,EMBEDDING_SIZE ])
for word, token in tqdm(a.items()):
    #用词向量填充
        if word in wvmodel.vocab.keys():
            static_embeddings[token, :] = wvmodel[word]
        elif word == '<pad>':                                                           #如果是空白，用零向量填充
            static_embeddings[token, :] = np.zeros(EMBEDDING_SIZE)
        else:                                                                           #如果没有对应的词向量，则用随机数填充
            static_embeddings[token, :] = 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1

print(static_embeddings.shape)
X_train,X_test,y_train,y_test=train_test_split(x_data, y, test_size=0.3)
print(X_train.shape, y_train.shape)
y=np.reshape(-1,1)

rnn = LSTM_model.BiLSTM_Attention(
                                  vocab_size=VOCAB_SIZE,
                                  embedding_dim=EMBEDDING_SIZE,
                                  hidden_dim=hidden_dim,
                                  n_layers=n_layers
    )
rnn.embedding.weight.data.copy_(torch.FloatTensor(static_embeddings))
optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()


# 计算准确率
def binary_acc(preds, y):
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc



# 训练函数
def train(rnn, optimizer, criteon):
    avg_loss = []
    avg_acc = []
    rnn.train()  # 表示进入训练模式
    for x_batch, y_batch in get_batch(X_train,y_train):
        x_batch = torch.LongTensor(x_batch)
        y_batch = torch.LongTensor(y_batch.to_numpy())

        pred = rnn(x_batch)
        print(pred.shape)
        acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
        avg_acc.append(acc)

        loss = criteon(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = np.array(avg_acc).mean()
    return avg_acc, loss


# 评估函数
def evaluate(rnn, criteon):
    avg_loss = []
    avg_acc = []
    rnn.eval()  # 进入测试模式

    with torch.no_grad():
        try:
            for x_batch, y_batch in get_batch(X_test, y_test):
                x_batch = torch.LongTensor(x_batch)
                y_batch = torch.LongTensor(y_batch.to_numpy())

                pred = rnn(x_batch)
                loss = criteon(pred, y_batch)
                acc = binary_acc(torch.max(pred, dim=1)[1], y_batch).item()

                avg_acc.append(acc)
        except:
            pass

    avg_acc = np.array(avg_acc).mean()
    return avg_acc


for epoch in range(10):

    train_acc,loss = train(rnn, optimizer, criterion)
    print('epoch={},训练准确率={},loss ={}'.format(epoch, train_acc,loss))
    test_acc= evaluate(rnn, criterion)
    print("epoch={},测试准确率={}".format(epoch, test_acc))

torch.save(rnn.state_dict(), 'model-LSTM4.pkl')