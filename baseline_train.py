import jieba
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import backbone

SENTENCE_LIMIT_SIZE=100
EMBEDDING_SIZE=192
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

stopwordFile='./Data/stopwords.txt'
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
    y=a['logistics']
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
VOCAB_SIZE = 883183
x,y=loaddata(trainFile,stopwordFile)
print(x)
print(y)
def convert_text_to_token(sentence, word_to_token_map=a, limit_size=SENTENCE_LIMIT_SIZE):
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]

    tokens = [word_to_token_map.get(word, unk_id) for word in sentence]

    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    else:
        tokens = tokens[:limit_size]

    return tokens

x_data=[convert_text_to_token(sentence) for sentence in x]
x_data=np.array(x_data)
x_data_1=torch.LongTensor(x_data)
print(x_data.shape)

wvmodel=KeyedVectors.load_word2vec_format('word.vector')
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
print(X_train.shape, y_train.shape)
y=np.reshape(-1,1)

def get_batch(x,y,batch_size=BATCH_SIZE, shuffle=True):
    assert x.shape[0] == y.shape[0]


    n_batches = int(x.shape[0] / batch_size)

    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i+1)*batch_size]
        y_batch = y[i*batch_size: (i+1)*batch_size]

        yield x_batch, y_batch

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = backbone.Conv4()
        self.fc = nn.Linear(4608, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        out =self.convs(x)
        out = out.view(x.size(0), -1)
        out = self.dropout(out)
        logit = self.fc(out)
        return logit
#
#
#
#
# # textcnn=TextCNN(352217,300,2)
# # textcnn.load_state_dict(torch.load('bigmodel6.pkl'))
#
cnn = TextCNN(VOCAB_SIZE, 192, 2)
cnn.embedding.weight.data.copy_(torch.FloatTensor(static_embeddings))

optimizer = optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
criteon = nn.CrossEntropyLoss()

def binary_acc(preds, y):
    correct = torch.eq(preds, y).float()
    if len(correct)==0:
        acc=0
    else:
        acc = correct.sum() / len(correct)
    return acc

def train(cnn, optimizer, criteon):


    avg_acc = []
    cnn.train()


    for x_batch, y_batch in get_batch(X_train,y_train):
        try:
            x_batch = torch.LongTensor(x_batch)
            y_batch = torch.LongTensor(y_batch.to_numpy())

            # lasize = y_batch.squeeze().size()
            # zes = Variable(torch.zeros(lasize).type(torch.LongTensor))
            #
            # ons = Variable(torch.ones(lasize).type(torch.LongTensor))

            y_batch = y_batch.squeeze()
            pred = cnn(x_batch)
            # y_1 = textcnn(x_batch)



            # train_correct11 = ((torch.max(pred, dim=1)[1] == ons) & (y_batch == ons)).sum()
            # train_correct00 = ((torch.max(pred, dim=1)[1] == zes) & (y_batch == zes)).sum()
            #
            # FN+= train_correct01.data[0]
            #
            # FP+= train_correct10.data[0]
            #
            # TP+= train_correct11.data[0]
            # TN+= train_correct00.data[0]

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

def evaluate(cnn, criteon):
    avg_acc = []
    cnn.eval()

    with torch.no_grad():
        for x_batch, y_batch in get_batch(X_test, y_test):
            try:
                x_batch = torch.LongTensor(x_batch)
                y_batch = torch.LongTensor(y_batch.to_numpy())

                y_batch = y_batch.squeeze()       #torch.Size([128])
                pred = cnn(x_batch)               #torch.Size([128, 2])

                acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
                avg_acc.append(acc)
            except:
                pass

    avg_acc = np.array(avg_acc).mean()
    return avg_acc

cnn_train_acc, cnn_test_acc = [], []

for epoch in range(10):
    train_acc,loss = train(cnn, optimizer, criteon)
    print('epoch={},训练准确率={},误判率 ={}'.format(epoch, train_acc,loss))
    test_acc = evaluate(cnn, criteon)
    print("epoch={},测试准确率={}".format(epoch, test_acc))
    cnn_train_acc.append(train_acc)
    cnn_test_acc.append(test_acc)

# torch.save(cnn.state_dict(), 'model-baseline5.pkl')
