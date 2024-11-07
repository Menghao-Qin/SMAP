#!/usr/bin/env python
import os
os.environ['MPLCONFIGDIR'] ='/hwfssz1/ST_EARTH/P18Z10200N0112/USER/qinmenghao/ml/amp'
import numpy as np
from Bio import SeqIO
import torch
import sys
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import *
from dataset import *
import os, sys
import numpy as np
import glob
import scikitplot as skplt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
sequences1 = []
labels1 = []
for record in SeqIO.parse('/hwfssz1/ST_EARTH/P18Z10200N0112/USER/qinmenghao/ml/amp/AMP.fa', 'fasta'):
    sequences1.append(record.seq)
    labels1.append('amp')
amino_acids= {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
              'T':17,'V':18,'W':19,'Y':20}
index_sequences1 = []
for sequence in sequences1:
    index_sequence1 = []
    for aa in sequence:
        index_sequence1.append(amino_acids.get(aa))
    index_sequences1.append(index_sequence1)
sequences2 = []
labels2 = []
for record in SeqIO.parse('/hwfssz1/ST_EARTH/P18Z10200N0112/USER/qinmenghao/ml/amp/non-AMPs.fa','fasta'):
    sequences2.append(str(record.seq))
    labels2.append('Non-amp')
    index_sequences2 = []
for sequence in sequences2:
    index_sequence2 = []
    for aa in sequence:
        index_sequence2.append(amino_acids.get(aa))
    index_sequences2.append(index_sequence2)

index_sequences_1 = [[x for x in sublist if x is not None] for sublist in index_sequences1]
index_sequences_2 = [[x for x in sublist if x is not None] for sublist in index_sequences2]
max_length = 200
Y_sequence = [np.pad(seq, (max_length - len(seq), 0), mode='constant') for seq in index_sequences_1]
N_sequence = [np.pad(seq, (max_length - len(seq), 0), mode='constant') for seq in index_sequences_2]

Y_Sequences2 = torch.from_numpy(np.array(Y_sequence))
N_Sequences2 = torch.from_numpy(np.array(N_sequence))

Data = torch.cat((Y_Sequences2, N_Sequences2), 0).type(torch.FloatTensor)

Labels = [1 if label == 'amp' else 0 for label in (labels1 + labels2)]
Labels = torch.Tensor(Labels)

seqs, labels = Data, Labels

X_train, X_test, y_train, y_test = train_test_split(seqs.long().numpy(), labels.long().numpy(), stratify=labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=42)
class Dataset:
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        if mode == 'train':
            self.seqs, self.labels = X_train, y_train
        elif mode == 'test':
            self.seqs, self.labels = X_test, y_test
        elif mode == 'val':
            self.seqs, self.labels = X_val, y_val

        self.seqs = self.seqs
        self.labels = self.labels
    def __len__(self):
            return self.labels.shape[0]

    def __getitem__(self, index):
        seq, label = self.seqs[index], self.labels[index]
        seq = torch.from_numpy(seq)
        label = torch.tensor(label)

        return seq, label
if __name__ == '__main__':
    dataset = Dataset()
    seq, label = dataset[0]

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, output, target):
        positive_samples = torch.sum(target == 1)
        negative_samples = torch.sum(target == 0)
        total_samples = positive_samples + negative_samples
        
        positive_weight = total_samples / (2 * positive_samples)
        negative_weight = total_samples / (2 * negative_samples)
        
        weight = torch.tensor([negative_weight, positive_weight])
        loss = nn.CrossEntropyLoss(weight=weight)
        return loss(output, target)


loss_func = WeightedLoss()
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.softmax = nn.Softmax(dim=1)

        nn.init.uniform_(self.att_weights.data, -0.1, 0.1)

    def forward(self, hidden):
        weights = torch.matmul(hidden, self.att_weights).squeeze(2)
        weights = self.softmax(weights)
        weighted_hidden = torch.mul(hidden, weights.unsqueeze(2).expand_as(hidden))
        return torch.sum(weighted_hidden, dim=1)
class LSTM(nn.Module):
    def __init__(self, vocab_size=21, embedding_dim=32, bilstm=True, hidden_size=128, num_layers=2, n_cls=2):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bilstm, dropout=0.05)
        self.attention = Attention(hidden_size*2 if bilstm else hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2 if bilstm else hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_cls)
        )

    def forward(self, x):
        embedding = self.embedding(x)
        features, _ = self.lstm(embedding)
        attention_output = self.attention(features)
        logits = self.fc(attention_output)

        return logits
if __name__ == '__main__':
    model = LSTM()
    batch_size = 32
    input_x = torch.randint(0, 21, size=(batch_size, 50))

    out = model(input_x)
root = sys.path[0]
train_data = Dataset(mode='train')
test_data = Dataset(mode='test')
val_data = Dataset(mode='val')

# 构造批量数据
batch_size = 32 # 批次大小
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 20
bilstm_lstm_acc = []
bilstm_lstm_loss = []
model = LSTM(vocab_size=21, embedding_dim=32, bilstm=True, hidden_size=128, num_layers=2, n_cls=2).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3) # 优化器
#loss_func = nn.CrossEntropyLoss() # 损失函数
loss_func = WeightedLoss()
train_loss = []
val_loss = []
train_acc = []
val_acc = []
best_acc = 0

for epoch in range(epochs):
    print('\n*****************\n\nepoch', epoch+1)

    _loss = 0
    _acc = 0
    model.train()
    for i, (seqs, labels) in enumerate(train_loader):
        seqs = seqs.to(device)
        labels = labels.to(device)

        out = model(seqs) # 推理 返回logits
        _acc += (out.argmax(dim=1) == labels).sum().item() / seqs.size(0) # 计算准确率
        loss = loss_func(out, labels) # 计算交叉熵损失函数
        _loss += loss.item()
        optim.zero_grad() # 梯度清零
        loss.backward() # 误差反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3., norm_type=2) # 梯度裁减
        optim.step() # 更新神经元参数

    train_loss.append(_loss/(i+1))
    train_acc.append(_acc/(i+1))
    print(f'bilstm' , 'train loss:', train_loss[-1], 'acc:', train_acc[-1])

    with torch.no_grad():
        _loss = 0
        _acc = 0
        model.eval()
        for i, (seqs, labels) in enumerate(val_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            out = model(seqs)

            _acc += (out.argmax(dim=1) == labels).sum().item() / seqs.size(0)
            _loss += loss_func(out, labels).item()
        val_loss.append(_loss/(i+1))
        val_acc.append(_acc/(i+1))

    if val_acc[-1] >= best_acc:
        best_acc = val_acc[-1]
        torch.save(model, os.path.join(root, f"bilstm_classifier.pth"))
    l = list(range(1, epoch+2))
    plt.plot(l, train_loss, color='r', label='train')
    plt.plot(l, val_loss, color='y', label='val')
    plt.title('loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(root, f"bilstm_loss.jpg")) # 保存迭代误差曲线图
    plt.clf()

    plt.plot(l, train_acc, color='r', label='train')
    plt.plot(l, val_acc, color='b', label='val')
    plt.legend(loc='best')
    plt.title('accuracy')
    plt.savefig(os.path.join(root, f"bilstm_acc.jpg")) # 保存迭代准确率变化曲线图
    plt.clf()
bilstm_lstm_acc.append([train_acc, val_acc]) 
bilstm_lstm_loss.append([train_loss, val_loss])
for model_type in ['bilstm_classifier2']:
    print(model_type, ':')
    model = torch.load(os.path.join(root, f'{model_type}.pth'), map_location=device).eval()
    y_true = []
    predict_proba = []
    with torch.no_grad():
        model.eval()
        for i, (seqs, labels) in enumerate(test_loader):
            print(f'TEST... {i+1}/{len(test_loader)}')
            seqs = seqs.to(device)
            y_true.append(labels)
            labels = labels.to(device)

            out = model(seqs)
            predict_proba.append(out.detach().cpu())
skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
plt.savefig(os.path.join(root, f'{model_type}_confusion matrix.jpg'))
plt.show()
def plot_precision_recall_curve(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    average_precision = auc(recall, precision)

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve (AUC={0:0.2f})'.format(average_precision))   
    plt.savefig(os.path.join(root, 'Precision-Recall2.jpg'))
    plt.show()


# 绘制ROC曲线
def plot_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(root, 'ROC2.jpg'))
    plt.show()
# 绘制PR曲线
plot_precision_recall_curve(y_true, y_pred)
skplt.metrics.plot_precision_recall(y_true, predict_proba)
plt.savefig(os.path.join(root, f'{model_type}_PR.jpg'))

# 绘制ROC曲线
plot_roc_curve(y_true, y_pred) 
# 计算召回率
print("Recall score:")
print(recall_score(y_true, y_pred, average='macro'))

# 计算精确率
print("Precision score:")
print(precision_score(y_true, y_pred, average='macro'))

# 计算F1值
print("F1 score:")
print(f1_score(y_true, y_pred, average='macro'))

