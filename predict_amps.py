#!/usr/bin/env python
import argparse
from Bio import SeqIO
import torch
import matplotlib.pyplot as plt
from torchvision.models import *
from dataset import *
import os
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('input_file', type=str, help='input')
    parser.add_argument('input_model', type=str, help='input_model')
    parser.add_argument('output_num', type=str, help='output_num')
    parser.add_argument('output_seq', type=str, help='output_seq')
    parser.add_argument('output_fa', type=str, help='output_fa')
    return parser.parse_args()

sequences1 = []
if __name__ == "__main__":

    args = parse_args()
    input_file = args.input_file
    input_model = args.input_model
    output_num = args.output_num
    output_seq = args.output_seq
    output_fa = args.output_fa

for record in SeqIO.parse(input_file, 'fasta'):
    sequences1.append(record.seq)
amino_acids= {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,
              'T':17,'V':18,'W':19,'Y':20}
index_sequences1 = []
for sequence in sequences1:
    index_sequence1 = []
    for aa in sequence:
        index_sequence1.append(amino_acids.get(aa))
    index_sequences1.append(index_sequence1)
Y_sequence= [[x for x in sublist if x is not None] for sublist in index_sequences1]
max_length = 200
for i in range(len(Y_sequence)):
        Y_sequence[i] = np.pad(Y_sequence[i], (max_length - len(Y_sequence[i]),0), mode='constant')
Y_sequence = torch.Tensor(Y_sequence)

seqs = Y_sequence
seqs = seqs.long().numpy()
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
model = torch.load(input_model)
class Dataset:
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        
        self.seqs = seqs

        self.seqs = self.seqs
    

    def __getitem__(self, index):
        seq = self.seqs[index]
        seq = torch.from_numpy(seq)
        

        return seq
predict_loader = torch.utils.data.DataLoader(seqs, batch_size=batch_size, shuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
predict_proba = []
with torch.no_grad():
        model.eval()
        for i, seqs in enumerate(predict_loader):
            seqs = seqs.to(device)
            out = model(seqs)
            predict_proba.append(out.detach().cpu())
output_np = out.detach().numpy()
predict_proba = torch.cat(predict_proba, dim=0).numpy()
y_pred = np.argmax(predict_proba, axis=1)
indices = np.where(y_pred == 1)[0]
lst = list(indices)
matching_rows = [row for idx, row in enumerate(sequences1) if idx in lst]
lst = [x + 1 for x in lst]

with open(output_num, 'w') as file:
    for row in matching_rows:
        file.write(' '.join(str(num) for num in row) + '\n')
np.savetxt(output_num, lst, fmt='%d')

with open(output_seq, 'w') as f:
        for seq in sequences1:
            f.write(f"{seq}\n")

# 读取行号
if __name__ == "__main__":
    # 获取输入参数
    args = parse_args()
    output_num1 = args.output_num

    with open(output_num1, 'r') as f:
        line_numbers = {int(line.strip()) for line in f}

    # 提取对应的序列
    extracted_sequences = []
    with open(input_file, 'r') as f:
        records = list(SeqIO.parse(f, 'fasta'))

    for line_num in line_numbers:
        if line_num - 1 < len(records):
            extracted_sequences.append(records[line_num - 1])

    # 将提取的序列写入输出文件
    SeqIO.write(extracted_sequences, output_fa, 'fasta')

