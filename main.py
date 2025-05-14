# coding:utf-8

from __future__ import absolute_import, division, unicode_literals
import random
import numpy as np
import pandas as pd
import os
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import logging
import sys
from datetime import datetime
from transformers import BertModel, BertTokenizer
import argparse
from gensim.models import KeyedVectors
from euphemism_detection.static_vector_Classifier_model_compare_by_sentences_embeddings import TextFNN,TextLSTM,TextLSTM_Att,TextBiLSTM,TextBiLSTM_Att
import re
import torch.nn.functional as F

"""
请增加描述
nohup python word_embedding_Classifier_model_compare.py --encoding_model_name roberta-base > logs/roberta-base_model_compare_20240718.log 2>&1 &

nohup python word_embedding_Classifier_model_compare.py --encoding_model_name roberta-large > logs/roberta-large_model_compare_20240718.log 2>&1 &

nohup python word_embedding_Classifier_model_compare.py --encoding_model_name bert-large > logs/bert-large_model_compare_20240718.log 2>&1 &

nohup python word_embedding_Classifier_model_compare.py --encoding_model_name bert-base > logs/bert-base_model_compare_20240718.log 2>&1 &
"""

MODEL_SAVE_PATH = "/data1/zg/euphemism_detection/model_save/trained_transformer_models/"

model_list = {
    'roberta-base':'/data1/zg/euphemism_detection/pretrained_model/chinese-roberta-wwm-ext-base',
    'roberta-large':'/data1/zg/euphemism_detection/pretrained_model/chinese-roberta-wwm-ext-large',
    'bert-base':'/data1/zg/euphemism_detection/pretrained_model/bert-base-chinese',
    'bert-large':'/data1/zg/euphemism_detection/pretrained_model/bert-large-chinese',
    'sbert':'/data1/zg/euphemism_detection/pretrained_model/sbert-chinese-general-v2',
    'consert-base':'/data1/zg/euphemism_detection/pretrained_model/consert-sequence/consert_NumEpoch-10_MaxSeqLength-48_BatchSize-96',
    'consert-large':'/data1/zg/euphemism_detection/pretrained_model/consert-roberta-large_batch32_epochs10_length128',
}

def parse_args():
    """Argument settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooling_type", type=str, choices=["cls","avg","none"], default="none", help="choose cls token or mean tokens")
    parser.add_argument("--encoding_model_name", type=str,choices=["bert-base","bert-large","roberta-base","roberta-large","sbert","consert-base","consert-large"], default="bert-base",  help="choose encoding model name")
    parser.add_argument("--device_name", type=str, choices=["cuda:0","cuda:1","cuda:2","cuda:3"], default="cuda:1", help="choose gpu device")
    parser.add_argument("--n_class", type=int, default= 2, help="Training mini-batch size")
    parser.add_argument("--random_seed", type=int, default= random.randint(1000,1200), help="Training mini-batch size")
    parser.add_argument("--batch_size", type=int, default= 64, help="Training mini-batch size")
    parser.add_argument("--max_epoch", type=int, default= 200, help="Number of max training epochs if model keeps improving")
    parser.add_argument("--learning_rate", type=float, default= 1e-3, help="The learning rate of model")
    parser.add_argument("--eval_epochs", type=int, default= 1, help="every each epochs to evaluate model performance")
    parser.add_argument("--early_stop_epochs", type=int, default= 50, help="Max training epoch that performance did not improve")
    parser.add_argument("--max_sequence_length", type=int, default= 96, help="Training max length of texts")
    parser.add_argument("--model_mode", type=str, choices=["train","test"], default='train', help="choose model to train or test")
    parser.add_argument("--log", type=str, choices=["yes","no"], default='no', help="choose to record model training history files")
    return parser.parse_args()


def sentences_encoding(encoding_model,input_ids, attention_mask, token_type_ids, pooling_type):
    with torch.no_grad():
        out_puts = encoding_model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

    last_hidden_state = out_puts[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min = 1e-9)
    mean_embeddings = sum_embeddings/sum_mask
    embeddings = mean_embeddings

    return embeddings

class FNN_for_bert(nn.Module):
    def __init__(self, embedding_dim):
        super(FNN_for_bert, self).__init__()
        # 加载预训练模型
        self.embedding_dim = embedding_dim
        self.hidden_dim = 16
        self.class_num = 2

        self.dropout1 = nn.Dropout(0.05) # dropout层
        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim) # 隐藏层
        self.BN1 = nn.BatchNorm1d(self.hidden_dim) # 归一化层
        self.relu1 = nn.ReLU(inplace=True) # 激活层
        self.dropout2 = nn.Dropout(0.05) # dropout层
        self.fc2 = nn.Linear(self.hidden_dim, self.class_num) # 输出层

    def forward(self, x):
        x = self.BN1(self.fc1(self.dropout1(x)))
        x = self.fc2(self.dropout2(self.relu1(x)))
        return x


class TED_SCL(nn.Module):
    def __init__(self, embedding_dim):
        super(TED_SCL, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim1 = 64
        self.hidden_dim2 = 16
        self.n_class = 2

        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_dim1, num_layers=2,
                             bidirectional=False, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(self.embedding_dim, self.hidden_dim1, num_layers=2,
                             bidirectional=False, batch_first=True, dropout=0.5)

        self.fc1 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)

        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.hidden_dim2 * 2))

        self.fc_out1 = nn.Linear(self.hidden_dim2 * 2 + 3, self.hidden_dim2)
        self.BN1 = nn.BatchNorm1d(self.hidden_dim2)
        self.dropout1 = nn.Dropout(0.05)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc_out2 = nn.Linear(self.hidden_dim2, self.n_class)
        self.BN2 = nn.BatchNorm1d(self.n_class)
        self.dropout2 = nn.Dropout(0.05)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, sentences1, sentences2, senti):  

        x_left = torch.unsqueeze(sentences1, 1)
        x_right = torch.unsqueeze(sentences2, 1)

        out1, _ = self.lstm1(x_left)
        out1 = self.fc1(out1[:, -1, :])  

        out2, _ = self.lstm2(x_right)
        out2 = self.fc2(out2[:, -1, :])  
        
        out_H = torch.cat((out1, out2), dim=1)
        out_H = torch.unsqueeze(out_H, 1)
        M = self.tanh(out_H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = out_H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)

        x_out = torch.cat((out, senti), dim = 1)
        x_out = self.fc_out1(x_out)
        x_out = self.relu1(self.dropout1(self.BN1(x_out)))
        x_out = self.fc_out2(x_out)
        x_out = self.relu2(self.dropout2(self.BN2(x_out)))

        return x_out

def logger_init(log_file_name='monitor', log_level=logging.DEBUG, log_dir='./logs/', only_file=False):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'

    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level=log_level, format=formatter, datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])


class MyDataset(Dataset):
    def __init__(self, X, Y):

        self.sentence = X.values
        self.label = Y.values


    def __getitem__(self, idx):

        sentence = self.sentence[idx]
        label = self.label[idx]
        return str(sentence), label

    def __len__(self):
        return len(self.sentence)

class MyDatasetNew(Dataset):
    def __init__(self, X, Y):

        self.sentences1 = X['content'].values
        self.euphemisms = X['euphemism'].values
        self.label = Y.values
        # 载入委婉词映射表
        self.eu_pairs_path = '/data1/zg/euphemism_detection/corpus/eu-senti-datasets/eu-pairs.csv'
        self.eu_pairs = self.load_euphemism_knowledge_dict(self.eu_pairs_path)
        self.sentences2 = self.euphemism_backgourd_konwledge_fusion(self.sentences1, self.euphemisms, self.label)
        self.senti = np.array(X.loc[:,['positive','negative','neutral']])

    def __getitem__(self, idx):

        sentence1 = self.sentences1[idx]
        sentence2 = self.sentences2[idx]
        senti = self.senti[idx]
        label = self.label[idx]
        return str(sentence1), str(sentence2), senti, label

    def load_euphemism_knowledge_dict(self, path):
        df = pd.read_csv(path,encoding='gbk')
        eu_diction_nary = {}
        for index in df.index:
            euphemism = df['euphemism'][index]
            mean = df['mean'][index]
            eu_diction_nary[euphemism] = mean
        return eu_diction_nary

    def euphemism_backgourd_konwledge_fusion(self, sentences,euphemisms,labels):
        for idx in range(0, len(sentences)):
            sentence = sentences[idx]
            euphemism = euphemisms[idx]
            # label = labels[idx]
            if euphemism != '无':
                mean = self.eu_pairs[euphemism]
                # print(mean, euphemism)
                sentences[idx] = str(sentence).replace(euphemism, mean)
            else:
                pass
        return sentences

    def __len__(self):
        return len(self.sentences1)

class PyTorchClassifier(object):
    def __init__(self, args):
        # 模型搭建
        super(PyTorchClassifier, self).__init__()
        seed = args.random_seed
        # 设置种子值大小
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # 设定gpu设备
        self.device = torch.device(args.device_name if torch.cuda.is_available() else "cpu")
        # 载入bert或者其他的预训练模型
        self.encoding_model_name = args.encoding_model_name
        self.pooling_type = args.pooling_type

        if self.encoding_model_name in ['bert-base','roberta-base','consert-base','sbert']:
            self.embedding_dim = 768
        elif self.encoding_model_name in ['bert-large','roberta-large','consert-large']:
            self.embedding_dim = 1024
        else:
            pass

        if self.encoding_model_name in ['bert-base','bert-large','roberta-base','roberta-large']:
            self.pooling_type = 'avg'
            self.vocab_file = model_list[self.encoding_model_name]+'/vocab.txt'
            self.encoding_model = BertModel.from_pretrained(model_list[self.encoding_model_name], output_hidden_states=True)
            self.tokenizer = BertTokenizer(self.vocab_file)
            # 载入分类器模型
            self.classifier_model= FNN_for_bert(self.embedding_dim)

            # 冻结预训练模型的参数
            for param in self.encoding_model.parameters():
                param.requires_grad_(False)

            # 模型移动到gpu上进行加速运算
            self.classifier_model.to(self.device)
            self.encoding_model.to(self.device)

        elif self.encoding_model_name in ['sbert','consert-base','consert-large']:
            self.pooling_type == 'none'
            encoding_model_name = model_list[self.encoding_model_name]
            self.encoding_model = SentenceTransformer(encoding_model_name)
            # 载入分类器模型
            if self.encoding_model_name in ["sbert"]:
                self.classifier_model = FNN_for_bert(self.embedding_dim)

            elif self.encoding_model_name in ["consert-base","consert-large"]:
                self.classifier_model = TED_SCL(self.embedding_dim)

            # 模型移动到gpu上进行加速运算
            self.classifier_model.to(self.device)
            self.encoding_model.to(self.device)

            # 冻结预训练模型的参数
            for param in self.encoding_model.parameters():
                param.requires_grad_(False)

        # 分类数量
        self.n_class = args.n_class
        # 学习率
        self.learning_rate = args.learning_rate  # 1e-4效果较差，收敛速度很慢,1e-3较好
        # 训练的batch大小
        self.batch_size = args.batch_size
        # 训练的epochs数量
        self.max_epoch = args.max_epoch
        # 多少个epoch评估一次
        self.eval_epoch = args.eval_epochs
        # 多少个epoch以后准确率没有提高，就停止训练
        self.early_stop_epochs = args.early_stop_epochs
        # 最长序列的长度
        self.max_sequence_length = args.max_sequence_length
        # 保存模型的名称
        time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.file_name = self.encoding_model_name + '_' + self.pooling_type + '_' + str(time_stamp)
        # 保存模型的路径
        self.model_save_path = MODEL_SAVE_PATH + self.file_name + ".h5"
        # 优化函数
        # self.opt_fn = optim.SGD(self.model.parameters(), lr=self.l2reg, momentum=0.001, dampening=0.001)
        self.opt_fn = optim.Adam(self.classifier_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        # 模型训练记录器
        logger_init(self.file_name, log_level=logging.INFO, log_dir='./logs')
        logging.info(f"this is word embedding classifier model compare, and now model name is {self.encoding_model_name}")

    # 掩码函数，用于bert和roberta模型
    def collate_fn(self, data):
        sents = [i[0] for i in data]
        labels = [i[1] for i in data]
        #编码
        data = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=self.max_sequence_length,
                                    return_tensors='pt',
                                    return_length=True)
        #input_ids:编码之后的数字
        #attention_mask:是补零的位置是0,其他位置是1
        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        #print(data['length'], data['length'].max())
        return input_ids, attention_mask, token_type_ids, labels

    def prepare_datasets(self):

        train_dataset = pd.read_csv('/data1/zg/euphemism_detection/corpus/eu-senti-datasets/train.csv', encoding='gbk').loc[:,['content','toxic']]
        eval_dataset = pd.read_csv('/data1/zg/euphemism_detection/corpus/eu-senti-datasets/eval.csv', encoding='gbk').loc[:,['content','toxic']]
        test_dataset = pd.read_csv('/data1/zg/euphemism_detection/corpus/eu-senti-datasets/test.csv', encoding='gbk').loc[:,['content','toxic']]

        train_dataset = MyDataset(train_dataset.loc[:,'content'],train_dataset.loc[:,'toxic'])
        eval_dataset = MyDataset(eval_dataset.loc[:,'content'],eval_dataset.loc[:,'toxic'])
        test_dataset = MyDataset(test_dataset.loc[:,'content'],test_dataset.loc[:,'toxic'])

        if self.encoding_model_name in ['bert-base','bert-large','roberta-base','roberta-large']:
            train_loader = DataLoader(dataset=train_dataset, batch_size= self.batch_size, shuffle=True, collate_fn=self.collate_fn)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size= self.batch_size , shuffle=True, collate_fn=self.collate_fn)
            test_loader = DataLoader(dataset=test_dataset, batch_size= self.batch_size , shuffle=True, collate_fn=self.collate_fn)

        elif self.encoding_model_name in ['sbert']:
            train_loader = DataLoader(dataset=train_dataset, batch_size= self.batch_size, shuffle=True)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size= self.batch_size , shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size= self.batch_size , shuffle=True)

        elif self.encoding_model_name in ['consert-base','consert-large']:
            train_dataset = pd.read_csv('/data1/zg/euphemism_detection/corpus/eu-senti-datasets/train.csv', encoding='gbk').loc[:,['content','euphemism','toxic','positive','negative','neutral']]
            eval_dataset = pd.read_csv('/data1/zg/euphemism_detection/corpus/eu-senti-datasets/eval.csv', encoding='gbk').loc[:,['content','euphemism','toxic','positive','negative','neutral']]
            test_dataset = pd.read_csv('/data1/zg/euphemism_detection/corpus/eu-senti-datasets/test.csv', encoding='gbk').loc[:,['content','euphemism','toxic','positive','negative','neutral']]

            train_dataset = MyDatasetNew(train_dataset.loc[:,['content','euphemism','positive','negative','neutral']], train_dataset.loc[:,'toxic'])
            eval_dataset = MyDatasetNew(eval_dataset.loc[:,['content','euphemism','positive','negative','neutral']], eval_dataset.loc[:,'toxic'])
            test_dataset = MyDatasetNew(test_dataset.loc[:,['content','euphemism','positive','negative','neutral']], test_dataset.loc[:,'toxic'])

            train_loader = DataLoader(dataset=train_dataset, batch_size= int(self.batch_size), shuffle=True)
            eval_loader = DataLoader(dataset=eval_dataset, batch_size=len(eval_dataset), shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
        
        return train_loader, eval_loader, test_loader

    def fit(self):
        # 设置训练相关参数
        total_epoch = 0  # 记录进行到多少epch
        eval_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的epoch数
        # flag = False  # 记录是否很久没有效果提升

        # 用于画图的epoch、val_los,val_acc参数
        epoch_list = []
        eval_loss_list = []
        eval_acc_list = []

        writer = SummaryWriter(log_dir='train_logs/' + time.strftime('%m-%d_%H.%M', time.localtime()))

        # 准备数据集
        train_loader, eval_loader, test_loader = self.prepare_datasets()
        print("## successfully load  model,start train......")
        # self.model.to(self.device)
        # torch.backends.cudnn.enabled = False
        for e in range(self.max_epoch):
            self.classifier_model.train()
            # 训练1个epoch
            if self.encoding_model_name in ['bert-base','bert-large','roberta-base','roberta-large']: 
                for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
                    x_embeddings = sentences_encoding(self.encoding_model, input_ids, attention_mask, token_type_ids, self.pooling_type)
                    out = self.classifier_model(x_embeddings)
                    loss = self.loss_fn(out, labels)
                    loss.backward()
                    self.opt_fn.step()
                    self.opt_fn.zero_grad()

            elif self.encoding_model_name in ['sbert']:
                for i, (sentences, labels) in enumerate(train_loader):
                    sentences = self.encoding_model.encode(sentences, show_progress_bar=False)
                    labels = labels.to(self.device)
                    sentences = torch.from_numpy(np.array(sentences)).to(self.device)
                    out = self.classifier_model(sentences) 
                    loss = self.loss_fn(out, labels)
                    loss.backward()
                    self.opt_fn.step()
                    self.opt_fn.zero_grad()

            elif self.encoding_model_name in ['consert-base','consert-large']:
                for i, (sentences1, sentences2, senti, labels) in enumerate(train_loader):
                    labels = labels.to(self.device)
                    sentences1 = self.encoding_model.encode(sentences1, show_progress_bar=False)
                    sentences2 = self.encoding_model.encode(sentences2, show_progress_bar=False)
                    sentences1 = torch.from_numpy(np.array(sentences1)).to(self.device)
                    sentences2 = torch.from_numpy(np.array(sentences2)).to(self.device)
                    senti = torch.from_numpy(np.array(senti)).float().to(self.device)
                    out = self.classifier_model(sentences1, sentences2, senti)  # [batch_size,num_class]

                    loss = self.loss_fn(out, labels)
                    loss.backward()
                    self.opt_fn.step()
                    self.opt_fn.zero_grad()

            # epoch训练完成后进行评估
            # self.classifier_model.eval()
            if e % self.eval_epoch == 0:
                print('epoch:', str(e + 1) + '/' + str(self.max_epoch))
                # 每多少轮输出在训练集和验证集上的效果
                train_acc, _, _, _, _, _, _, train_loss  = self.evaluate(train_loader)
                eval_acc, _, _, _, _, _, _, eval_loss = self.evaluate(eval_loader)
                if e not in epoch_list:
                    # 画图
                    epoch_list.append(e)
                    eval_loss_list.append(eval_loss)
                    eval_acc_list.append(eval_acc)

                if eval_loss < eval_best_loss:
                    eval_best_loss = eval_loss
                    improve = '*'
                    last_improve = total_epoch
                    torch.save(self.classifier_model.state_dict(), self.model_save_path)
                    self.test()
                else:
                    improve = ''

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                time_dif = str(time.localtime())
                log = msg.format(total_epoch, train_loss, train_acc, eval_loss, eval_acc, time_dif, improve)
                print(log)
                writer.add_scalar("loss/train", train_loss, total_epoch)
                writer.add_scalar("loss/dev", eval_loss, total_epoch)
                writer.add_scalar("acc/train", train_acc, total_epoch)
                writer.add_scalar("acc/dev", eval_acc, total_epoch)

            # 记录总体的epoch个数
            total_epoch += 1
            if total_epoch - last_improve > self.early_stop_epochs:
                # 验证集loss超过最大epoch数量后没有下降，结束训练
                print("more than loss" + str(self.early_stop_epochs) + "epoch not get lower loss.finish training")
                break
   
        # 关闭writer函数
        writer.close()
        # 模型测试
        print('*******the training process of model is finished, and this is the final result of the model detection**********')
        self.test()

    def evaluate(self, eval_loader):
        self.classifier_model.eval()
        acc_all = 0
        positive_precision_all = 0
        positive_recall_all = 0
        positive_F1_all = 0
        negative_precision_all = 0
        negative_recall_all = 0
        negative_F1_all = 0
        loss_all = 0
        max_epoch = 0
        if self.encoding_model_name in ['bert-base','bert-large','roberta-base','roberta-large']: 
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(eval_loader):
                max_epoch = max_epoch + 1
                x_embeddings = sentences_encoding(self.encoding_model, input_ids, attention_mask, token_type_ids, self.pooling_type)
                logits = self.classifier_model(x_embeddings)
                loss = self.loss_fn(logits, labels)
                pred = logits.argmax(dim=1)
                labels_cpu = labels.cpu()
                pred_cpu = pred.cpu()
                acc,positive_precision,positive_recall,positive_F1,negative_precision,negative_recall,negative_F1 = self.calculate(pred_cpu, labels_cpu)
                acc_all = acc_all + acc
                positive_precision_all = positive_precision_all + positive_precision
                positive_recall_all = positive_recall_all + positive_recall
                positive_F1_all = positive_F1_all + positive_F1
                negative_precision_all = negative_precision_all + negative_precision
                negative_recall_all = negative_recall_all + negative_recall
                negative_F1_all = negative_F1_all + negative_F1
                loss_all = loss_all + loss.item()

        elif self.encoding_model_name in ['sbert']:
            for i, (sentences, labels) in enumerate(eval_loader):
                max_epoch = max_epoch + 1
                sentences = self.encoding_model.encode(sentences, show_progress_bar=False)
                sentences = torch.from_numpy(np.array(sentences)).to(self.device)
                labels = labels.to(self.device)
                logits = self.classifier_model(sentences)
                loss = self.loss_fn(logits, labels)
                pred = logits.argmax(dim=1)
                labels_cpu = labels.cpu()
                pred_cpu = pred.cpu()
                acc,positive_precision,positive_recall,positive_F1,negative_precision,negative_recall,negative_F1 = self.calculate(pred_cpu, labels_cpu)
                acc_all = acc_all + acc
                positive_precision_all = positive_precision_all + positive_precision
                positive_recall_all = positive_recall_all + positive_recall
                positive_F1_all = positive_F1_all + positive_F1
                negative_precision_all = negative_precision_all + negative_precision
                negative_recall_all = negative_recall_all + negative_recall
                negative_F1_all = negative_F1_all + negative_F1
                loss_all = loss_all + loss.item()

        elif self.encoding_model_name in ['consert-base','consert-large']:
            for i, (sentences1, sentences2, senti, labels) in enumerate(eval_loader):
                max_epoch = max_epoch + 1
                labels = labels.to(self.device)
                sentences1 = self.encoding_model.encode(sentences1,show_progress_bar=False)
                sentences2 = self.encoding_model.encode(sentences2,show_progress_bar=False)
                sentences1 = torch.from_numpy(np.array(sentences1)).to(self.device)
                sentences2 = torch.from_numpy(np.array(sentences2)).to(self.device)
                senti = torch.from_numpy(np.array(senti)).float().to(self.device)
                logits = self.classifier_model(sentences1, sentences2, senti)
                loss = self.loss_fn(logits, labels)
                pred = logits.argmax(dim=1)
                labels_cpu = labels.cpu()
                pred_cpu = pred.cpu()
                acc,positive_precision,positive_recall,positive_F1,negative_precision,negative_recall,negative_F1 = self.calculate(pred_cpu, labels_cpu)
                acc_all = acc_all + acc
                positive_precision_all = positive_precision_all + positive_precision
                positive_recall_all = positive_recall_all + positive_recall
                positive_F1_all = positive_F1_all + positive_F1
                negative_precision_all = negative_precision_all + negative_precision
                negative_recall_all = negative_recall_all + negative_recall
                negative_F1_all = negative_F1_all + negative_F1
                loss_all = loss_all + loss.item()

        # 计算最终指标
        final_acc = round(acc_all/max_epoch,5)
        final_positive_precision = round(positive_precision_all/max_epoch,5)
        final_positive_recall = round(positive_recall_all/max_epoch,5)
        final_positive_F1 = round(positive_F1_all/max_epoch,5)
        final_negative_precision = round(negative_precision_all/max_epoch,5)
        final_negative_recall = round(negative_recall_all/max_epoch,5)
        final_negative_F1 = round(negative_F1_all/max_epoch,5)
        final_loss = round(loss_all/max_epoch,5)
        return final_acc, final_positive_precision, final_positive_recall, final_positive_F1, final_negative_precision,final_negative_recall,final_negative_F1, final_loss

    def calculate(self, pred, labels):
        FP = 0
        TP = 0
        FN = 0
        TN = 0
        i = 0
        for label in labels:
            if label == 0 and pred[i] == 0:
                TN = TN + 1
            elif label == 0 and pred[i] == 1:
                FP = FP + 1
            elif label == 1 and pred[i] == 0:
                FN = FN + 1
            elif label == 1 and pred[i] == 1:
                TP = TP + 1
            i = i + 1

        try: 
            acc = (TN + TP) / (FP + TP + FN + TN)
        except:
            acc = 0

        try:
            positive_precision = TP / (TP + FP)
            positive_recall = TP / (TP + FN)
            positive_F1 = 2 * positive_precision * positive_recall / (positive_recall + positive_precision )
        except: 
            positive_precision = 0
            positive_recall = 0
            positive_F1 = 0

        try:
            negative_precision = TN / (TN + FN)
            negative_recall = TN / (TN + FP)
            negative_F1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall )
        except:
            negative_precision = 0
            negative_recall = 0
            negative_F1 = 0

        return acc,positive_precision,positive_recall,positive_F1,negative_precision,negative_recall,negative_F1

    def test(self):
        # 准备数据集
        _, _, test_loader = self.prepare_datasets()
        # logging.info("## successfully model trained, start test the final model")
        print(("## successfully model trained, start test the final model"))
        max_epoch = 0
        acc_all = 0
        positive_precision_all = 0
        positive_recall_all = 0
        positive_F1_all = 0
        negative_precision_all = 0
        negative_recall_all = 0
        negative_F1_all = 0

        # 载入效果最好的模型保存节点
        state_dict = torch.load(self.model_save_path)
        self.classifier_model.load_state_dict(state_dict)
        # 固定模型，进行测试
        self.classifier_model.eval()

        if self.encoding_model_name in ['bert-base','bert-large','roberta-base','roberta-large']: 
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
                max_epoch = max_epoch + 1
                x_embeddings = sentences_encoding(self.encoding_model, input_ids, attention_mask, token_type_ids, self.pooling_type)
                logits = self.classifier_model(x_embeddings)
                pred = logits.argmax(dim=1)
                labels_cpu = labels.cpu()
                pred_cpu = pred.cpu()
                acc,positive_precision,positive_recall,positive_F1,negative_precision,negative_recall,negative_F1 = self.calculate(pred_cpu, labels_cpu)
                if positive_F1 != 0 and negative_F1 != 0:
                    acc_all = acc_all + acc
                    positive_precision_all = positive_precision_all + positive_precision
                    positive_recall_all = positive_recall_all + positive_recall
                    positive_F1_all = positive_F1_all + positive_F1
                    negative_precision_all = negative_precision_all + negative_precision
                    negative_recall_all = negative_recall_all + negative_recall
                    negative_F1_all = negative_F1_all + negative_F1

        elif self.encoding_model_name in ['sbert']:
            for i, (sentences, labels) in enumerate(test_loader):
                max_epoch = max_epoch + 1
                labels = labels.to(self.device)
                sentences = self.encoding_model.encode(sentences, show_progress_bar=False)
                sentences = torch.from_numpy(np.array(sentences)).to(self.device)
                labels = labels
                logits = self.classifier_model(sentences)
                pred = logits.argmax(dim=1)

                labels_cpu = labels.cpu()
                pred_cpu = pred.cpu()
                acc,positive_precision,positive_recall,positive_F1,negative_precision,negative_recall,negative_F1 = self.calculate(pred_cpu, labels_cpu)

                if positive_F1 != 0 and negative_F1 != 0:
                    acc_all = acc_all + acc
                    positive_precision_all = positive_precision_all + positive_precision
                    positive_recall_all = positive_recall_all + positive_recall
                    positive_F1_all = positive_F1_all + positive_F1
                    negative_precision_all = negative_precision_all + negative_precision
                    negative_recall_all = negative_recall_all + negative_recall
                    negative_F1_all = negative_F1_all + negative_F1

        elif self.encoding_model_name in ['consert-base','consert-large']:
            for i, (sentences1, sentences2, senti, labels) in enumerate(test_loader):
                max_epoch = max_epoch + 1
                labels = labels.to(self.device)
                sentences1 = self.encoding_model.encode(sentences1,show_progress_bar=False)
                sentences2 = self.encoding_model.encode(sentences2,show_progress_bar=False)
                sentences1 = torch.from_numpy(np.array(sentences1)).to(self.device)
                sentences2 = torch.from_numpy(np.array(sentences2)).to(self.device)
                senti = torch.from_numpy(np.array(senti)).float().to(self.device)
                labels = labels
                logits = self.classifier_model(sentences1, sentences2, senti)
                pred = logits.argmax(dim=1)
                pred_cpu = pred.cpu()
                labels_cpu = labels.cpu()
                acc,positive_precision,positive_recall,positive_F1,negative_precision,negative_recall,negative_F1 = self.calculate(pred_cpu, labels_cpu)

                if positive_F1 != 0 and negative_F1 != 0:
                    acc_all = acc_all + acc
                    positive_precision_all = positive_precision_all + positive_precision
                    positive_recall_all = positive_recall_all + positive_recall
                    positive_F1_all = positive_F1_all + positive_F1
                    negative_precision_all = negative_precision_all + negative_precision
                    negative_recall_all = negative_recall_all + negative_recall
                    negative_F1_all = negative_F1_all + negative_F1

        print(f"acc is{round((acc_all/max_epoch) * 100, 2)}")
        print(f"positive_precision is{round( (positive_precision_all/max_epoch)* 100, 2)}")
        print(f"positive_recall is{round((positive_recall_all/max_epoch) * 100, 2)}")
        print(f"positive_F1 is{round((positive_F1_all/max_epoch) * 100, 2)}")
        print(f"negative_precision is{round((negative_precision_all/max_epoch) * 100, 2)}")
        print(f"negative_recall is{round( (negative_recall_all/max_epoch)* 100, 2)}")
        print(f"negative_F1 is{round((negative_F1_all/max_epoch) * 100, 2)}")

    def choose_model_path(self, model_name):
        import os
        file_names = os.listdir(MODEL_SAVE_PATH)
        target_list = []
        # "model_name + cls_type + time_stamp"
        for file_name in file_names:
            # print(file_name)
            file_name_fixed = str(file_name).split("_")[2].replace(".h5","")
            file_name_pre = str(file_name).split("_")[0]
            if file_name_pre == model_name:
                target_list.append(int(file_name_fixed))

        target = max(target_list)
        target_file_name = str(model_name) + "_" + str(self.pooling_type) + "_" + str(target) + ".h5"
        target_file_path = MODEL_SAVE_PATH + target_file_name

        return str(target_file_path)


def main(args):
    pytorch_classifier = PyTorchClassifier(args)
    print("batch_size:{}".format(args.batch_size))
    print("max_epoch:{}".format(args.max_epoch))
    print("learning_rate:{}".format(args.learning_rate))
    print("max_sequence_length:{}".format(args.max_sequence_length))
    print("encoding_model_name:{}".format(args.encoding_model_name))
    print("random_seed:{}".format(args.random_seed))
    if args.model_mode == 'train':
        pytorch_classifier.fit()
    elif args.model_mode == 'test':
        pytorch_classifier.model_save_path = pytorch_classifier.choose_model_path(args.encoding_model_name)


if __name__ == '__main__':
    outputs = {}
    args = parse_args()
    main(args)
    # contribute
