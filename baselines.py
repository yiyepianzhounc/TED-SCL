# 导入依赖
from __future__ import absolute_import, division, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# ---------------- 通用模块 ----------------
# 功能: 多层感知机网络
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.05):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 功能: 基于 LSTM 的编码模块
class BaseLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return out[:, -1, :]

# 功能: 注意力池化模块
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, H):
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        return torch.sum(H * alpha, dim=1)

# ---------------- 神经网络模型 ----------------
# 功能: 多层感知机分类器
class TextFNN(MLP):
    def __init__(self):
        super().__init__(input_dim=128, hidden_dims=[64, 16, 8], output_dim=2, dropout=0.05)

# 功能: 单向 LSTM 分类器
class TextLSTM(BaseLSTM):
    def __init__(self):
        super().__init__(embed_dim=128, hidden_size=32, num_layers=1, bidirectional=False, dropout=0.05)
        self.fc = nn.Linear(self.out_dim, 2)

    def forward(self, x):
        h = super().forward(x)
        return self.fc(h)

# 功能: 单向 LSTM + 注意力 分类器
class TextLSTM_Att(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BaseLSTM(128, 32, num_layers=1, bidirectional=False, dropout=0.05)
        self.att = Attention(32)
        self.fc = nn.Sequential(nn.Linear(32, 2), nn.ReLU(inplace=True))

    def forward(self, x):
        x = x.unsqueeze(1)
        H, _ = self.encoder.lstm(x)
        a = self.att(H)
        return self.fc(a)

# 功能: 双向 LSTM 分类器
class TextBiLSTM(BaseLSTM):
    def __init__(self):
        super().__init__(embed_dim=128, hidden_size=64, num_layers=2, bidirectional=True, dropout=0.05)
        self.fc = nn.Sequential(nn.Linear(self.out_dim, 2), nn.ReLU(inplace=True))

    def forward(self, x):
        h = super().forward(x)
        return self.fc(h)

# 功能: 双向 LSTM + 注意力 分类器
class TextBiLSTM_Att(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BaseLSTM(128, 64, num_layers=2, bidirectional=True, dropout=0.05)
        self.att = Attention(self.encoder.out_dim)
        self.fc = nn.Sequential(nn.Linear(self.encoder.out_dim, 2), nn.ReLU(inplace=True))

    def forward(self, x):
        x = x.unsqueeze(1)
        H, _ = self.encoder.lstm(x)
        a = self.att(H)
        return self.fc(a)

# 功能: TextCNN 分类器
class TextCNN(nn.Module):
    def __init__(self, embed_dim=128, num_filters=50, kernel_sizes=[2, 3], num_classes=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        xs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        xs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in xs]
        x = torch.cat(xs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ---------------- 传统机器学习模型 ----------------
# 功能: 随机森林分类器
class RFClassifier:
    def __init__(self, n_estimators=100, random_state=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# 功能: 支持向量机分类器
class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, random_state=None):
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# 功能: AdaBoost 分类器
class AdaBoostClassifierSK:
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
