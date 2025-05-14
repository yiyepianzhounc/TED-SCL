import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import random
import argparse
from tqdm import tqdm

# 数据增强：打乱 token 顺序
def token_shuffle(text):
    words = text.split()
    if len(words) <= 1:
        return text
    idx = list(range(len(words)))
    random.shuffle(idx)
    shuffled = [words[i] for i in idx]
    return " ".join(shuffled)

# 特征层剪切：对 CLS 向量随机置 0
def feature_cutoff(embeddings, rate=0.1):
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    mask = torch.rand_like(embeddings) > rate
    return embeddings * mask

# 数据增强：替换为模糊表达（模拟）
def euphemism_replace(text):
    replacements = {"stupid": "not very bright", "hate": "dislike", "kill": "remove"}
    for word, euphemism in replacements.items():
        text = text.replace(word, euphemism)
    return text

# 构建对比学习数据集
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, aug_strategy="none"):
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
        self.sentences = [line.strip() for line in lines if line.strip()]
        self.tokenizer = tokenizer
        self.aug_strategy = aug_strategy

    # 根据指定策略增强句子
    def augment(self, text):
        if self.aug_strategy == "token_shuffle":
            return token_shuffle(text)
        elif self.aug_strategy == "euphemism_replace":
            return euphemism_replace(text)
        else:
            return text  # 无增强或其他策略非文本处理类

    def __getitem__(self, idx):
        s1 = self.sentences[idx]
        s2 = self.augment(s1)
        return s1, s2

    def __len__(self):
        return len(self.sentences)

# 构建对比学习模型
class ContrastiveModel(nn.Module):
    def __init__(self, model_name, proj_dim=128, dropout_rate=0.1,
                 use_embed_dropout=False, use_feature_cutoff=False, cutoff_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, proj_dim)
        )
        self.use_embed_dropout = use_embed_dropout
        self.use_feature_cutoff = use_feature_cutoff
        self.cutoff_rate = cutoff_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # 如果启用嵌入层 Dropout，则手动获取 embedding 并施加 dropout
        if self.use_embed_dropout and input_ids is not None:
            inputs_embeds = self.encoder.embeddings(input_ids)
            inputs_embeds = self.dropout(inputs_embeds)
            output = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0][:, 0]
        else:
            output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]

        # 如果启用特征层剪切
        if self.use_feature_cutoff:
            output = feature_cutoff(output, rate=self.cutoff_rate)

        z = self.projector(output)
        return z

# SimCSE 对比损失函数
def simcse_loss(z1, z2, temperature=0.05):
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    similarity = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    return nn.CrossEntropyLoss()(similarity, labels)

# 模型训练主函数
def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_set = TextDataset(args.train_path, tokenizer, aug_strategy=args.aug_strategy)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据增强策略判断是否启用 embedding dropout 或特征剪切
    use_embed_dropout = (args.aug_strategy == "embedding_dropout")
    use_feature_cutoff = (args.aug_strategy == "feature_cutoff")

    model = ContrastiveModel(
        args.model_name,
        proj_dim=args.proj_dim,
        dropout_rate=args.dropout,
        use_embed_dropout=use_embed_dropout,
        use_feature_cutoff=use_feature_cutoff,
        cutoff_rate=args.cutoff_rate
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for s1, s2 in tqdm(train_loader, desc=f"训练中 Epoch {epoch + 1}"):
            inputs1 = tokenizer(s1, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
            inputs2 = tokenizer(s2, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
            z1 = model(**inputs1)
            z2 = model(**inputs2)
            loss = simcse_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} 平均损失: {total_loss / len(train_loader):.4f}")

# 命令行参数解析入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True, help="训练文本文件路径")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="预训练模型名称")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--proj_dim", type=int, default=128, help="投影层维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")
    parser.add_argument("--cutoff_rate", type=float, default=0.1, help="feature_cutoff 的遮蔽比例")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--aug_strategy", choices=[
        "none", "token_shuffle", "feature_cutoff", "embedding_dropout", "euphemism_replace"
    ], default="none", help="选择数据增强策略")
    args = parser.parse_args()
    train(args)
