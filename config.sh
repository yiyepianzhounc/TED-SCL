#!/usr/bin/env bash
# config.sh — 更新版，用于配合 run.sh

# —— 模型列表 —— 
# 与 main.py 中 --encoding_model_name 参数对应
MODELS=(
  "bert-base"
  "bert-large"
  "roberta-base"
  "roberta-large"
  "sbert"
  "consert-base"
  "consert-large"
)

# —— 公共超参数 —— 
DEVICE="cuda:0"            # GPU 设备
BATCH_SIZE=64              # 批大小
LR=1e-3                    # 学习率
MAX_SEQ_LEN=96             # 最大序列长度
MAX_EPOCH=100              # 最大训练轮数
EVAL_EPOCHS=1              # 每多少轮评估一次
EARLY_STOP=20              # 验证不升的早停轮数
POOL_TYPE="none"           # pooling 策略
N_CLASS=2                  # 分类数
LOG="no"                   # 是否记录训练日志 (yes/no)

# —— 多轮训练 & 随机种子 —— 
REPEAT=5                   # 每个模型重复训练次数
SEEDS=(42 43 44 45 46)     # 随机种子列表，长度 ≥ REPEAT，短则循环使用最后一个

# —— 训练模式 —— 
# 模式(train/test)，train 会训练并在 eval 上评估，test 只用已保存模型做一次测试
MODE="train"
