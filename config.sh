#!/usr/bin/env bash
# config.sh ——————————————————————————————————————————————————————
# 列出要跑的模型名称（与 main.py 中 --encoding_model_name 对应）
MODELS=(
  "bert-base"
  "bert-large"
  "roberta-base"
  "roberta-large"
  "sbert"
  "consert-base"
  "consert-large"
)

# 公共参数
DEVICE="cuda:0"
BATCH_SIZE=64
LR=1e-3
MAX_SEQ_LEN=96
MAX_EPOCH=100
EVAL_EPOCHS=1
EARLY_STOP=20
POOL_TYPE="none"
N_CLASS=2
LOG="no"

# 多轮训练设置
REPEAT=5
# 随机种子列表（长度 ≥ REPEAT，若少则循环使用最后一个）
SEEDS=(42 43 44 45 46)
# 默认训练模式：train 或 test
MODE="train"
# ———————————————————————————————————————————————————————————
