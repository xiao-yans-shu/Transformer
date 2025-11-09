#!/usr/bin/env bash
# 示例流程：训练模型 -> 翻译样例 -> 评估BLEU

set -e

RUN_NAME="baseline_run"
CHECKPOINT_DIR="runs/${RUN_NAME}/checkpoints"
CHECKPOINT_BEST_BLEU="${CHECKPOINT_DIR}/best_model_bleu.pt"

if [ ! -d "de-en" ]; then
  echo "[Error] 请先将 IWSLT17 英德数据放入 de-en/ 目录"
  exit 1
fi

# 1. 训练
python train.py --run_name "$RUN_NAME"

# 2. 翻译示例（使用生成的检查点）
python translate.py

# 3. 评估（修改 evaluate_dataset.py 中的 CHECKPOINT_PATH 即可）
python evaluate_dataset.py

# 4. 提示输出路径
echo "\n训练结果目录: runs/${RUN_NAME}/"
ls "runs/${RUN_NAME}"
