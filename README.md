# 自实现 Transformer 英德翻译

本项目在 PyTorch 中从零实现了完整的 Transformer 模型，支持训练、推理、评估与消融实验，可在 IWSLT2017 英语→德语数据集上运行。核心模块包括多头注意力、前馈网络、相对位置编码、学习率调度器、早停机制、日志与曲线记录等。

## 目录结构
```
Transformer/
├── attention.py              # 多头注意力（含相对位置版本）
├── config.py                 # 全局配置，含数据规模、超参数开关
├── data_loader.py            # 数据预处理与 DataLoader
├── decoder.py                # Decoder 结构
├── encoder.py                # Encoder 结构
├── evaluate.py               # 验证集评估 + BLEU（自定义实现）
├── evaluate_dataset.py       # 使用 sacreBLEU 的一键评估脚本
├── logger.py                 # TeeLogger，实现控制台与文件双写
├── position_encoding.py      # 正弦与相对位置编码
├── train.py                  # 训练入口脚本
├── transformer.py            # Transformer 封装（Encoder + Decoder）
├── translate.py              # 单句或文件翻译脚本
├── requirements.txt          # Python 依赖
└── scripts/
    └── run.sh                # 训练 + 评估示例流程
```

## 环境准备
```bash
conda create -n transformer python=3.10
conda activate transformer
pip install -r requirements.txt
```

## 数据准备
将 IWSLT2017 英德数据放在 `de-en/` 目录下（包含 `train.tags.de-en.*`、`IWSLT17.TED.dev2010.de-en.*.xml` 以及 `IWSLT17.TED.tst201*.de-en.*.xml` 等文件）。若更换数据集，需要在 `config.py` 中更新对应路径。

## 使用说明
### 训练模型
```bash
python train.py --run_name baseline_200k
```
训练输出会保存到 `runs/<run_name>/`，其中包含：
- `logs/training_log_*.txt`
- `figures/training_curves*.png`
- `checkpoints/best_model.pt` 与 `best_model_bleu.pt`

常用参数：
- `--max_train_samples`：截取训练样本数量（默认 200000）
- `--vocab_size`：词表大小（默认 20000）
- `--disable_pos_encoding`：禁用绝对位置编码
- `--dropout`、`--lr`、`--patience` 等均可在命令行覆盖

### 翻译与推断
编辑 `translate.py` 顶部的 `SENTENCES_TO_TRANSLATE` 或 `INPUT_FILE/OUTPUT_FILE`，然后运行：
```bash
python translate.py
```
脚本会载入 `CHECKPOINT_PATH` 指向的模型，生成翻译并打印/写入文件。

### 验证模型效果
在 `evaluate_dataset.py` 中设置 `CHECKPOINT_PATH`、`USE_POS_ENCODING`、`BEAM_SEARCH` 等参数，然后执行：
```bash
python evaluate_dataset.py
```
脚本会：
1. 从 XML 解析指定数据集；
2. 用模型生成翻译；
3. 通过 sacreBLEU （大小写敏感/不敏感）计算 BLEU；
4. 将源句、参考、输出与结果写入 `experiments/manual_eval/<dataset>/`。

## 主要特性
- 手工实现的 Transformer Encoder/Decoder、相对位置编码、多头注意力
- AdamW + 学习率 Warmup 调度 + 梯度裁剪 + 早停机制
- 训练过程中自动保存日志、曲线、模型权重
- 单句/批量翻译、全量评估脚本（含 sacreBLEU）
- 训练与推理均支持禁用绝对位置编码的消融实验

## 未来改进方向
- 引入 BPE / SentencePiece 子词分词，缓解 OOV 问题
- 更多相对位置编码与优化策略的实验
- 结合外部预训练模型或官方实现进行对比

## 关于 runs 和 experiments 目录
- `runs/`：包含所有训练产物（模型权重、日志、曲线），并提供包含消融实验在内的模型权重下载链接。
- `experiments/`：存放各类实验（含消融）导出的训练日志和损失曲线；尤其注意训练过程中内置的评估逻辑存在问题，当时得到的 BLEU 结果不可靠，因此我们在训练结束后重新加载模型权重、选取全新数据批次完成评估，并以该 BLEU 作为最终报告值。

