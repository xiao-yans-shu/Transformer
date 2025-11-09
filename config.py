"""
配置文件 - Transformer模型训练参数
"""
import torch

class Config:
    # 数据路径
    DATA_DIR = "de-en"
    TRAIN_EN = "de-en/train.tags.de-en.en"
    TRAIN_DE = "de-en/train.tags.de-en.de"
    DEV_EN = "de-en/IWSLT17.TED.dev2010.de-en.en.xml"
    DEV_DE = "de-en/IWSLT17.TED.dev2010.de-en.de.xml"
    
    # 模型参数
    VOCAB_SIZE = 20000  # 词汇表大小（从10000增加到20000，减少OOV）
    EMBED_DIM = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    FFN_DIM = 2048
    DROPOUT = 0.2  # 增加dropout缓解过拟合
    MAX_LEN = 100  # 最大序列长度
    
    # 相对位置编码参数
    MAX_REL_POS = 128  # 最大相对位置距离
    
    # 训练参数
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5  # 降低学习率，更稳定的训练
    WARMUP_STEPS = 8000  # 增加warmup步数
    WEIGHT_DECAY = 0.05  # 增加权重衰减，更强的正则化
    GRAD_CLIP = 1.0
    
    # AdamW参数
    BETA1 = 0.9
    BETA2 = 0.98
    EPS = 1e-9
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存路径
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    FIGURE_DIR = "figures"
    
    # 评估参数
    BEAM_SIZE = 4
    MAX_DECODE_LEN = 100
    
    # 其他
    SEED = 42
    PRINT_FREQ = 100
    SAVE_FREQ = 5
    
    # 数据相关参数
    MAX_TRAIN_SAMPLES = 200000  # 训练集中使用的最大样本数
    RUN_NAME = None  # 实验名称，用于区分输出

    # 位置编码设置
    USE_POS_ENCODING = True  # 是否使用绝对位置编码
    
    # 早停参数
    EARLY_STOPPING = True  # 是否启用早停
    EARLY_STOPPING_PATIENCE = 5  # 验证损失不改善的epoch数，超过此值则停止训练
    EARLY_STOPPING_MIN_DELTA = 0.001  # 验证损失改善的最小阈值
    EARLY_STOPPING_RELATIVE_THRESHOLD = 0.05  # 相对阈值：如果当前损失比最佳损失高5%，立即触发早停

