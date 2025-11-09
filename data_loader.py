"""
IWSLT17数据集加载和预处理
"""
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import xml.etree.ElementTree as ET


class Vocabulary:
    """词汇表"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # 特殊token
        self.PAD = '<PAD>'
        self.UNK = '<UNK>'
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        
        self.special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        
        # 初始化特殊token
        for token in self.special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token
    
    def build_vocab(self, sentences, max_size=10000):
        """构建词汇表"""
        # 统计词频
        for sentence in sentences:
            words = sentence.lower().split()
            self.word_count.update(words)
        
        # 选择最常见的词
        most_common = self.word_count.most_common(max_size - len(self.special_tokens))
        
        # 添加到词汇表
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word
    
    def encode(self, sentence, add_bos=False, add_eos=False):
        """将句子编码为token ids"""
        words = sentence.lower().split()
        tokens = []
        
        if add_bos:
            tokens.append(self.word2idx[self.BOS])
        
        for word in words:
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.word2idx[self.UNK])
        
        if add_eos:
            tokens.append(self.word2idx[self.EOS])
        
        return tokens
    
    def decode(self, token_ids):
        """将token ids解码为句子"""
        words = []
        for idx in token_ids:
            if idx == self.word2idx[self.PAD]:
                break
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word not in [self.BOS, self.EOS]:
                    words.append(word)
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)


def parse_xml_file(filepath):
    """解析XML格式的文件（dev/test集）"""
    sentences = []
    
    # 检查文件是否存在
    import os
    if not os.path.exists(filepath):
        print(f"警告: 文件不存在: {filepath}")
        return sentences
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # 尝试不同的XML结构
        # 方法1: 查找所有seg标签
        segs = root.findall('.//seg')
        if segs:
            for seg in segs:
                text = seg.text
                if text:
                    text = text.strip()
                    if text:
                        sentences.append(text)
        else:
            # 方法2: 查找doc下的seg
            for doc in root.findall('doc'):
                for seg in doc.findall('seg'):
                    text = seg.text
                    if text:
                        text = text.strip()
                        if text:
                            sentences.append(text)
        
        if not sentences:
            # 如果XML解析没有找到句子，尝试按行读取
            print(f"XML解析未找到句子，尝试按行读取: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # 跳过XML标签
                    if not line.strip().startswith('<'):
                        text = re.sub(r'<[^>]+>', '', line).strip()
                        if text and len(text) > 5:  # 过滤太短的文本
                            sentences.append(text)
    except ET.ParseError as e:
        print(f"XML解析错误 {filepath}: {e}")
        # 如果XML解析失败，尝试按行读取
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # 跳过XML标签
                if not line.strip().startswith('<'):
                    text = re.sub(r'<[^>]+>', '', line).strip()
                    if text and len(text) > 5:  # 过滤太短的文本
                        sentences.append(text)
    except Exception as e:
        print(f"读取文件错误 {filepath}: {e}")
        # 如果XML解析失败，尝试按行读取
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # 跳过XML标签
                    if not line.strip().startswith('<'):
                        text = re.sub(r'<[^>]+>', '', line).strip()
                        if text and len(text) > 5:  # 过滤太短的文本
                            sentences.append(text)
        except Exception as e2:
            print(f"按行读取也失败 {filepath}: {e2}")
    
    return sentences


def parse_text_file(filepath):
    """解析文本文件（训练集）"""
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过XML标签
            if not line.strip().startswith('<'):
                text = re.sub(r'<[^>]+>', '', line).strip()
                if text:
                    sentences.append(text)
    return sentences


class TranslationDataset(Dataset):
    """翻译数据集"""
    
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len=100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
        # 过滤太长的句子
        self.filtered_pairs = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_words = len(src.lower().split())
            tgt_words = len(tgt.lower().split())
            if src_words <= max_len and tgt_words <= max_len:
                self.filtered_pairs.append((src, tgt))
    
    def __len__(self):
        return len(self.filtered_pairs)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.filtered_pairs[idx]
        
        # 编码源序列（不添加BOS/EOS）
        src_ids = self.src_vocab.encode(src_text, add_bos=False, add_eos=False)
        
        # 编码目标序列（添加BOS和EOS）
        tgt_input_ids = self.tgt_vocab.encode(tgt_text, add_bos=True, add_eos=False)
        tgt_output_ids = self.tgt_vocab.encode(tgt_text, add_bos=False, add_eos=True)
        
        # 转换为tensor
        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_input_ids = torch.tensor(tgt_input_ids, dtype=torch.long)
        tgt_output_ids = torch.tensor(tgt_output_ids, dtype=torch.long)
        
        return {
            'src': src_ids,
            'tgt_input': tgt_input_ids,
            'tgt_output': tgt_output_ids,
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def collate_fn(batch):
    """批处理函数"""
    # 获取最大长度
    max_src_len = max(item['src'].size(0) for item in batch)
    max_tgt_len = max(item['tgt_input'].size(0) for item in batch)
    
    batch_size = len(batch)
    src_vocab_size = len(batch[0]['src'])
    
    # 初始化tensor
    src_ids = torch.zeros(batch_size, max_src_len, dtype=torch.long)
    tgt_input_ids = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    tgt_output_ids = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    src_mask = torch.zeros(batch_size, max_src_len, dtype=torch.long)
    tgt_mask = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    
    src_texts = []
    tgt_texts = []
    
    for i, item in enumerate(batch):
        src_len = item['src'].size(0)
        tgt_len = item['tgt_input'].size(0)
        
        # 填充源序列
        src_ids[i, :src_len] = item['src']
        src_mask[i, :src_len] = 1
        
        # 填充目标序列
        tgt_input_ids[i, :tgt_len] = item['tgt_input']
        tgt_output_ids[i, :tgt_len] = item['tgt_output']
        tgt_mask[i, :tgt_len] = 1
        
        src_texts.append(item['src_text'])
        tgt_texts.append(item['tgt_text'])
    
    return {
        'src': src_ids,
        'tgt_input': tgt_input_ids,
        'tgt_output': tgt_output_ids,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_text': src_texts,
        'tgt_text': tgt_texts
    }


def load_iwslt_data(config):
    """加载IWSLT17数据集"""
    print("加载训练数据...")
    # 加载训练数据
    src_train = parse_text_file(config.TRAIN_EN)
    tgt_train = parse_text_file(config.TRAIN_DE)
    
    # 限制数据量（使用配置中的限制）
    max_samples = getattr(config, 'MAX_TRAIN_SAMPLES', None)
    if max_samples is not None and len(src_train) > max_samples:
        print(f"训练数据量: {len(src_train)}，限制为: {max_samples}")
        src_train = src_train[:max_samples]
        tgt_train = tgt_train[:max_samples]
    else:
        print(f"使用全部训练数据: {len(src_train)}")
    
    print(f"训练集大小: {len(src_train)}")
    
    print("构建词汇表...")
    # 构建源语言词汇表
    src_vocab = Vocabulary()
    src_vocab.build_vocab(src_train, max_size=config.VOCAB_SIZE)
    print(f"源语言词汇表大小: {len(src_vocab)}")
    
    # 构建目标语言词汇表
    tgt_vocab = Vocabulary()
    tgt_vocab.build_vocab(tgt_train, max_size=config.VOCAB_SIZE)
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    
    print("加载验证数据...")
    # 加载验证数据
    src_dev = parse_xml_file(config.DEV_EN)
    tgt_dev = parse_xml_file(config.DEV_DE)
    
    # 限制验证集大小
    if len(src_dev) > 1000:
        src_dev = src_dev[:1000]
        tgt_dev = tgt_dev[:1000]
    
    print(f"验证集大小: {len(src_dev)}")
    
    # 创建数据集
    train_dataset = TranslationDataset(
        src_train, tgt_train, src_vocab, tgt_vocab, max_len=config.MAX_LEN
    )
    dev_dataset = TranslationDataset(
        src_dev, tgt_dev, src_vocab, tgt_vocab, max_len=config.MAX_LEN
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    return train_loader, dev_loader, src_vocab, tgt_vocab

