"""
位置编码模块 - 包括绝对位置编码和相对位置编码
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """绝对位置编码（正弦位置编码）"""
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        
        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            [batch_size, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RelativePositionEncoding(nn.Module):
    """
    相对位置编码（Shaw et al. 2018）
    为每个注意力头学习相对位置的嵌入
    """
    
    def __init__(self, embed_dim, num_heads, max_rel_pos=128):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.max_rel_pos = max_rel_pos
        
        # 相对位置嵌入矩阵
        # [2*max_rel_pos-1, head_dim]
        self.rel_pos_emb = nn.Parameter(
            torch.randn(2 * max_rel_pos - 1, self.head_dim)
        )
        
    def forward(self, seq_len):
        """
        计算相对位置索引矩阵
        
        Args:
            seq_len: 序列长度
        Returns:
            rel_pos_idx: [seq_len, seq_len] 相对位置索引矩阵
        """
        # 创建位置索引矩阵
        # pos_i - pos_j 的范围是 [-(seq_len-1), (seq_len-1)]
        pos_i = torch.arange(seq_len, dtype=torch.long).unsqueeze(1)
        pos_j = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        rel_pos = pos_i - pos_j  # [seq_len, seq_len]
        
        # 将相对位置映射到 [0, 2*max_rel_pos-2] 范围
        rel_pos = torch.clamp(rel_pos, -self.max_rel_pos + 1, self.max_rel_pos - 1)
        rel_pos = rel_pos + self.max_rel_pos - 1  # 映射到 [0, 2*max_rel_pos-2]
        
        return rel_pos
    
    def get_embeddings(self, rel_pos_idx):
        """
        获取相对位置嵌入
        
        Args:
            rel_pos_idx: [seq_len, seq_len] 相对位置索引
        Returns:
            [seq_len, seq_len, head_dim] 相对位置嵌入
        """
        return self.rel_pos_emb[rel_pos_idx]

