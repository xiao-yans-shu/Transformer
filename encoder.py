"""
Encoder模块
"""
import math
import torch
import torch.nn as nn
from attention import MultiHeadAttention, RelativePositionMultiHeadAttention


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        Returns:
            [batch_size, seq_len, embed_dim]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder层"""
    
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, use_relative_pos=False, max_rel_pos=128):
        super().__init__()
        # 自注意力
        if use_relative_pos:
            self.self_attn = RelativePositionMultiHeadAttention(
                embed_dim, num_heads, max_rel_pos, dropout
            )
        else:
            self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 前馈网络
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len, seq_len] 或 None
        Returns:
            [batch_size, seq_len, embed_dim]
        """
        # 自注意力 + 残差连接
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        
        return x


class Encoder(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_dim, 
                 max_len=5000, dropout=0.1, use_relative_pos=False, max_rel_pos=128,
                 use_positional_encoding=True):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码
        if use_relative_pos or not use_positional_encoding:
            self.pos_encoding = None
        else:
            from position_encoding import PositionalEncoding
            self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        # Encoder层
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ffn_dim, dropout, use_relative_pos, max_rel_pos)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len] token ids
            mask: [batch_size, seq_len] 或 None，1表示有效位置，0表示padding
        Returns:
            [batch_size, seq_len, embed_dim]
        """
        # 词嵌入
        x = self.token_embedding(x) * math.sqrt(self.embed_dim)
        
        # 位置编码
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        else:
            x = self.dropout(x)
        
        # 创建注意力mask（如果需要）
        if mask is not None:
            # 将mask扩展为 [batch_size, 1, seq_len, seq_len]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, -1, mask.size(1), -1)  # [batch_size, 1, seq_len, seq_len]
            attn_mask = attn_mask.squeeze(1)  # [batch_size, seq_len, seq_len]
        else:
            attn_mask = None
        
        # 通过Encoder层
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        # 最终归一化
        x = self.norm(x)
        
        return x

