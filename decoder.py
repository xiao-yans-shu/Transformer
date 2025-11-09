"""
Decoder模块
"""
import torch
import torch.nn as nn
import math
from attention import MultiHeadAttention, RelativePositionMultiHeadAttention
from encoder import FeedForward


class DecoderLayer(nn.Module):
    """Decoder层"""
    
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, use_relative_pos=False, max_rel_pos=128):
        super().__init__()
        # 掩码自注意力
        if use_relative_pos:
            self.self_attn = RelativePositionMultiHeadAttention(
                embed_dim, num_heads, max_rel_pos, dropout
            )
        else:
            self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 交叉注意力（Encoder-Decoder Attention）
        if use_relative_pos:
            self.cross_attn = RelativePositionMultiHeadAttention(
                embed_dim, num_heads, max_rel_pos, dropout
            )
        else:
            self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # 前馈网络
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: [batch_size, tgt_len, embed_dim] decoder输入
            encoder_output: [batch_size, src_len, embed_dim] encoder输出
            self_mask: [batch_size, tgt_len, tgt_len] 自注意力mask（因果mask + padding mask）
            cross_mask: [batch_size, tgt_len, src_len] 交叉注意力mask
        Returns:
            [batch_size, tgt_len, embed_dim]
        """
        # 掩码自注意力 + 残差连接
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = residual + self.dropout(attn_output)
        
        # 交叉注意力 + 残差连接
        residual = x
        x = self.norm2(x)
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = residual + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接
        residual = x
        x = self.norm3(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        
        return x


class Decoder(nn.Module):
    """Transformer Decoder"""
    
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
        
        # Decoder层
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ffn_dim, dropout, use_relative_pos, max_rel_pos)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x: [batch_size, tgt_len] token ids
            encoder_output: [batch_size, src_len, embed_dim] encoder输出
            tgt_mask: [batch_size, tgt_len] 目标序列mask，1表示有效位置
            src_mask: [batch_size, src_len] 源序列mask，1表示有效位置
        Returns:
            [batch_size, tgt_len, embed_dim]
        """
        # 词嵌入
        x = self.token_embedding(x) * math.sqrt(self.embed_dim)
        
        # 位置编码
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        else:
            x = self.dropout(x)
        
        # 创建因果mask（防止看到未来信息）
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        causal_mask = causal_mask.bool()  # [seq_len, seq_len]，True表示需要mask的位置
        
        # 创建自注意力mask（因果mask + padding mask）
        if tgt_mask is not None:
            # tgt_mask: [batch_size, tgt_len]
            tgt_mask_expanded = tgt_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
            tgt_mask_expanded = tgt_mask_expanded.expand(-1, -1, seq_len, -1)  # [batch_size, 1, tgt_len, tgt_len]
            tgt_mask_expanded = tgt_mask_expanded.squeeze(1)  # [batch_size, tgt_len, tgt_len]
            # 组合因果mask和padding mask
            self_mask = (causal_mask.unsqueeze(0) | (~tgt_mask_expanded.bool()))
            self_mask = (~self_mask).float()  # 1表示有效位置，0表示需要mask
        else:
            self_mask = (~causal_mask.unsqueeze(0)).float()  # [1, tgt_len, tgt_len]
        
        # 创建交叉注意力mask
        if src_mask is not None:
            src_mask_expanded = src_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
            src_mask_expanded = src_mask_expanded.expand(-1, -1, seq_len, -1)  # [batch_size, 1, tgt_len, src_len]
            cross_mask = src_mask_expanded.squeeze(1).float()  # [batch_size, tgt_len, src_len]
        else:
            cross_mask = None
        
        # 通过Decoder层
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        
        # 最终归一化
        x = self.norm(x)
        
        return x

