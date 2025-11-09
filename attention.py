"""
注意力机制模块 - 包括标准多头注意力和相对位置注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from position_encoding import RelativePositionEncoding


class MultiHeadAttention(nn.Module):
    """标准多头注意力（绝对位置编码）"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, embed_dim]
            key: [batch_size, seq_len_k, embed_dim]
            value: [batch_size, seq_len_k, embed_dim]
            mask: [batch_size, seq_len_q, seq_len_k] 或 None
        Returns:
            output: [batch_size, seq_len_q, embed_dim]
            attn_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 线性投影并重塑为多头
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 应用mask
        if mask is not None:
            # mask: [batch_size, 1, seq_len_q, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到value
        output = torch.matmul(attn_weights, V)
        # output: [batch_size, num_heads, seq_len_q, head_dim]
        
        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, attn_weights


class RelativePositionMultiHeadAttention(nn.Module):
    """带相对位置编码的多头注意力（Shaw et al. 2018）"""
    
    def __init__(self, embed_dim, num_heads, max_rel_pos=128, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 相对位置编码
        self.rel_pos_encoding = RelativePositionEncoding(
            embed_dim, num_heads, max_rel_pos
        )
        
        # 相对位置的key和value投影（可选）
        self.rel_k_proj = nn.Linear(embed_dim, embed_dim)
        self.rel_v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, embed_dim]
            key: [batch_size, seq_len_k, embed_dim]
            value: [batch_size, seq_len_k, embed_dim]
            mask: [batch_size, seq_len_q, seq_len_k] 或 None
        Returns:
            output: [batch_size, seq_len_q, embed_dim]
            attn_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 线性投影并重塑为多头
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]
        
        # 计算内容注意力分数: QK^T
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # content_scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 计算相对位置注意力分数
        # 使用向量化方法计算相对位置分数
        # 计算相对位置索引矩阵
        pos_i = torch.arange(seq_len_q, dtype=torch.long, device=query.device).unsqueeze(1)
        pos_j = torch.arange(seq_len_k, dtype=torch.long, device=query.device).unsqueeze(0)
        rel_pos = pos_j - pos_i  # [seq_len_q, seq_len_k]
        rel_pos = torch.clamp(rel_pos, -self.rel_pos_encoding.max_rel_pos + 1, 
                            self.rel_pos_encoding.max_rel_pos - 1)
        rel_pos = rel_pos + self.rel_pos_encoding.max_rel_pos - 1
        
        # 获取相对位置嵌入 [seq_len_q, seq_len_k, head_dim]
        rel_pos_emb = self.rel_pos_encoding.rel_pos_emb[rel_pos]
        
        # 计算相对位置分数: Q @ rel_pos_emb^T
        # Q: [batch_size, num_heads, seq_len_q, head_dim]
        # rel_pos_emb: [seq_len_q, seq_len_k, head_dim]
        # 扩展维度以便批量计算
        Q_expanded = Q.unsqueeze(3)  # [batch_size, num_heads, seq_len_q, 1, head_dim]
        rel_pos_emb_expanded = rel_pos_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, seq_len_k, head_dim]
        rel_scores = torch.sum(Q_expanded * rel_pos_emb_expanded, dim=-1) / self.scale
        # rel_scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 合并内容分数和相对位置分数
        scores = content_scores + rel_scores
        
        # 应用mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到value
        output = torch.matmul(attn_weights, V)
        # output: [batch_size, num_heads, seq_len_q, head_dim]
        
        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, attn_weights

