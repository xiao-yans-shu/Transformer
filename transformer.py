"""
完整的Transformer模型（手工实现）
"""
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    """完整的Transformer模型（Encoder-Decoder架构）"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, ffn_dim=2048,
                 max_len=5000, dropout=0.1, use_relative_pos=False, max_rel_pos=128,
                 use_positional_encoding=True):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ffn_dim=ffn_dim,
            max_len=max_len,
            dropout=dropout,
            use_relative_pos=use_relative_pos,
            max_rel_pos=max_rel_pos,
            use_positional_encoding=use_positional_encoding
        )
        
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            ffn_dim=ffn_dim,
            max_len=max_len,
            dropout=dropout,
            use_relative_pos=use_relative_pos,
            max_rel_pos=max_rel_pos,
            use_positional_encoding=use_positional_encoding
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(embed_dim, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            src: [batch_size, src_len] 源序列token ids
            tgt: [batch_size, tgt_len] 目标序列token ids
            src_mask: [batch_size, src_len] 源序列mask，1表示有效位置
            tgt_mask: [batch_size, tgt_len] 目标序列mask，1表示有效位置
        Returns:
            [batch_size, tgt_len, tgt_vocab_size] 输出logits
        """
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        output = self.output_proj(decoder_output)
        
        return output
    
    def count_parameters(self):
        """统计模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self):
        """获取模型大小（MB）"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

