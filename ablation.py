"""
消融实验脚本 - 超参敏感性分析
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import Config
from transformer import Transformer
from data_loader import load_iwslt_data
from utils import WarmupScheduler, clip_gradients, TrainingLogger, calculate_perplexity, set_seed


def train_quick(model, train_loader, dev_loader, config, device, epochs=3):
    """快速训练用于消融实验"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD token
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
        eps=config.EPS,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = WarmupScheduler(optimizer, warmup_steps=config.WARMUP_STEPS, d_model=config.EMBED_DIM)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        train_tokens = 0
        
        for batch in train_loader:
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)
            
            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask)
            logits = logits.view(-1, logits.size(-1))
            tgt_output = tgt_output.view(-1)
            loss = criterion(logits, tgt_output)
            loss.backward()
            clip_gradients(model, config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            train_tokens += tgt_mask.sum().item()
            train_loss += loss.item() * tgt_mask.sum().item()
        
        avg_train_loss = train_loss / train_tokens if train_tokens > 0 else 0
        
        # 验证
        model.eval()
        val_loss = 0
        val_tokens = 0
        
        with torch.no_grad():
            for batch in dev_loader:
                src = batch['src'].to(device)
                tgt_input = batch['tgt_input'].to(device)
                tgt_output = batch['tgt_output'].to(device)
                src_mask = batch['src_mask'].to(device)
                tgt_mask = batch['tgt_mask'].to(device)
                
                logits = model(src, tgt_input, src_mask, tgt_mask)
                logits = logits.view(-1, logits.size(-1))
                tgt_output = tgt_output.view(-1)
                loss = criterion(logits, tgt_output)
                
                val_tokens += tgt_mask.sum().item()
                val_loss += loss.item() * tgt_mask.sum().item()
        
        avg_val_loss = val_loss / val_tokens if val_tokens > 0 else 0
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    return best_val_loss, calculate_perplexity(best_val_loss)


def ablation_study():
    """消融实验"""
    set_seed(Config.SEED)
    
    # 加载数据
    print("加载数据...")
    train_loader, dev_loader, src_vocab, tgt_vocab = load_iwslt_data(Config)
    
    results = {}
    
    # 1. 位置编码消融
    print("\n" + "="*50)
    print("位置编码消融实验")
    print("="*50)
    
    for use_relative_pos in [False, True]:
        print(f"\n使用相对位置编码: {use_relative_pos}")
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=Config.EMBED_DIM,
            num_heads=Config.NUM_HEADS,
            num_encoder_layers=Config.NUM_LAYERS,
            num_decoder_layers=Config.NUM_LAYERS,
            ffn_dim=Config.FFN_DIM,
            max_len=Config.MAX_LEN * 2,
            dropout=Config.DROPOUT,
            use_relative_pos=use_relative_pos,
            max_rel_pos=Config.MAX_REL_POS
        ).to(Config.DEVICE)
        
        val_loss, val_ppl = train_quick(model, train_loader, dev_loader, Config, Config.DEVICE, epochs=3)
        results[f'pos_encoding_{"relative" if use_relative_pos else "absolute"}'] = {
            'val_loss': val_loss,
            'val_ppl': val_ppl
        }
        print(f"验证损失: {val_loss:.4f}, 困惑度: {val_ppl:.2f}")
    
    # 2. 层数敏感性分析
    print("\n" + "="*50)
    print("层数敏感性分析")
    print("="*50)
    
    for num_layers in [2, 4, 6, 8]:
        print(f"\n层数: {num_layers}")
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=Config.EMBED_DIM,
            num_heads=Config.NUM_HEADS,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            ffn_dim=Config.FFN_DIM,
            max_len=Config.MAX_LEN * 2,
            dropout=Config.DROPOUT,
            use_relative_pos=False,
            max_rel_pos=Config.MAX_REL_POS
        ).to(Config.DEVICE)
        
        param_count = model.count_parameters()
        val_loss, val_ppl = train_quick(model, train_loader, dev_loader, Config, Config.DEVICE, epochs=3)
        results[f'layers_{num_layers}'] = {
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'params': param_count
        }
        print(f"参数数量: {param_count:,}, 验证损失: {val_loss:.4f}, 困惑度: {val_ppl:.2f}")
    
    # 3. 注意力头数敏感性分析
    print("\n" + "="*50)
    print("注意力头数敏感性分析")
    print("="*50)
    
    for num_heads in [4, 8, 16]:
        if Config.EMBED_DIM % num_heads != 0:
            continue
        print(f"\n注意力头数: {num_heads}")
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=Config.EMBED_DIM,
            num_heads=num_heads,
            num_encoder_layers=Config.NUM_LAYERS,
            num_decoder_layers=Config.NUM_LAYERS,
            ffn_dim=Config.FFN_DIM,
            max_len=Config.MAX_LEN * 2,
            dropout=Config.DROPOUT,
            use_relative_pos=False,
            max_rel_pos=Config.MAX_REL_POS
        ).to(Config.DEVICE)
        
        val_loss, val_ppl = train_quick(model, train_loader, dev_loader, Config, Config.DEVICE, epochs=3)
        results[f'heads_{num_heads}'] = {
            'val_loss': val_loss,
            'val_ppl': val_ppl
        }
        print(f"验证损失: {val_loss:.4f}, 困惑度: {val_ppl:.2f}")
    
    # 4. Dropout敏感性分析
    print("\n" + "="*50)
    print("Dropout敏感性分析")
    print("="*50)
    
    for dropout in [0.0, 0.1, 0.2, 0.3]:
        print(f"\nDropout: {dropout}")
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_dim=Config.EMBED_DIM,
            num_heads=Config.NUM_HEADS,
            num_encoder_layers=Config.NUM_LAYERS,
            num_decoder_layers=Config.NUM_LAYERS,
            ffn_dim=Config.FFN_DIM,
            max_len=Config.MAX_LEN * 2,
            dropout=dropout,
            use_relative_pos=False,
            max_rel_pos=Config.MAX_REL_POS
        ).to(Config.DEVICE)
        
        val_loss, val_ppl = train_quick(model, train_loader, dev_loader, Config, Config.DEVICE, epochs=3)
        results[f'dropout_{dropout}'] = {
            'val_loss': val_loss,
            'val_ppl': val_ppl
        }
        print(f"验证损失: {val_loss:.4f}, 困惑度: {val_ppl:.2f}")
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 绘制结果图
    plot_ablation_results(results)
    
    print("\n" + "="*50)
    print("消融实验完成！结果已保存到 results/ablation_results.json")
    print("="*50)


def plot_ablation_results(results):
    """绘制消融实验结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 位置编码对比
    if 'pos_encoding_absolute' in results and 'pos_encoding_relative' in results:
        pos_types = ['绝对位置编码', '相对位置编码']
        pos_losses = [
            results['pos_encoding_absolute']['val_loss'],
            results['pos_encoding_relative']['val_loss']
        ]
        axes[0, 0].bar(pos_types, pos_losses)
        axes[0, 0].set_ylabel('验证损失')
        axes[0, 0].set_title('位置编码对比')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 层数敏感性
    layer_results = {k: v for k, v in results.items() if k.startswith('layers_')}
    if layer_results:
        layers = sorted([int(k.split('_')[1]) for k in layer_results.keys()])
        losses = [layer_results[f'layers_{l}']['val_loss'] for l in layers]
        axes[0, 1].plot(layers, losses, marker='o')
        axes[0, 1].set_xlabel('层数')
        axes[0, 1].set_ylabel('验证损失')
        axes[0, 1].set_title('层数敏感性分析')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 注意力头数敏感性
    head_results = {k: v for k, v in results.items() if k.startswith('heads_')}
    if head_results:
        heads = sorted([int(k.split('_')[1]) for k in head_results.keys()])
        losses = [head_results[f'heads_{h}']['val_loss'] for h in heads]
        axes[1, 0].plot(heads, losses, marker='o')
        axes[1, 0].set_xlabel('注意力头数')
        axes[1, 0].set_ylabel('验证损失')
        axes[1, 0].set_title('注意力头数敏感性分析')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Dropout敏感性
    dropout_results = {k: v for k, v in results.items() if k.startswith('dropout_')}
    if dropout_results:
        dropouts = sorted([float(k.split('_')[1]) for k in dropout_results.keys()])
        losses = [dropout_results[f'dropout_{d}']['val_loss'] for d in dropouts]
        axes[1, 1].plot(dropouts, losses, marker='o')
        axes[1, 1].set_xlabel('Dropout率')
        axes[1, 1].set_ylabel('验证损失')
        axes[1, 1].set_title('Dropout敏感性分析')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/ablation_results.png', dpi=300, bbox_inches='tight')
    print("消融实验结果图已保存到 figures/ablation_results.png")
    plt.close()


if __name__ == '__main__':
    ablation_study()

