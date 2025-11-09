# -*- coding: utf-8 -*-
"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
from tqdm import tqdm
import argparse
from datetime import datetime

from config import Config
from transformer import Transformer
from data_loader import load_iwslt_data
from utils import WarmupScheduler, clip_gradients, save_checkpoint, TrainingLogger, calculate_perplexity


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(pbar):
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # 计算损失（只对非padding位置计算）
        logits = logits.view(-1, logits.size(-1))
        tgt_output = tgt_output.view(-1)
        loss = criterion(logits, tgt_output)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        clip_gradients(model, config.GRAD_CLIP)
        
        # 更新参数
        optimizer.step()
        scheduler.step()
        
        # 统计
        num_tokens = tgt_mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        # 记录学习率（用于绘制学习率曲线）
        current_lr = scheduler.get_lr()
        if step % 10 == 0:  # 每10步记录一次，减少数据点
            logger.log('lr', current_lr, step=step)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{calculate_perplexity(loss.item()):.2f}',
            'lr': f'{current_lr:.2e}'
        })
        
        if step % config.PRINT_FREQ == 0:
            print(f"\nStep {step}: Loss={loss.item():.4f}, "
                  f"PPL={calculate_perplexity(loss.item()):.2f}, "
                  f"LR={current_lr:.2e}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    avg_ppl = calculate_perplexity(avg_loss)
    
    return avg_loss, avg_ppl


def validate(model, dev_loader, criterion, device, tgt_vocab=None, compute_bleu=False):
    """
    验证
    
    Args:
        model: 模型
        dev_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        tgt_vocab: 目标语言词汇表（用于计算BLEU）
        compute_bleu: 是否计算BLEU分数（需要解码，较慢）
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # 检查验证集是否为空
    if len(dev_loader.dataset) == 0:
        print("警告: 验证集为空，跳过验证")
        return None, None, None
    
    # 用于BLEU计算
    all_candidates = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Validating"):
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)
            
            # 前向传播
            logits = model(src, tgt_input, src_mask, tgt_mask)
            
            # 计算损失
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_output_flat = tgt_output.view(-1)
            loss = criterion(logits_flat, tgt_output_flat)
            
            # 统计
            num_tokens = tgt_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # 如果需要计算BLEU，进行解码
            if compute_bleu and tgt_vocab is not None:
                from evaluate import greedy_decode
                for i in range(src.size(0)):
                    candidate = greedy_decode(
                        model, src[i], src_mask[i], tgt_vocab, 
                        max_len=100, device=device
                    )
                    all_candidates.append(candidate)
                all_references.extend(batch['tgt_text'])
    
    if total_tokens == 0:
        print("警告: 验证集没有有效token，跳过验证")
        return None, None, None
    
    avg_loss = total_loss / total_tokens
    avg_ppl = calculate_perplexity(avg_loss)
    
    # 计算BLEU（如果请求）
    avg_bleu = None
    if compute_bleu and tgt_vocab is not None and len(all_candidates) > 0:
        from evaluate import bleu_score
        bleu_scores = []
        for cand, ref in zip(all_candidates, all_references):
            bleu = bleu_score(cand, ref)
            bleu_scores.append(bleu)
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    return avg_loss, avg_ppl, avg_bleu


def main(args):
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 生成运行名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = Config.RUN_NAME or timestamp
    run_dir = os.path.join('runs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # 创建子目录
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    fig_dir = os.path.join(run_dir, 'figures')
    log_dir = os.path.join(run_dir, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志文件（保存控制台输出）
    log_filename = f"training_log_{timestamp}.txt"
    log_filepath = os.path.join(log_dir, log_filename)
    from logger import TeeLogger
    tee_logger = TeeLogger(log_filepath)
    sys.stdout = tee_logger
    
    # 加载数据
    print("=" * 50)
    print(f"运行名称: {run_name}")
    print("加载数据...")
    train_loader, dev_loader, src_vocab, tgt_vocab = load_iwslt_data(Config)
    
    # 创建模型
    print("=" * 50)
    print("创建模型...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim if args.embed_dim else Config.EMBED_DIM,
        num_heads=args.num_heads if args.num_heads else Config.NUM_HEADS,
        num_encoder_layers=args.num_layers if args.num_layers else Config.NUM_LAYERS,
        num_decoder_layers=args.num_layers if args.num_layers else Config.NUM_LAYERS,
        ffn_dim=args.ffn_dim if args.ffn_dim else Config.FFN_DIM,
        max_len=Config.MAX_LEN * 2,
        dropout=args.dropout if args.dropout is not None else Config.DROPOUT,
        use_relative_pos=args.use_relative_pos if args.use_relative_pos else False,
        max_rel_pos=Config.MAX_REL_POS,
        use_positional_encoding=Config.USE_POS_ENCODING
    ).to(Config.DEVICE)
    
    # 打印模型信息
    trainable_params, total_params = model.count_parameters(), sum(p.numel() for p in model.parameters())
    model_size = model.get_model_size()
    print(f"模型参数统计:")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  总参数: {total_params:,}")
    print(f"  模型大小: {model_size:.2f} MB")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.word2idx[src_vocab.PAD])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr if args.lr else Config.LEARNING_RATE,
        betas=(Config.BETA1, Config.BETA2),
        eps=Config.EPS,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps if args.warmup_steps else Config.WARMUP_STEPS,
        d_model=args.embed_dim if args.embed_dim else Config.EMBED_DIM
    )
    
    # 训练日志
    logger = TrainingLogger(log_dir)
    
    # 训练循环
    print("=" * 50)
    print("开始训练...")
    best_val_loss = float('inf')
    best_train_loss = float('inf')  # 用于验证集为空时保存模型
    start_epoch = 0
    
    # 早停相关变量
    patience_counter = 0
    best_val_bleu = 0.0  # 用于基于BLEU的早停
    
    # 如果指定了检查点，加载
    if args.resume:
        checkpoint_path = os.path.join(ckpt_dir, args.resume)
        if os.path.exists(checkpoint_path):
            start_epoch, _ = load_checkpoint(
                model, optimizer, scheduler, checkpoint_path, Config.DEVICE
            )
            print(f"从epoch {start_epoch}继续训练...")
    
    for epoch in range(start_epoch, args.epochs if args.epochs else Config.NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs if args.epochs else Config.NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_ppl = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            Config.DEVICE, Config, logger
        )
        
        # 验证（每3个epoch计算一次BLEU，更频繁地监控性能）
        compute_bleu = (epoch + 1) % 3 == 0
        val_loss, val_ppl, val_bleu = validate(
            model, dev_loader, criterion, Config.DEVICE, 
            tgt_vocab=tgt_vocab, compute_bleu=compute_bleu
        )
        
        # 记录日志（按epoch记录）
        logger.log('train_loss', train_loss, step=epoch)  # 训练损失按epoch记录
        if val_loss is not None:
            logger.log('val_loss', val_loss, step=epoch)
        logger.log('train_ppl', train_ppl, step=epoch)
        if val_ppl is not None:
            logger.log('val_ppl', val_ppl, step=epoch)
        if val_bleu is not None:
            logger.log('val_bleu', val_bleu, step=epoch)
        
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  训练损失: {train_loss:.4f}, 困惑度: {train_ppl:.2f}")
        if val_loss is not None:
            print(f"  验证损失: {val_loss:.4f}, 困惑度: {val_ppl:.2f}")
            if val_bleu is not None:
                print(f"  验证BLEU: {val_bleu:.4f} ({val_bleu*100:.2f}%)")
        else:
            print(f"  验证损失: N/A (验证集为空)")
        
        # 保存最佳模型和早停检查
        improved = False
        should_stop = False
        stop_reason = ""
        
        if val_loss is not None:
            # 基于验证损失判断是否改善
            if val_loss < best_val_loss - Config.EARLY_STOPPING_MIN_DELTA:
                best_val_loss = val_loss
                improved = True
                patience_counter = 0
                checkpoint_path = os.path.join(ckpt_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)
                print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  验证损失未改善 (patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE})")
                print(f"  当前: {val_loss:.4f}, 最佳: {best_val_loss:.4f}, 差距: {val_loss - best_val_loss:.4f}")
            
            # 基于BLEU判断（如果可用）
            if val_bleu is not None and val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                if not improved:  # 如果损失没改善但BLEU改善了，也保存
                    checkpoint_path = os.path.join(ckpt_dir, 'best_model_bleu.pt')
                    save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)
                    print(f"  ✓ 保存最佳BLEU模型 (BLEU: {val_bleu:.4f})")
            
            # 早停检查（更严格的逻辑）
            if Config.EARLY_STOPPING:
                # 检查1: patience计数器
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    should_stop = True
                    stop_reason = f"验证损失在 {Config.EARLY_STOPPING_PATIENCE} 个epoch内未改善"
                
                # 检查2: 相对阈值（如果当前损失比最佳损失高一定比例，立即停止）
                if best_val_loss > 0:
                    relative_increase = (val_loss - best_val_loss) / best_val_loss
                    if relative_increase > Config.EARLY_STOPPING_RELATIVE_THRESHOLD:
                        should_stop = True
                        stop_reason = f"验证损失比最佳值高 {relative_increase*100:.1f}% (超过 {Config.EARLY_STOPPING_RELATIVE_THRESHOLD*100}% 阈值)"
                
                if should_stop:
                    print(f"\n{'='*50}")
                    print(f"早停触发！{stop_reason}")
                    print(f"当前验证损失: {val_loss:.4f}")
                    print(f"最佳验证损失: {best_val_loss:.4f} (epoch {epoch - patience_counter + 1})")
                    if best_val_bleu > 0:
                        print(f"最佳验证BLEU: {best_val_bleu:.4f}")
                    print(f"{'='*50}")
                    # 绘制最终曲线
                    logger.plot_curves(os.path.join(fig_dir, 'training_curves_final.png'))
                    # 恢复标准输出并关闭日志文件
                    sys.stdout = tee_logger.terminal
                    tee_logger.close()
                    print(f"日志已保存到: {log_filepath}")
                    break
                    
        elif val_loss is None:
            # 如果验证集为空，基于训练损失保存
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                improved = True
                checkpoint_path = os.path.join(ckpt_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_path)
                print(f"  ✓ 保存模型 (训练损失: {train_loss:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % Config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_loss = val_loss if val_loss is not None else train_loss
            save_checkpoint(model, optimizer, scheduler, epoch, save_loss, checkpoint_path)
        
        # 绘制训练曲线
        if (epoch + 1) % 5 == 0:
            logger.plot_curves(os.path.join(fig_dir, 'training_curves.png'))
    
    # 最终保存
    logger.plot_curves(os.path.join(fig_dir, 'training_curves_final.png'))
    print("\n训练完成！")
    
    # 恢复标准输出并关闭日志文件
    sys.stdout = tee_logger.terminal
    tee_logger.close()
    print(f"日志已保存到: {log_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    parser.add_argument('--embed_dim', type=int, default=None, help='嵌入维度')
    parser.add_argument('--num_heads', type=int, default=None, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=None, help='层数')
    parser.add_argument('--ffn_dim', type=int, default=None, help='前馈网络维度')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=None, help='Warmup步数')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--use_relative_pos', action='store_true', help='使用相对位置编码')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout率（用于缓解过拟合，建议0.2-0.3）')
    parser.add_argument('--patience', type=int, default=None, help='早停patience（验证损失不改善的epoch数）')
    parser.add_argument('--vocab_size', type=int, default=None, help='词汇表大小（覆盖配置）')
    parser.add_argument('--max_train_samples', type=int, default=None, help='训练样本数量上限（覆盖配置）')
    parser.add_argument('--run_name', type=str, default=None, help='自定义实验名称，用于保存日志/模型/图像')
    parser.add_argument('--disable_pos_encoding', action='store_true', help='禁用绝对位置编码（仅保留嵌入）')
    
    args = parser.parse_args()
    
    # 如果指定了patience，更新配置
    if args.patience is not None:
        Config.EARLY_STOPPING_PATIENCE = args.patience
    
    if args.dropout is not None:
        Config.DROPOUT = args.dropout
    
    if args.vocab_size is not None:
        Config.VOCAB_SIZE = args.vocab_size
    
    if args.max_train_samples is not None:
        Config.MAX_TRAIN_SAMPLES = args.max_train_samples

    if args.disable_pos_encoding:
        Config.USE_POS_ENCODING = False

    if args.run_name is not None:
        Config.RUN_NAME = args.run_name

    main(args)

