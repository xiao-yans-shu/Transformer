"""
训练工具函数
"""
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
from collections import defaultdict


class WarmupScheduler:
    """带warmup的学习率调度器"""
    
    def __init__(self, optimizer, warmup_steps, d_model=512):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_step = 0
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup阶段：线性增加
            lr = self.d_model ** (-0.5) * min(
                self.current_step ** (-0.5),
                self.current_step * self.warmup_steps ** (-1.5)
            )
        else:
            # 正常阶段：按步数衰减
            lr = self.d_model ** (-0.5) * self.current_step ** (-0.5)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


def clip_gradients(model, max_norm=1.0):
    """梯度裁剪"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def count_parameters(model):
    """统计模型参数数量"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.__dict__ if scheduler else None,
        'loss': loss,
    }, filepath)
    print(f"检查点已保存: {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath, device):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.__dict__.update(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"检查点已加载: {filepath}")
    return epoch, loss


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logs = defaultdict(list)
    
    def log(self, key, value, step=None):
        """记录日志"""
        self.logs[key].append((step, value) if step is not None else value)
    
    def plot_curves(self, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练Loss曲线（按epoch记录的值）
        if 'train_loss' in self.logs and len(self.logs['train_loss']) > 0:
            train_loss_data = self.logs['train_loss']
            if isinstance(train_loss_data[0], tuple):
                epochs, losses = zip(*train_loss_data)
                axes[0, 0].plot(epochs, losses, label='Train Loss', marker='o')
            else:
                axes[0, 0].plot(train_loss_data, label='Train Loss', marker='o')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss (per epoch)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 验证Loss曲线（按epoch记录的值）
        if 'val_loss' in self.logs and len(self.logs['val_loss']) > 0:
            val_loss_data = self.logs['val_loss']
            if isinstance(val_loss_data[0], tuple):
                epochs, losses = zip(*val_loss_data)
                # 过滤None值
                epochs_filtered = [e for e, l in zip(epochs, losses) if l is not None]
                losses_filtered = [l for l in losses if l is not None]
                if len(losses_filtered) > 0:
                    axes[0, 1].plot(epochs_filtered, losses_filtered, label='Val Loss', marker='o')
            else:
                losses_filtered = [l for l in val_loss_data if l is not None]
                if len(losses_filtered) > 0:
                    axes[0, 1].plot(losses_filtered, label='Val Loss', marker='o')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Validation Loss (per epoch)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No validation data', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Validation Loss (per epoch)')
        
        # Learning rate曲线
        if 'lr' in self.logs and len(self.logs['lr']) > 0:
            lr_data = self.logs['lr']
            if isinstance(lr_data[0], tuple):
                steps, lrs = zip(*lr_data)
                axes[1, 0].plot(steps, lrs, label='Learning Rate', alpha=0.7)
            else:
                axes[1, 0].plot(lr_data, label='Learning Rate', alpha=0.7)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # BLEU或Perplexity曲线（按epoch记录的值）
        if 'val_bleu' in self.logs and len(self.logs['val_bleu']) > 0:
            bleu_data = self.logs['val_bleu']
            if isinstance(bleu_data[0], tuple):
                epochs, bleus = zip(*bleu_data)
                # 过滤None值
                epochs_filtered = [e for e, b in zip(epochs, bleus) if b is not None]
                bleus_filtered = [b for b in bleus if b is not None]
                if len(bleus_filtered) > 0:
                    axes[1, 1].plot(epochs_filtered, bleus_filtered, label='Val BLEU', marker='o', color='green')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('BLEU Score')
            axes[1, 1].set_title('Validation BLEU (per epoch)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        elif 'train_ppl' in self.logs and len(self.logs['train_ppl']) > 0:
            ppl_data = self.logs['train_ppl']
            if isinstance(ppl_data[0], tuple):
                epochs, ppls = zip(*ppl_data)
                axes[1, 1].plot(epochs, ppls, label='Train PPL', marker='o')
            else:
                axes[1, 1].plot(ppl_data, label='Train PPL', marker='o')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Perplexity')
            axes[1, 1].set_title('Training Perplexity (per epoch)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")
        else:
            plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()


def calculate_perplexity(loss):
    """计算困惑度"""
    return math.exp(min(loss, 10))  # 防止溢出

