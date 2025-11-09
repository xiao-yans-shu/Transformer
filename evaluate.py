"""
评估脚本 - 计算BLEU分数
"""
import torch
import torch.nn.functional as F
from collections import Counter
import math
import argparse
from tqdm import tqdm

from config import Config
from transformer import Transformer
from data_loader import load_iwslt_data, Vocabulary
from utils import load_checkpoint


def ngram_counts(tokens, n):
    """计算n-gram计数"""
    counts = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        counts[ngram] += 1
    return counts


def bleu_score(candidate, reference, n=4):
    """计算BLEU分数"""
    # 转换为列表
    if isinstance(candidate, str):
        candidate = candidate.split()
    if isinstance(reference, str):
        reference = reference.split()
    
    # 计算各n-gram的精确率
    precisions = []
    for i in range(1, n + 1):
        cand_ngrams = ngram_counts(candidate, i)
        ref_ngrams = ngram_counts(reference, i)
        
        # 计算匹配的n-gram数量
        matches = sum(min(cand_ngrams[ngram], ref_ngrams[ngram]) 
                     for ngram in cand_ngrams)
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)
    
    # 几何平均
    if any(p == 0 for p in precisions):
        return 0.0
    
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / n)
    
    # 长度惩罚
    if len(candidate) == 0:
        return 0.0
    
    brevity_penalty = min(1.0, len(reference) / len(candidate))
    
    return geo_mean * brevity_penalty


def greedy_decode(model, src, src_mask, tgt_vocab, max_len=100, device='cuda'):
    """贪婪解码"""
    model.eval()
    src = src.unsqueeze(0) if src.dim() == 1 else src
    src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 1 else src_mask
    
    batch_size = src.size(0)
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    # 编码源序列
    encoder_output = model.encoder(src, src_mask)
    
    # 初始化目标序列
    tgt = torch.zeros(batch_size, 1, dtype=torch.long).to(device)
    tgt[:, 0] = tgt_vocab.word2idx[tgt_vocab.BOS]
    
    decoded = []
    
    with torch.no_grad():
        for _ in range(max_len):
            # 获取当前目标序列的mask
            tgt_len = tgt.size(1)
            tgt_mask = torch.ones(batch_size, tgt_len, dtype=torch.long).to(device)
            
            # 解码
            decoder_output = model.decoder(tgt, encoder_output, tgt_mask, src_mask)
            logits = model.output_proj(decoder_output)
            
            # 获取最后一个位置的预测
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # 添加到序列
            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
            
            # 检查是否到达EOS
            if (next_token == tgt_vocab.word2idx[tgt_vocab.EOS]).all():
                break
    
    # 解码为目标文本
    for i in range(batch_size):
        tokens = tgt[i].cpu().tolist()
        decoded.append(tgt_vocab.decode(tokens))
    
    return decoded if batch_size > 1 else decoded[0]


def beam_search_decode(model, src, src_mask, tgt_vocab, beam_size=4, max_len=100, device='cuda'):
    """Beam Search解码"""
    model.eval()
    src = src.unsqueeze(0) if src.dim() == 1 else src
    src_mask = src_mask.unsqueeze(0) if src_mask.dim() == 1 else src_mask
    
    batch_size = src.size(0)
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    # 编码源序列
    with torch.no_grad():
        encoder_output = model.encoder(src, src_mask)
    
    # 初始化beam
    beams = [{
        'sequence': torch.tensor([[tgt_vocab.word2idx[tgt_vocab.BOS]]], dtype=torch.long).to(device),
        'score': 0.0
    }]
    
    for step in range(max_len):
        new_beams = []
        
        for beam in beams:
            tgt = beam['sequence']
            tgt_len = tgt.size(1)
            tgt_mask = torch.ones(1, tgt_len, dtype=torch.long).to(device)
            
            # 解码
            with torch.no_grad():
                decoder_output = model.decoder(tgt, encoder_output, tgt_mask, src_mask)
                logits = model.output_proj(decoder_output)
                next_token_logits = logits[:, -1, :]
                probs = F.log_softmax(next_token_logits, dim=-1)
            
            # 获取top-k候选
            top_probs, top_indices = torch.topk(probs, beam_size, dim=-1)
            
            for i in range(beam_size):
                token = top_indices[0, i].item()
                token_prob = top_probs[0, i].item()
                
                new_sequence = torch.cat([tgt, torch.tensor([[token]], dtype=torch.long).to(device)], dim=1)
                new_score = beam['score'] + token_prob
                
                new_beams.append({
                    'sequence': new_sequence,
                    'score': new_score
                })
                
                # 如果遇到EOS，停止扩展
                if token == tgt_vocab.word2idx[tgt_vocab.EOS]:
                    break
        
        # 选择top beam_size个beam
        beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)[:beam_size]
        
        # 检查是否所有beam都结束
        if all(beam['sequence'][0, -1].item() == tgt_vocab.word2idx[tgt_vocab.EOS] for beam in beams):
            break
    
    # 选择最佳beam
    best_beam = beams[0]
    tokens = best_beam['sequence'][0].cpu().tolist()
    decoded = tgt_vocab.decode(tokens)
    
    return decoded


def evaluate(model, data_loader, tgt_vocab, device, use_beam_search=False, beam_size=4):
    """评估模型"""
    model.eval()
    all_candidates = []
    all_references = []
    
    print("开始评估...")
    for batch in tqdm(data_loader):
        src = batch['src'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_texts = batch['tgt_text']
        
        # 解码
        if use_beam_search:
            for i in range(src.size(0)):
                candidate = beam_search_decode(
                    model, src[i], src_mask[i], tgt_vocab,
                    beam_size=beam_size, device=device
                )
                all_candidates.append(candidate)
        else:
            candidates = greedy_decode(
                model, src, src_mask, tgt_vocab, device=device
            )
            if isinstance(candidates, str):
                all_candidates.append(candidates)
            else:
                all_candidates.extend(candidates)
        
        all_references.extend(tgt_texts)
    
    # 计算BLEU分数
    bleu_scores = []
    for cand, ref in zip(all_candidates, all_references):
        bleu = bleu_score(cand, ref)
        bleu_scores.append(bleu)
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    return avg_bleu, all_candidates, all_references


def main(args):
    # 加载数据
    print("加载数据...")
    _, dev_loader, src_vocab, tgt_vocab = load_iwslt_data(Config)
    
    # 创建模型
    print("创建模型...")
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
        use_relative_pos=args.use_relative_pos if args.use_relative_pos else False,
        max_rel_pos=Config.MAX_REL_POS
    ).to(Config.DEVICE)
    
    # 加载模型
    checkpoint_path = args.checkpoint if args.checkpoint else 'checkpoints/best_model.pt'
    print(f"加载模型: {checkpoint_path}")
    load_checkpoint(model, None, None, checkpoint_path, Config.DEVICE)
    
    # 评估
    avg_bleu, candidates, references = evaluate(
        model, dev_loader, tgt_vocab, Config.DEVICE,
        use_beam_search=args.beam_search,
        beam_size=args.beam_size if args.beam_size else Config.BEAM_SIZE
    )
    
    print(f"\n{'='*50}")
    print(f"评估结果:")
    print(f"  平均BLEU分数: {avg_bleu:.4f}")
    print(f"{'='*50}")
    
    # 打印一些示例
    print("\n示例翻译:")
    for i in range(min(5, len(candidates))):
        print(f"\n参考: {references[i]}")
        print(f"生成: {candidates[i]}")
        print(f"BLEU: {bleu_score(candidates[i], references[i]):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估Transformer模型')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--beam_search', action='store_true', help='使用Beam Search（推荐启用）')
    parser.add_argument('--no_beam_search', dest='beam_search', action='store_false', help='禁用Beam Search')
    parser.set_defaults(beam_search=True)  # 默认启用Beam Search
    parser.add_argument('--beam_size', type=int, default=4, help='Beam大小')
    parser.add_argument('--use_relative_pos', action='store_true', help='使用相对位置编码')
    
    args = parser.parse_args()
    main(args)

