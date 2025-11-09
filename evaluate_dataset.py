# -*- coding: utf-8 -*-
"""
一次性完成评估脚本

步骤：
1. 指定要评估的数据集（XML路径）与模型权重
2. 将XML转为纯文本
3. 使用训练好的模型生成翻译结果
4. 调用sacrebleu计算BLEU分数（含大小写敏感/不敏感两种）
5. 产出所有中间文件与结果到 experiments/manual_eval/<dataset_name>/
"""
import os
import sys
from datetime import datetime
from typing import List

import torch
try:
    from sacrebleu import corpus_bleu
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 sacrebleu: pip install sacrebleu") from exc

from config import Config
from data_loader import parse_xml_file, load_iwslt_data
from transformer import Transformer
from evaluate import greedy_decode, beam_search_decode

# ============== 可在此处修改配置 ==============
EVAL_CONFIGS = [
    {
        "name": "tst2013",
        "src_xml": "de-en/IWSLT17.TED.tst2013.de-en.en.xml",
        "tgt_xml": "de-en/IWSLT17.TED.tst2013.de-en.de.xml",
    },
]

CHECKPOINT_PATH = "runs/no_pos_encoding/checkpoints/best_model.pt"
USE_RELATIVE_POS = False
USE_POS_ENCODING = Ture
BEAM_SEARCH = True
BEAM_SIZE = 4
MAX_LEN = Config.MAX_DECODE_LEN
OUTPUT_ROOT = "experiments/manual_eval/no_pos_encoding"
# ============================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_model(checkpoint_path: str, use_relative_pos: bool, use_pos_encoding: bool):
    device = Config.DEVICE
    # 仅为了获得词表
    _, _, src_vocab, tgt_vocab = load_iwslt_data(Config)

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
        max_rel_pos=Config.MAX_REL_POS,
        use_positional_encoding=use_pos_encoding,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到检查点: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab


def translate_sent(model: Transformer, sentence: str, src_vocab, tgt_vocab,
                   beam_search: bool, beam_size: int, max_len: int) -> str:
    device = Config.DEVICE
    tokens = src_vocab.encode(sentence, add_bos=False, add_eos=True)
    src = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = torch.ones_like(src)

    if beam_search:
        translation = beam_search_decode(
            model,
            src.squeeze(0),
            src_mask.squeeze(0),
            tgt_vocab,
            beam_size=beam_size,
            max_len=max_len,
            device=device,
        )
    else:
        translation = greedy_decode(
            model,
            src,
            src_mask,
            tgt_vocab,
            max_len=max_len,
            device=device,
        )
    return translation


def translate_batch(model: Transformer, sentences: List[str], src_vocab, tgt_vocab,
                    beam_search: bool, beam_size: int, max_len: int) -> List[str]:
    outputs = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            outputs.append("")
            continue
        outputs.append(
            translate_sent(model, sent, src_vocab, tgt_vocab, beam_search, beam_size, max_len)
        )
    return outputs


def save_list(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")


def evaluate_split(model: Transformer, src_vocab, tgt_vocab, config: dict, output_root: str,
                   beam_search: bool, beam_size: int, max_len: int) -> None:
    name = config["name"]
    out_dir = os.path.join(output_root, name)
    ensure_dir(out_dir)

    print(f"\n{'='*60}")
    print(f"评估数据集: {name}")
    print(f"  源文件: {config['src_xml']}")
    print(f"  目标文件: {config['tgt_xml']}")

    src_sentences = parse_xml_file(config["src_xml"])
    tgt_sentences = parse_xml_file(config["tgt_xml"])

    if len(src_sentences) != len(tgt_sentences):
        raise ValueError(f"源/参考句子数量不一致: {len(src_sentences)} vs {len(tgt_sentences)}")

    print(f"  句子数: {len(src_sentences)}")

    src_txt = os.path.join(out_dir, f"{name}_src.txt")
    ref_txt = os.path.join(out_dir, f"{name}_ref.txt")
    hyp_txt = os.path.join(out_dir, f"{name}_hyp.txt")

    save_list(src_txt, src_sentences)
    save_list(ref_txt, tgt_sentences)

    print("  开始翻译...")
    translations = translate_batch(
        model,
        src_sentences,
        src_vocab,
        tgt_vocab,
        beam_search=beam_search,
        beam_size=beam_size,
        max_len=max_len,
    )
    save_list(hyp_txt, translations)
    print(f"  翻译完成，结果保存至 {hyp_txt}")

    # 计算 BLEU
    bleu_case = corpus_bleu(translations, [tgt_sentences], tokenize="intl", lowercase=False)
    bleu_lc = corpus_bleu(translations, [tgt_sentences], tokenize="intl", lowercase=True)

    result_path = os.path.join(out_dir, "bleu_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"数据集: {name}\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Beam Search: {beam_search}, Beam Size: {beam_size}\n")
        f.write(f"BLEU (大小写敏感): {bleu_case.score:.2f}\n")
        f.write(f"BLEU (大小写不敏感): {bleu_lc.score:.2f}\n")
        f.write(f"BP: {bleu_case.bp:.4f}, ratio: {bleu_case.sys_len/bleu_case.ref_len:.4f}\n")
        f.write("\n")
    print("  BLEU 结果:")
    print(f"    大小写敏感: {bleu_case.score:.2f}")
    print(f"    大小写不敏感: {bleu_lc.score:.2f}")
    print(f"    详细结果已保存: {result_path}")


def main():
    ensure_dir(OUTPUT_ROOT)
    
    model, src_vocab, tgt_vocab = load_model(CHECKPOINT_PATH, USE_RELATIVE_POS, USE_POS_ENCODING)

    print("模型加载完成，开始遍历数据集...\n")
    for cfg in EVAL_CONFIGS:
        evaluate_split(
            model,
            src_vocab,
            tgt_vocab,
            cfg,
            OUTPUT_ROOT,
            beam_search=BEAM_SEARCH,
            beam_size=BEAM_SIZE,
            max_len=MAX_LEN,
        )

    print("\n评估完成。所有输出保存在:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
