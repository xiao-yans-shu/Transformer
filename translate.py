"""
翻译脚本 - 使用训练好的Transformer模型将英语翻译成德语
"""
import os
import sys
from typing import List

import torch

from config import Config
from transformer import Transformer
from data_loader import load_iwslt_data
from evaluate import greedy_decode, beam_search_decode

# ================== 自定义配置 ==================
CHECKPOINT_PATH = "runs/200k/checkpoints/best_model_bleu.pt"
USE_RELATIVE_POS = False  # 如果训练时使用了相对位置编码，这里设为True
BEAM_SEARCH = True
BEAM_SIZE = 4
MAX_DECODE_LEN = Config.MAX_DECODE_LEN

# 要翻译的句子（可以自行修改）
SENTENCES_TO_TRANSLATE = [
    "Today, we have to rethink the way we use technology.",
    "Education is the most powerful weapon which you can use to change the world.",
    "Artificial intelligence may trigger a series of ethical issues.",
    "Yesterday I bought a bicycle. It's very beautiful but very expensive.",
]

# 如果需要批量翻译文件，指定输入输出路径（留空则不处理文件）
INPUT_FILE = ""  # 例如 "data/sample_input.txt"
OUTPUT_FILE = ""  # 例如 "data/sample_output.txt"

# =================================================


def load_model(checkpoint_path: str, use_relative_pos: bool = False):
    """加载训练好的Transformer模型"""
    device = Config.DEVICE
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
        max_rel_pos=Config.MAX_REL_POS
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到检查点文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, src_vocab, tgt_vocab


def translate_sentence(model: Transformer,
                       sentence: str,
                       src_vocab,
                       tgt_vocab,
                       beam_search: bool = True,
                       beam_size: int = 4,
                       max_len: int = 100) -> str:
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
            device=device
        )
    else:
        translation = greedy_decode(
            model,
            src,
            src_mask,
            tgt_vocab,
            max_len=max_len,
            device=device
        )

    return translation


def translate_lines(model: Transformer,
                    sentences: List[str],
                    src_vocab,
                    tgt_vocab,
                    beam_search: bool,
                    beam_size: int,
                    max_len: int) -> List[str]:
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            results.append("")
            continue
        translation = translate_sentence(
            model,
            sentence,
            src_vocab,
            tgt_vocab,
            beam_search=beam_search,
            beam_size=beam_size,
            max_len=max_len
        )
        results.append(translation)
    return results


def main():
    print("加载模型和词汇表...")
    model, src_vocab, tgt_vocab = load_model(CHECKPOINT_PATH, use_relative_pos=USE_RELATIVE_POS)
    print("模型加载完成！\n")

    if SENTENCES_TO_TRANSLATE:
        print("=== 单句翻译 ===")
        for sentence in SENTENCES_TO_TRANSLATE:
            translation = translate_sentence(
                model,
                sentence,
                src_vocab,
                tgt_vocab,
                beam_search=BEAM_SEARCH,
                beam_size=BEAM_SIZE,
                max_len=MAX_DECODE_LEN
            )
            print(f"来源: {sentence}")
            print(f"翻译: {translation}\n")

    if INPUT_FILE:
        if not os.path.exists(INPUT_FILE):
            print(f"输入文件不存在: {INPUT_FILE}")
            sys.exit(1)

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f]

        translations = translate_lines(
            model,
            sentences,
            src_vocab,
            tgt_vocab,
            beam_search=BEAM_SEARCH,
            beam_size=BEAM_SIZE,
            max_len=MAX_DECODE_LEN
        )

        if OUTPUT_FILE:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for translation in translations:
                    f.write(translation + '\n')
            print(f"翻译结果已保存到: {OUTPUT_FILE}")
        else:
            print("=== 文件翻译结果 ===")
            for src_line, tgt_line in zip(sentences, translations):
                print(f"来源: {src_line}\n翻译: {tgt_line}\n")


if __name__ == '__main__':
    main()
